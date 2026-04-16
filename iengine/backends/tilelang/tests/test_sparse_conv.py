"""Correctness tests for 2:4 sparse Conv2d+BN+IF TileLang kernel.

Creates synthetic 2:4 sparse weights, compresses them via
weight_compress, runs the sparse kernel, and compares against a dense
PyTorch reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available",
)


def _make_24_sparse_weight(F_out, C_in, KH, KW):
    """Create a random Conv weight with 2:4 structured sparsity.

    The sparsity pattern is applied along the KH*KW*C_in dimension
    (im2col k-index ordering: KH outer, C_in inner), matching the
    compression in ``weight_compress.compress_conv_weight()``.

    Returns weight in PyTorch OIHW layout ``(F_out, C_in, KH, KW)``.
    """
    K_red = C_in * KH * KW
    # Work in (F, KH, KW, C) = im2col k-ordering
    w = torch.randn(F_out, K_red, device="cuda", dtype=torch.float16) * 0.1
    # Enforce 2:4: in each group of 4 along K, zero out the 2 smallest
    w_grouped = w.reshape(F_out, K_red // 4, 4)
    _, indices = w_grouped.abs().topk(2, dim=-1, largest=False)
    mask = torch.ones_like(w_grouped, dtype=torch.bool)
    mask.scatter_(-1, indices, False)
    w_grouped = w_grouped * mask
    # Reshape back to (F, KH, KW, C), then to PyTorch OIHW
    w_hwc = w_grouped.reshape(F_out, KH, KW, C_in)
    return w_hwc.permute(0, 3, 1, 2).contiguous()  # (F, C_in, KH, KW)


def ref_conv2d_bn_if(data_nchw, weight_oihw, state_nchw, bn_scale, bn_bias,
                      stride, padding, dilation, v_threshold=1.0, v_reset=0.0):
    conv_out = F.conv2d(data_nchw.float(), weight_oihw.float(),
                        stride=stride, padding=padding, dilation=dilation)
    bn_out = conv_out * bn_scale.view(1, -1, 1, 1) + bn_bias.view(1, -1, 1, 1)
    h = state_nchw + bn_out
    spike = (h >= v_threshold).float()
    v_new = (1.0 - spike) * h + spike * v_reset
    return spike, v_new


# (N, C, H, W, F, K, S, D, P, desc)
SPARSE_CONFIGS = [
    (1, 64, 16, 16, 64, 3, 1, 1, 1, "interior_3x3"),
    (1, 128, 8, 8, 128, 3, 1, 1, 1, "mid_3x3"),
    (1, 64, 16, 16, 128, 1, 1, 1, 0, "pointwise_1x1"),
]


class TestSparseConv2dBnIF:

    @pytest.mark.parametrize(
        "N,C,H,W,F_out,K,S,D,P,desc", SPARSE_CONFIGS,
        ids=[c[-1] for c in SPARSE_CONFIGS],
    )
    def test_correctness(self, N, C, H, W, F_out, K, S, D, P, desc):
        from iengine.backends.tilelang.conv2d_bn_neuron_sparse import (
            conv2d_bn_if_sparse_kernel,
        )
        from tilelang.utils.sparse import compress as _tl_compress
        from tilelang.contrib import nvcc
        arch = nvcc.get_target_compute_version()
        def compress_conv_weight(w4d, block_k=64):
            F_o, C_i, KH, KW = w4d.shape
            w_hwc = w4d.permute(0, 2, 3, 1).contiguous().reshape(F_o, -1)
            return _tl_compress(w_hwc, transposed=False, block_k=block_k, arch=arch)

        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        torch.manual_seed(42)
        # NHWC for TileLang
        data_nhwc = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
        # Create 2:4 sparse weight (PyTorch OIHW layout)
        weight_oihw = _make_24_sparse_weight(F_out, C, K, K)
        bn_scale = torch.ones(F_out, device="cuda", dtype=torch.float32) * 0.5
        bn_bias = torch.randn(F_out, device="cuda", dtype=torch.float32) * 0.1
        state_nhwf = torch.zeros(N, OH, OW, F_out, device="cuda", dtype=torch.float32)

        # Compress weight for T.gemm_sp
        block_K = 64
        W_sp, E_meta = compress_conv_weight(weight_oihw, block_k=block_K)

        # Build & run sparse kernel
        kernel = conv2d_bn_if_sparse_kernel(
            N=N, C_in=C, H=H, W=W, F=F_out, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=block_K,
            num_stages=1, threads=128,
            v_threshold=1.0, v_reset=0.0,
        )
        spikes_tl = kernel(
            data_nhwc, W_sp, E_meta, state_nhwf, bn_scale, bn_bias,
        )

        # PyTorch dense reference (on the same sparse weight — zeros included)
        data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous().float()
        state_nchw = torch.zeros(N, F_out, OH, OW, device="cuda", dtype=torch.float32)
        spikes_ref, state_ref = ref_conv2d_bn_if(
            data_nchw, weight_oihw, state_nchw, bn_scale, bn_bias,
            stride=S, padding=P, dilation=D,
        )

        spikes_tl_nchw = spikes_tl.permute(0, 3, 1, 2).float()
        state_tl_nchw = state_nhwf.permute(0, 3, 1, 2).float()

        torch.testing.assert_close(
            spikes_tl_nchw, spikes_ref, rtol=1e-2, atol=1e-2,
            msg=f"Sparse spike mismatch for {desc}",
        )
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=0.1,
            msg=f"Sparse state mismatch for {desc}",
        )

    def test_temporal_persistence(self):
        """Run T=4 with sparse weights, verify state evolves correctly."""
        from iengine.backends.tilelang.conv2d_bn_neuron_sparse import (
            conv2d_bn_if_sparse_kernel,
        )
        from tilelang.utils.sparse import compress as _tl_compress
        from tilelang.contrib import nvcc
        arch = nvcc.get_target_compute_version()
        def compress_conv_weight(w4d, block_k=64):
            F_o, C_i, KH, KW = w4d.shape
            w_hwc = w4d.permute(0, 2, 3, 1).contiguous().reshape(F_o, -1)
            return _tl_compress(w_hwc, transposed=False, block_k=block_k, arch=arch)

        N, C, H, W, F_out, K, S, D, P = 1, 64, 8, 8, 64, 3, 1, 1, 1
        T_steps = 4
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        torch.manual_seed(99)
        weight_oihw = _make_24_sparse_weight(F_out, C, K, K)
        bn_scale = torch.ones(F_out, device="cuda", dtype=torch.float32) * 0.3
        bn_bias = torch.zeros(F_out, device="cuda", dtype=torch.float32)

        block_K = 64
        W_sp, E_meta = compress_conv_weight(weight_oihw, block_k=block_K)

        kernel = conv2d_bn_if_sparse_kernel(
            N=N, C_in=C, H=H, W=W, F=F_out, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=block_K,
            num_stages=1, threads=128,
        )

        state_tl = torch.zeros(N, OH, OW, F_out, device="cuda", dtype=torch.float32)
        state_ref = torch.zeros(N, F_out, OH, OW, device="cuda", dtype=torch.float32)

        for t in range(T_steps):
            torch.manual_seed(3000 + t)
            data_nhwc = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
            data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous().float()

            kernel(data_nhwc, W_sp, E_meta, state_tl, bn_scale, bn_bias)

            _, state_ref = ref_conv2d_bn_if(
                data_nchw, weight_oihw, state_ref, bn_scale, bn_bias,
                stride=S, padding=P, dilation=D,
            )

        state_tl_nchw = state_tl.permute(0, 3, 1, 2).float()
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=0.15,
            msg="Sparse state diverged after T=4",
        )
