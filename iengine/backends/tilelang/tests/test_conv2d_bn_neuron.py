"""Correctness tests for dense Conv2d+BN+IF/LIF TileLang kernels.

Compares TileLang fused kernels against a PyTorch reference that
performs Conv2d + BN (manual scale/bias) + IF/LIF neuron step-by-step.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------

def ref_conv2d_bn_if(
    data_nchw: torch.Tensor,
    weight_oihw: torch.Tensor,
    state: torch.Tensor,
    bn_scale: torch.Tensor,
    bn_bias: torch.Tensor,
    stride: int, padding: int, dilation: int,
    v_threshold: float = 1.0, v_reset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: Conv2d → BN scale/bias → IF neuron."""
    conv_out = F.conv2d(
        data_nchw.float(), weight_oihw.float(),
        stride=stride, padding=padding, dilation=dilation,
    )
    # BN epilogue: y = conv_out * scale + bias (per output channel)
    bn_out = conv_out * bn_scale.view(1, -1, 1, 1) + bn_bias.view(1, -1, 1, 1)
    # IF integrate: h = v + bn_out
    h = state + bn_out
    # Fire
    spike = (h >= v_threshold).float()
    # Reset
    v_new = (1.0 - spike) * h + spike * v_reset
    return spike, v_new


def ref_conv2d_bn_lif(
    data_nchw: torch.Tensor,
    weight_oihw: torch.Tensor,
    state: torch.Tensor,
    bn_scale: torch.Tensor,
    bn_bias: torch.Tensor,
    stride: int, padding: int, dilation: int,
    v_threshold: float = 1.0, v_reset: float = 0.0,
    recip_tau: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: Conv2d → BN scale/bias → LIF neuron."""
    conv_out = F.conv2d(
        data_nchw.float(), weight_oihw.float(),
        stride=stride, padding=padding, dilation=dilation,
    )
    bn_out = conv_out * bn_scale.view(1, -1, 1, 1) + bn_bias.view(1, -1, 1, 1)
    # LIF integrate: h = (1 - 1/tau)*v + (1/tau)*bn_out
    h = (1.0 - recip_tau) * state + recip_tau * bn_out
    spike = (h >= v_threshold).float()
    v_new = (1.0 - spike) * h + spike * v_reset
    return spike, v_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tensors(N, C, H, W, F, K, S, D, P):
    """Create random input tensors for a Conv2d+BN+Neuron test."""
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    torch.manual_seed(42)
    # NHWC layout for TileLang
    data_nhwc = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
    weight_hwcf = torch.randn(K, K, C, F, device="cuda", dtype=torch.float16) * 0.1
    spikes_nhwf = torch.zeros(N, OH, OW, F, device="cuda", dtype=torch.float16)
    state_nhwf = torch.zeros(N, OH, OW, F, device="cuda", dtype=torch.float32)
    bn_scale = torch.ones(F, device="cuda", dtype=torch.float32) * 0.5
    bn_bias = torch.randn(F, device="cuda", dtype=torch.float32) * 0.1

    # NCHW equivalents for PyTorch reference
    data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous().float()
    weight_oihw = weight_hwcf.permute(3, 2, 0, 1).contiguous().float()
    state_nchw = torch.zeros(N, F, OH, OW, device="cuda", dtype=torch.float32)

    return {
        "tl": (data_nhwc, weight_hwcf, spikes_nhwf, state_nhwf, bn_scale, bn_bias),
        "ref": (data_nchw, weight_oihw, state_nchw, bn_scale, bn_bias),
        "OH": OH, "OW": OW,
    }


# ---------------------------------------------------------------------------
# Test cases — Conv2d + BN + IF
# ---------------------------------------------------------------------------

# (N, C, H, W, F, K, S, D, P, description)
CONV_CONFIGS = [
    (1, 64, 56, 56, 64, 3, 1, 1, 1, "interior_3x3"),
    (2, 64, 56, 56, 128, 3, 2, 1, 1, "downsample_3x3"),
    (1, 128, 28, 28, 128, 3, 1, 1, 1, "mid_3x3"),
    (1, 256, 14, 14, 64, 1, 1, 1, 0, "bottleneck_1x1"),
    (1, 3, 32, 32, 64, 3, 1, 1, 1, "cifar_first"),
]


class TestConv2dBnIF:
    """Test dense Conv2d + BN + IF neuron kernel."""

    @pytest.mark.parametrize(
        "N,C,H,W,F,K,S,D,P,desc", CONV_CONFIGS,
        ids=[c[-1] for c in CONV_CONFIGS],
    )
    def test_single_timestep(self, N, C, H, W, F, K, S, D, P, desc):
        from iengine.backends.tilelang.conv2d_bn_neuron import conv2d_bn_if_kernel

        tensors = _make_tensors(N, C, H, W, F, K, S, D, P)
        data_nhwc, weight_hwcf, spikes_buf, state_buf, bn_scale, bn_bias = tensors["tl"]
        data_nchw, weight_oihw, state_nchw, bn_scale_ref, bn_bias_ref = tensors["ref"]

        # Build & run TileLang kernel
        kernel = conv2d_bn_if_kernel(
            N=N, C_in=C, H=H, W=W, F=F, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
            v_threshold=1.0, v_reset=0.0,
        )
        # state is modified in-place; spikes is returned
        spikes_tl = kernel(
            data_nhwc, weight_hwcf, state_buf, bn_scale, bn_bias,
        )
        state_tl = state_buf  # in-place update

        # PyTorch reference
        spikes_ref, state_ref = ref_conv2d_bn_if(
            data_nchw, weight_oihw, state_nchw, bn_scale_ref, bn_bias_ref,
            stride=S, padding=P, dilation=D,
        )

        # Compare (NHWC → NCHW for comparison)
        spikes_tl_nchw = spikes_tl.permute(0, 3, 1, 2).float()
        state_tl_nchw = state_tl.permute(0, 3, 1, 2).float()

        torch.testing.assert_close(
            spikes_tl_nchw, spikes_ref, rtol=1e-2, atol=1e-2,
            msg=f"Spike mismatch for {desc}",
        )
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=5e-2,
            msg=f"State mismatch for {desc}",
        )

    def test_temporal_state_persistence(self):
        """Run T=4 timesteps, verify membrane state evolves correctly."""
        from iengine.backends.tilelang.conv2d_bn_neuron import conv2d_bn_if_kernel

        N, C, H, W, F, K, S, D, P = 1, 64, 16, 16, 64, 3, 1, 1, 1
        T_steps = 4
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        torch.manual_seed(123)
        weight_hwcf = torch.randn(K, K, C, F, device="cuda", dtype=torch.float16) * 0.05
        weight_oihw = weight_hwcf.permute(3, 2, 0, 1).contiguous().float()
        bn_scale = torch.ones(F, device="cuda", dtype=torch.float32) * 0.3
        bn_bias = torch.zeros(F, device="cuda", dtype=torch.float32)

        kernel = conv2d_bn_if_kernel(
            N=N, C_in=C, H=H, W=W, F=F, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
            v_threshold=1.0, v_reset=0.0,
        )

        # TileLang path — accumulate state across timesteps
        state_tl = torch.zeros(N, OH, OW, F, device="cuda", dtype=torch.float32)
        # Reference path
        state_ref = torch.zeros(N, F, OH, OW, device="cuda", dtype=torch.float32)

        for t in range(T_steps):
            torch.manual_seed(1000 + t)
            data_nhwc = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
            data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous().float()

            kernel(
                data_nhwc, weight_hwcf, state_tl, bn_scale, bn_bias,
            )

            _, state_ref = ref_conv2d_bn_if(
                data_nchw, weight_oihw, state_ref, bn_scale, bn_bias,
                stride=S, padding=P, dilation=D,
            )

        state_tl_nchw = state_tl.permute(0, 3, 1, 2).float()
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=0.1,
            msg="State diverged after T=4 timesteps",
        )


# ---------------------------------------------------------------------------
# Test cases — Conv2d + BN + LIF
# ---------------------------------------------------------------------------

class TestConv2dBnLIF:
    """Test dense Conv2d + BN + LIF neuron kernel."""

    @pytest.mark.parametrize(
        "N,C,H,W,F,K,S,D,P,desc", CONV_CONFIGS[:3],
        ids=[c[-1] for c in CONV_CONFIGS[:3]],
    )
    def test_single_timestep(self, N, C, H, W, F, K, S, D, P, desc):
        from iengine.backends.tilelang.conv2d_bn_neuron import conv2d_bn_lif_kernel

        tensors = _make_tensors(N, C, H, W, F, K, S, D, P)
        data_nhwc, weight_hwcf, spikes_buf, state_buf, bn_scale, bn_bias = tensors["tl"]
        data_nchw, weight_oihw, state_nchw, bn_scale_ref, bn_bias_ref = tensors["ref"]

        recip_tau = 0.5
        kernel = conv2d_bn_lif_kernel(
            N=N, C_in=C, H=H, W=W, F=F, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
            v_threshold=1.0, v_reset=0.0, recip_tau=recip_tau,
        )
        spikes_tl = kernel(
            data_nhwc, weight_hwcf, state_buf, bn_scale, bn_bias,
        )
        state_tl = state_buf

        spikes_ref, state_ref = ref_conv2d_bn_lif(
            data_nchw, weight_oihw, state_nchw, bn_scale_ref, bn_bias_ref,
            stride=S, padding=P, dilation=D,
            recip_tau=recip_tau,
        )

        spikes_tl_nchw = spikes_tl.permute(0, 3, 1, 2).float()
        state_tl_nchw = state_tl.permute(0, 3, 1, 2).float()

        torch.testing.assert_close(
            spikes_tl_nchw, spikes_ref, rtol=1e-2, atol=1e-2,
            msg=f"LIF spike mismatch for {desc}",
        )
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=5e-2,
            msg=f"LIF state mismatch for {desc}",
        )

    def test_temporal_state_persistence(self):
        """Run T=4 with LIF leak, verify membrane decays correctly."""
        from iengine.backends.tilelang.conv2d_bn_neuron import conv2d_bn_lif_kernel

        N, C, H, W, F, K, S, D, P = 1, 32, 16, 16, 32, 3, 1, 1, 1
        T_steps = 4
        recip_tau = 0.5
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

        torch.manual_seed(456)
        weight_hwcf = torch.randn(K, K, C, F, device="cuda", dtype=torch.float16) * 0.05
        weight_oihw = weight_hwcf.permute(3, 2, 0, 1).contiguous().float()
        bn_scale = torch.ones(F, device="cuda", dtype=torch.float32) * 0.3
        bn_bias = torch.zeros(F, device="cuda", dtype=torch.float32)

        kernel = conv2d_bn_lif_kernel(
            N=N, C_in=C, H=H, W=W, F=F, K=K, S=S, D=D, P=P,
            block_M=64, block_N=64, block_K=32,
            num_stages=2, threads=128,
            v_threshold=1.0, v_reset=0.0, recip_tau=recip_tau,
        )

        state_tl = torch.zeros(N, OH, OW, F, device="cuda", dtype=torch.float32)
        state_ref = torch.zeros(N, F, OH, OW, device="cuda", dtype=torch.float32)

        for t in range(T_steps):
            torch.manual_seed(2000 + t)
            data_nhwc = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
            data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous().float()

            kernel(
                data_nhwc, weight_hwcf, state_tl, bn_scale, bn_bias,
            )
            _, state_ref = ref_conv2d_bn_lif(
                data_nchw, weight_oihw, state_ref, bn_scale, bn_bias,
                stride=S, padding=P, dilation=D,
                recip_tau=recip_tau,
            )

        state_tl_nchw = state_tl.permute(0, 3, 1, 2).float()
        torch.testing.assert_close(
            state_tl_nchw, state_ref, rtol=1e-2, atol=0.1,
            msg="LIF state diverged after T=4 timesteps",
        )
