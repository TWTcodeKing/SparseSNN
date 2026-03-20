"""CSR sparse forward hooks for Linear, Conv2d, and SSA attention.

Converts sparse SNN activations to CSR format on-the-fly and uses
torch.sparse.mm instead of dense matmul when density is below threshold.

Note: Per-call CSR conversion adds significant overhead. This backend
serves as a research baseline, not production inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from iengine.common.density import measure_density, should_use_sparse, to_sparse_csr_2d
from iengine.common.stats import LinearStats, AttentionStats


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def make_sparse_linear_forward(module, original_forward, layer_name, stats,
                                density_threshold=0.15, enabled_ref=None):
    """Create a replacement forward for nn.Linear using CSR sparse matmul.

    Args:
        module:            The nn.Linear module.
        original_forward:  Original forward to fall back to.
        layer_name:        Name for stats tracking.
        stats:             LinearStats instance.
        density_threshold: Max density for sparse path.
        enabled_ref:       List [bool] for runtime toggle.
    """
    weight_t = module.weight.t().contiguous()
    bias = module.bias
    out_features = module.weight.shape[0]

    def sparse_forward(x):
        if enabled_ref is not None and not enabled_ref[0]:
            return original_forward(x)

        if not should_use_sparse(x, threshold=density_threshold):
            stats.record(layer_name, ops=x.shape[-1] * out_features,
                         density=1.0, used_sparse=False)
            return original_forward(x)

        density = measure_density(x)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        try:
            result = torch.sparse.mm(to_sparse_csr_2d(x_2d), weight_t)
            if bias is not None:
                result = result + bias
            stats.record(layer_name, ops=x.shape[-1] * out_features,
                         density=density, used_sparse=True)
            return result.reshape(orig_shape[:-1] + (out_features,))
        except Exception:
            stats.record(layer_name, ops=x.shape[-1] * out_features,
                         density=1.0, used_sparse=False)
            return original_forward(x)

    return sparse_forward


# ---------------------------------------------------------------------------
# Conv2d (im2col + CSR sparse GEMM)
# ---------------------------------------------------------------------------

def make_sparse_conv2d_forward(module, original_forward, layer_name, stats,
                                density_threshold=0.15, enabled_ref=None):
    """Create a replacement forward for nn.Conv2d using im2col + CSR sparse.

    Args:
        module:            The nn.Conv2d module (groups=1 only).
        original_forward:  Original forward to fall back to.
        layer_name:        Name for stats tracking.
        stats:             LinearStats instance.
        density_threshold: Max density for sparse path.
        enabled_ref:       List [bool] for runtime toggle.
    """
    weight_2d = module.weight.reshape(module.out_channels, -1).t().contiguous()
    bias = module.bias
    out_channels = module.out_channels
    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    dilation = module.dilation

    def sparse_forward(x):
        if enabled_ref is not None and not enabled_ref[0]:
            return original_forward(x)

        if not should_use_sparse(x, threshold=density_threshold):
            ops = x.shape[0] * out_channels * x.shape[2] * x.shape[3]
            stats.record(layer_name, ops=ops, density=1.0, used_sparse=False)
            return original_forward(x)

        density = measure_density(x)
        B = x.shape[0]

        try:
            x_unf = F.unfold(x, kernel_size, dilation=dilation,
                              padding=padding, stride=stride)
            L, K = x_unf.shape[2], x_unf.shape[1]

            x_2d = x_unf.permute(0, 2, 1).reshape(B * L, K)
            out_2d = torch.sparse.mm(to_sparse_csr_2d(x_2d), weight_2d)

            if bias is not None:
                out_2d = out_2d + bias

            output = out_2d.reshape(B, L, out_channels).permute(0, 2, 1)
            H_out = (x.shape[2] + 2 * padding[0] - dilation[0]
                     * (kernel_size[0] - 1) - 1) // stride[0] + 1
            W_out = (x.shape[3] + 2 * padding[1] - dilation[1]
                     * (kernel_size[1] - 1) - 1) // stride[1] + 1

            stats.record(layer_name, ops=B * out_channels * L,
                         density=density, used_sparse=True)
            return output.reshape(B, out_channels, H_out, W_out)

        except Exception:
            ops = x.shape[0] * out_channels * x.shape[2] * x.shape[3]
            stats.record(layer_name, ops=ops, density=1.0, used_sparse=False)
            return original_forward(x)

    return sparse_forward


# ---------------------------------------------------------------------------
# SSA Attention (sparse Q @ K^T)
# ---------------------------------------------------------------------------

def _sparse_batched_matmul(a, b, threshold=0.15):
    """Sparse batched matmul: (T,B,H,M,K) @ (T,B,H,K,N) via per-slice CSR."""
    T, B, H, M, K = a.shape
    N = b.shape[-1]

    if not should_use_sparse(a, threshold=threshold):
        return a @ b, False, 1.0

    density = (a != 0).float().mean().item()
    batch = T * B * H
    a_flat = a.reshape(batch, M, K)
    b_flat = b.reshape(batch, K, N)

    try:
        results = []
        for i in range(batch):
            results.append(torch.sparse.mm(to_sparse_csr_2d(a_flat[i]), b_flat[i]))
        return torch.stack(results).reshape(T, B, H, M, N), True, density
    except Exception:
        return a @ b, False, 1.0


def make_sparse_ssa_forward(ssa_module, stats, layer_name,
                            density_threshold=0.15, enabled_ref=None):
    """Create a replacement forward for SSA with sparse Q @ K^T.

    Args:
        ssa_module:        The SSA module instance.
        stats:             AttentionStats instance.
        layer_name:        Name for stats tracking.
        density_threshold: Max density for sparse path.
        enabled_ref:       List [bool] for runtime toggle.
    """
    q_linear = ssa_module.q_linear
    q_bn = ssa_module.q_bn
    q_lif = ssa_module.q_lif
    k_linear = ssa_module.k_linear
    k_bn = ssa_module.k_bn
    k_lif = ssa_module.k_lif
    v_linear = ssa_module.v_linear
    v_bn = ssa_module.v_bn
    v_lif = ssa_module.v_lif
    attn_lif = ssa_module.attn_lif
    proj_linear = ssa_module.proj_linear
    proj_bn = ssa_module.proj_bn
    proj_lif = ssa_module.proj_lif
    num_heads = ssa_module.num_heads
    scale = ssa_module.scale

    def sparse_forward(x):
        T, B, N, C = x.shape
        d_head = C // num_heads
        x_tb = x.flatten(0, 1)

        q = q_lif(q_bn(q_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        q = q.reshape(T, B, N, num_heads, d_head).permute(0, 1, 3, 2, 4).contiguous()

        k = k_lif(k_bn(k_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        k = k.reshape(T, B, N, num_heads, d_head).permute(0, 1, 3, 2, 4).contiguous()

        v = v_lif(v_bn(v_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        v = v.reshape(T, B, N, num_heads, d_head).permute(0, 1, 3, 2, 4).contiguous()

        if enabled_ref is not None and not enabled_ref[0]:
            attn = (q @ k.transpose(-2, -1)) * scale
        else:
            attn_raw, qk_sparse, qk_density = _sparse_batched_matmul(
                q, k.transpose(-2, -1).contiguous(), threshold=density_threshold)
            attn = attn_raw * scale
            stats.record_qk(layer_name, ops=T * B * num_heads * N * N,
                            density=qk_density, used_sparse=qk_sparse)

        x = (attn @ v).transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = attn_lif(x)
        x = proj_lif(
            proj_bn(proj_linear(x.flatten(0, 1)).transpose(-1, -2))
            .transpose(-1, -2).reshape(T, B, N, C))
        return x

    return sparse_forward
