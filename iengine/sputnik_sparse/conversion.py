"""Sputnik sparse forward hooks for Linear and SSA attention.

Uses Google Research's Sputnik CUDA SpMM kernels to accelerate
sparse SNN activations through Linear layers and attention Q@K^T.
"""

import torch
import torch.nn as nn

from iengine.common.density import measure_density, should_use_sparse
from iengine.common.stats import LinearStats, AttentionStats
from .kernels import check_sputnik, require_sputnik, to_sputnik_csr, sputnik_spmm


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def make_sparse_linear_forward(module, original_forward, layer_name, stats,
                                density_threshold=0.15, min_elements=4096,
                                enabled_ref=None):
    """Create a Sputnik SpMM replacement for nn.Linear forward.

    Args:
        module:            nn.Linear module.
        original_forward:  Dense fallback.
        layer_name:        Name for stats.
        stats:             LinearStats instance.
        density_threshold: Max density for sparse path.
        min_elements:      Min tensor elements for sparse path.
        enabled_ref:       List [bool] for runtime toggle.
    """
    require_sputnik()

    def sparse_forward(x):
        if enabled_ref and not enabled_ref[0]:
            return original_forward(x)

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape
        N = module.weight.shape[0]
        ops = M * K * N
        density = measure_density(x_2d)

        if not should_use_sparse(x_2d, threshold=density_threshold,
                                 min_elements=min_elements):
            stats.record(layer_name, ops=ops, density=1.0, used_sparse=False)
            return original_forward(x)

        try:
            result = sputnik_spmm(x_2d, module.weight, module.bias)
            stats.record(layer_name, ops=ops, density=density, used_sparse=True)
            return result.reshape(orig_shape[:-1] + (N,))
        except Exception:
            stats.record(layer_name, ops=ops, density=1.0, used_sparse=False)
            return original_forward(x)

    return sparse_forward


# ---------------------------------------------------------------------------
# SSA Attention (sparse Q @ K^T via batched Sputnik SpMM)
# ---------------------------------------------------------------------------

def _batched_sputnik_spmm(sparse_batch, dense_batch):
    """Batched SpMM: sparse_batch[i] @ dense_batch[i] for each slice."""
    ts = require_sputnik()
    B, M, K = sparse_batch.shape
    N = dense_batch.shape[2]

    outputs = []
    for i in range(B):
        s2d = sparse_batch[i]
        if s2d.count_nonzero().item() == 0:
            outputs.append(torch.zeros(M, N, device=s2d.device, dtype=s2d.dtype))
            continue
        try:
            vals, row_idx, row_off, col_idx, nnz = to_sputnik_csr(s2d)
            result = ts.spmm(M, K, N, nnz, row_idx, vals, row_off,
                              col_idx, dense_batch[i].contiguous())
            outputs.append(result)
        except Exception:
            outputs.append(s2d @ dense_batch[i])

    return torch.stack(outputs, dim=0)


def make_sparse_ssa_forward(ssa_module, stats, layer_name,
                            density_threshold=0.15, min_elements=4096,
                            enabled_ref=None):
    """Create a Sputnik sparse replacement for SSA.forward.

    Applies Sputnik SpMM to the Q @ K^T attention matmul where Q is a
    binary spike tensor (~6-8% density).

    Args:
        ssa_module:        SSA module instance.
        stats:             AttentionStats instance.
        layer_name:        Name for stats.
        density_threshold: Max density for sparse path.
        min_elements:      Min elements for sparse path.
        enabled_ref:       List [bool] for runtime toggle.
    """
    require_sputnik()

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
        D = C // num_heads
        x_tb = x.flatten(0, 1)

        q = q_lif(q_bn(q_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        q = q.reshape(T, B, N, num_heads, D).permute(0, 1, 3, 2, 4).contiguous()

        k = k_lif(k_bn(k_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        k = k.reshape(T, B, N, num_heads, D).permute(0, 1, 3, 2, 4).contiguous()

        v = v_lif(v_bn(v_linear(x_tb).transpose(-1, -2)).transpose(-1, -2)
                  .reshape(T, B, N, C).contiguous())
        v = v.reshape(T, B, N, num_heads, D).permute(0, 1, 3, 2, 4).contiguous()

        TBH = T * B * num_heads
        q_flat = q.reshape(TBH, N, D)
        k_flat = k.reshape(TBH, N, D)
        v_flat = v.reshape(TBH, N, D)

        q_density = measure_density(q_flat)
        use_sparse = (enabled_ref is None or enabled_ref[0]) and \
                     should_use_sparse(q_flat, threshold=density_threshold,
                                       min_elements=min_elements) and \
                     q_flat.is_cuda

        if use_sparse:
            try:
                k_t = k_flat.transpose(-2, -1).contiguous()
                attn = _batched_sputnik_spmm(q_flat, k_t) * scale
                stats.record_qk(layer_name, ops=TBH * N * N,
                                density=q_density, used_sparse=True)
            except Exception:
                attn = (q_flat @ k_flat.transpose(-2, -1)) * scale
                stats.record_qk(layer_name, ops=TBH * N * N,
                                density=q_density, used_sparse=False)
        else:
            attn = (q_flat @ k_flat.transpose(-2, -1)) * scale
            stats.record_qk(layer_name, ops=TBH * N * N,
                            density=q_density, used_sparse=False)

        # attn @ v → reshape → attn_lif → proj
        x_out = (attn @ v_flat).reshape(T, B, num_heads, N, D)
        x_out = x_out.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x_out = attn_lif(x_out)
        x_out = proj_lif(
            proj_bn(proj_linear(x_out.flatten(0, 1)).transpose(-1, -2))
            .transpose(-1, -2).reshape(T, B, N, C))
        return x_out

    return sparse_forward
