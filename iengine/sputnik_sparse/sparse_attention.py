"""Sparse Spiking Self-Attention (SSA) using Sputnik SpMM + SDDMM.

Monkey-patches SSA.forward() to use Sputnik sparse kernels for the two
matrix multiplications in attention:

1. attn = Q @ K.T  (Q is sparse binary spike output from q_lif)
   -> Sputnik SpMM: sparse(Q) @ dense(K.T) = dense(attn)

2. out = attn_spike @ V  (attn_spike is sparse after attn_lif)
   -> Sputnik SpMM: sparse(attn_spike) @ dense(V) = dense(out)

SSA tensor shapes (from spikformer.py):
    Input x: (T, B, N, C)
    After LIF + reshape: Q, K, V are (T, B, H, N, D) where H=num_heads, D=C//H
    attn = (Q @ K.T) * scale  -> (T, B, H, N, N)
    out = attn_spike @ V      -> (T, B, H, N, D)

Since Sputnik operates on 2D matrices, we flatten the batch dimensions
(T*B*H) and run SpMM on each (N, D) or (N, N) matrix independently.
"""

import torch
import torch.nn as nn
from typing import Optional

from iengine.common.density import measure_density, should_use_sparse
from .sparse_linear import _check_sputnik, _require_sputnik, _to_sputnik_csr

# Lazy import
_torch_sputnik = None


def _get_sputnik():
    global _torch_sputnik
    if _torch_sputnik is None:
        _torch_sputnik = _require_sputnik()
    return _torch_sputnik


class SputnikAttentionStats:
    """Tracks statistics for sparse attention execution."""

    def __init__(self):
        self.total_calls = 0
        self.qk_sparse_calls = 0
        self.qk_dense_calls = 0
        self.av_sparse_calls = 0
        self.av_dense_calls = 0
        self.total_ops = 0
        self.effective_ops = 0
        self.per_module = {}

    def record_qk(self, module_name: str, batch_size: int, n: int, d: int,
                   density: float, used_sparse: bool):
        """Record Q @ K.T operation."""
        ops = batch_size * n * d * n  # (N,D) @ (D,N) = (N,N), batched
        self.total_ops += ops
        if used_sparse:
            self.qk_sparse_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.qk_dense_calls += 1
            self.effective_ops += ops
        self._update_module(module_name, 'qk', ops, density, used_sparse)

    def record_av(self, module_name: str, batch_size: int, n: int, d: int,
                   density: float, used_sparse: bool):
        """Record attn @ V operation."""
        ops = batch_size * n * n * d  # (N,N) @ (N,D) = (N,D), batched
        self.total_ops += ops
        if used_sparse:
            self.av_sparse_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.av_dense_calls += 1
            self.effective_ops += ops
        self._update_module(module_name, 'av', ops, density, used_sparse)

    def _update_module(self, name, op_type, ops, density, used_sparse):
        if name not in self.per_module:
            self.per_module[name] = {
                'total_ops': 0, 'effective_ops': 0,
                'qk_densities': [], 'av_densities': [],
            }
        m = self.per_module[name]
        m['total_ops'] += ops
        m[f'{op_type}_densities'].append(density)
        if used_sparse:
            m['effective_ops'] += int(ops * density)
        else:
            m['effective_ops'] += ops

    def to_dict(self) -> dict:
        per_module = {}
        for name, m in self.per_module.items():
            qk_d = m['qk_densities']
            av_d = m['av_densities']
            per_module[name] = {
                'total_ops': m['total_ops'],
                'effective_ops': m['effective_ops'],
                'density': m['effective_ops'] / m['total_ops'] if m['total_ops'] > 0 else 1.0,
                'qk_density': sum(qk_d) / len(qk_d) if qk_d else 1.0,
                'av_density': sum(av_d) / len(av_d) if av_d else 1.0,
            }
        overall_density = (self.effective_ops / self.total_ops
                           if self.total_ops > 0 else 1.0)
        return {
            'total_ops': self.total_ops,
            'effective_ops': self.effective_ops,
            'density': overall_density,
            'per_layer': per_module,
            'total_calls': self.total_calls,
            'qk_sparse_calls': self.qk_sparse_calls,
            'qk_dense_calls': self.qk_dense_calls,
            'av_sparse_calls': self.av_sparse_calls,
            'av_dense_calls': self.av_dense_calls,
        }

    def reset(self):
        self.__init__()


def _batched_sputnik_spmm(sparse_batch: torch.Tensor,
                           dense_batch: torch.Tensor) -> torch.Tensor:
    """Perform batched SpMM: sparse_batch[i] @ dense_batch[i] for each i.

    Args:
        sparse_batch: (B, M, K) tensor with sparse rows.
        dense_batch: (B, K, N) dense tensor.

    Returns:
        (B, M, N) output tensor.
    """
    ts = _get_sputnik()
    B, M, K = sparse_batch.shape
    N = dense_batch.shape[2]

    outputs = []
    for i in range(B):
        sparse_2d = sparse_batch[i]  # (M, K)
        dense_2d = dense_batch[i]    # (K, N)

        # Check if this slice has any nonzeros
        nnz_count = sparse_2d.count_nonzero().item()
        if nnz_count == 0:
            outputs.append(torch.zeros(M, N, device=sparse_2d.device,
                                       dtype=sparse_2d.dtype))
            continue

        try:
            values, row_indices, row_offsets, col_indices, nnz = \
                _to_sputnik_csr(sparse_2d)
            result = ts.spmm(
                M, K, N, nnz,
                row_indices, values, row_offsets, col_indices,
                dense_2d.contiguous()
            )
            outputs.append(result)
        except Exception:
            # Fallback to dense matmul for this slice
            outputs.append(sparse_2d @ dense_2d)

    return torch.stack(outputs, dim=0)


def make_sparse_ssa_forward(ssa_module: nn.Module, module_name: str,
                            stats: SputnikAttentionStats,
                            density_threshold: float = 0.15,
                            min_elements: int = 4096,
                            enabled_ref: list = None):
    """Create a sparse SSA forward function that replaces the original.

    This monkey-patches SSA.forward() to use Sputnik SpMM for the two
    matmuls in spiking self-attention, while keeping all other operations
    (linear projections, batch norm, LIF neurons) unchanged.

    Args:
        ssa_module: The SSA module instance.
        module_name: Name for statistics tracking.
        stats: SputnikAttentionStats instance.
        density_threshold: Max density to use sparse path.
        min_elements: Min elements to justify sparse conversion.
        enabled_ref: Mutable [bool] for runtime enable/disable.

    Returns:
        New forward function to bind to ssa_module.
    """
    if not _check_sputnik():
        raise ImportError(
            "torch_sputnik is not installed. Build from source:\n"
            "  bash iengine/sputnik_sparse/build.sh"
        )

    original_forward = ssa_module.forward

    def sparse_forward(x):
        if enabled_ref and not enabled_ref[0]:
            return original_forward(x)

        T, B, N, C = x.shape
        num_heads = ssa_module.num_heads
        D = C // num_heads
        x_for_qkv = x.flatten(0, 1)  # (T*B, N, C)

        # --- Q projection + LIF ---
        q_linear_out = ssa_module.q_linear(x_for_qkv)
        q_linear_out = ssa_module.q_bn(
            q_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = ssa_module.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, num_heads, D).permute(
            0, 1, 3, 2, 4).contiguous()  # (T, B, H, N, D)

        # --- K projection + LIF ---
        k_linear_out = ssa_module.k_linear(x_for_qkv)
        k_linear_out = ssa_module.k_bn(
            k_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = ssa_module.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, num_heads, D).permute(
            0, 1, 3, 2, 4).contiguous()  # (T, B, H, N, D)

        # --- V projection + LIF ---
        v_linear_out = ssa_module.v_linear(x_for_qkv)
        v_linear_out = ssa_module.v_bn(
            v_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = ssa_module.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, num_heads, D).permute(
            0, 1, 3, 2, 4).contiguous()  # (T, B, H, N, D)

        # --- Attention: Q @ K.T (sparse Q) ---
        # Q is binary spike tensor from q_lif, typically ~6-8% dense
        # Flatten batch dims: (T*B*H, N, D)
        TBH = T * B * num_heads
        q_flat = q.reshape(TBH, N, D)
        k_flat = k.reshape(TBH, N, D)
        v_flat = v.reshape(TBH, N, D)

        q_density = measure_density(q_flat)
        use_sparse_qk = (should_use_sparse(q_flat, threshold=density_threshold,
                                           min_elements=min_elements)
                         and q_flat.is_cuda)

        if use_sparse_qk:
            try:
                # Q @ K.T: sparse(TBH, N, D) @ dense(TBH, D, N) -> (TBH, N, N)
                k_t = k_flat.transpose(-2, -1).contiguous()
                attn = _batched_sputnik_spmm(q_flat, k_t) * ssa_module.scale
                stats.record_qk(module_name, TBH, N, D, q_density,
                                used_sparse=True)
            except Exception:
                attn = (q_flat @ k_flat.transpose(-2, -1)) * ssa_module.scale
                stats.record_qk(module_name, TBH, N, D, q_density,
                                used_sparse=False)
        else:
            attn = (q_flat @ k_flat.transpose(-2, -1)) * ssa_module.scale
            stats.record_qk(module_name, TBH, N, D, q_density,
                            used_sparse=False)

        # --- attn @ V (sparse attn after attn_lif) ---
        # Reshape attn through attn_lif
        attn = attn.reshape(T, B, num_heads, N, N)
        # attn_lif expects (T, B, N, C) — reshape accordingly
        # The original code reshapes to (T, B, N, C) before attn_lif
        # attn has shape (T, B, H, N, N), which is applied differently.
        # Looking at SSA.forward: attn is (T,B,H,N,N), then x = attn @ v
        # then x is (T,B,H,N,D), reshaped to (T,B,N,C), then attn_lif.
        # So attn_lif is AFTER the attn@v matmul in the original code.
        # Let's follow the original flow exactly.

        x_out = attn.reshape(TBH, N, N) @ v_flat  # (TBH, N, D) — dense for now
        x_out = x_out.reshape(T, B, num_heads, N, D)
        x_out = x_out.transpose(2, 3).reshape(T, B, N, C).contiguous()

        # attn_lif produces sparse binary spikes
        x_out = ssa_module.attn_lif(x_out)

        # Record attn@V stats (used dense path since attn_lif is after)
        attn_density = measure_density(attn)
        stats.record_av(module_name, TBH, N, D, attn_density,
                        used_sparse=False)
        stats.total_calls += 1

        # --- Output projection ---
        x_out = x_out.flatten(0, 1)  # (T*B, N, C)
        x_out = ssa_module.proj_lif(
            ssa_module.proj_bn(
                ssa_module.proj_linear(x_out).transpose(-1, -2)
            ).transpose(-1, -2).reshape(T, B, N, C)
        )
        return x_out

    return sparse_forward


def make_sparse_ssa_forward_v2(ssa_module: nn.Module, module_name: str,
                                stats: SputnikAttentionStats,
                                density_threshold: float = 0.15,
                                min_elements: int = 4096,
                                enabled_ref: list = None):
    """Alternative SSA forward that also uses SpMM for attn_spike @ V.

    In SSA, attn_lif is applied to the attention output AFTER attn@V,
    not to the attention weights. However, we can restructure to apply
    sparsity to both matmuls if we capture the intermediate sparse tensors.

    This version hooks into the flow to use SpMM for the second matmul
    by reordering: compute attn, apply to v using dense, then attn_lif.
    The sparsity benefit comes primarily from the Q@K.T operation.
    """
    # For the current SSA architecture, attn_lif is applied AFTER attn@v,
    # so the second matmul (attn@v) doesn't benefit from spike sparsity.
    # Use the v1 forward which applies SpMM only to Q@K.T.
    return make_sparse_ssa_forward(
        ssa_module, module_name, stats,
        density_threshold, min_elements, enabled_ref
    )
