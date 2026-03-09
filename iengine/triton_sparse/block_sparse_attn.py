"""Block-sparse attention for Spiking Self-Attention (SSA).

Monkey-patches SSA.forward to use block-sparse matmul for the attention
computation (q @ k^T) @ v. Since q, k, v are binary spikes after LIF neurons,
many block_size x block_size tiles of the attention matrix are guaranteed to
be zero. We compute a block mask and only process non-zero blocks via the
Triton block_sparse_matmul_kernel.

SSA in Spikformer:
    q, k, v are binary spikes, shape (T, B, H, N, D) where H=num_heads
    attn = (q @ k^T) * scale   -- shape (T, B, H, N, N)
    out  = attn @ v             -- shape (T, B, H, N, D)

Block-sparse optimization:
    For block_size=16 and N=64 (CIFAR 32x32, patch_size=4 -> 8x8=64 patches),
    we get a 4x4 grid of blocks in the NxN attention matrix.
    A block (i,j) is zero if q[block_i, :] is all-zero OR k[block_j, :] is all-zero.
"""

import torch
import torch.nn as nn

from iengine.common.hooks import monkey_patch_forward
from .kernels import block_sparse_matmul_kernel


def _compute_block_mask(q, k, block_size):
    """Compute which blocks of q @ k^T would be non-zero.

    Args:
        q: (N, D) binary spike tensor for one (t, b, h) slice.
        k: (N, D) binary spike tensor.
        block_size: Tile size for block-sparse structure.

    Returns:
        block_rows: (num_nonzero,) int32 tensor of non-zero block row indices
        block_cols: (num_nonzero,) int32 tensor of non-zero block col indices
        num_blocks_m: total blocks in M dimension
        num_blocks_n: total blocks in N dimension
    """
    N = q.shape[0]
    num_blocks_m = (N + block_size - 1) // block_size
    num_blocks_n = (N + block_size - 1) // block_size

    # A block (i, j) of q @ k^T is non-zero iff:
    # - q[i*bs:(i+1)*bs, :] has at least one non-zero element, AND
    # - k[j*bs:(j+1)*bs, :] has at least one non-zero element
    # (Since both are binary, if either block is all-zero, the product block is zero.)

    q_block_active = torch.zeros(num_blocks_m, dtype=torch.bool, device=q.device)
    k_block_active = torch.zeros(num_blocks_n, dtype=torch.bool, device=k.device)

    for i in range(num_blocks_m):
        start = i * block_size
        end = min(start + block_size, N)
        q_block_active[i] = q[start:end].any()

    for j in range(num_blocks_n):
        start = j * block_size
        end = min(start + block_size, N)
        k_block_active[j] = k[start:end].any()

    # Non-zero blocks: outer product of active rows and active cols
    active_rows = q_block_active.nonzero(as_tuple=False).squeeze(1)
    active_cols = k_block_active.nonzero(as_tuple=False).squeeze(1)

    if active_rows.numel() == 0 or active_cols.numel() == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=q.device)
        return empty, empty, num_blocks_m, num_blocks_n

    # Cartesian product of active rows and cols
    grid_rows = active_rows.repeat_interleave(active_cols.shape[0])
    grid_cols = active_cols.repeat(active_rows.shape[0])

    return grid_rows.to(torch.int32), grid_cols.to(torch.int32), num_blocks_m, num_blocks_n


def _block_sparse_attn_matmul(A, B, block_rows, block_cols, block_size):
    """Block-sparse matmul C = A @ B using Triton kernel.

    Only computes output blocks indicated by (block_rows, block_cols).

    Args:
        A: (M, K) dense matrix.
        B: (K, N) dense matrix.
        block_rows: (num_nz_blocks,) row block indices.
        block_cols: (num_nz_blocks,) col block indices.
        block_size: Block tile size.

    Returns:
        C: (M, N) result matrix with only specified blocks filled.
    """
    M, K = A.shape
    _, N = B.shape
    num_blocks = block_rows.shape[0]

    C = torch.zeros(M, N, device=A.device, dtype=torch.float32)

    if num_blocks == 0:
        return C

    # Ensure float32 and contiguous for Triton
    A_f = A.float().contiguous()
    B_f = B.float().contiguous()
    block_rows_c = block_rows.contiguous()
    block_cols_c = block_cols.contiguous()

    # The kernel BLOCK_SIZE must be a power of 2 and >= 16
    # Clamp block_size to valid Triton constexpr values
    if block_size < 16:
        # Pad to 16 -- the kernel handles bounds checking via masks
        kernel_block_size = 16
    else:
        # Round up to next power of 2
        kernel_block_size = 1
        while kernel_block_size < block_size:
            kernel_block_size *= 2

    grid = (num_blocks,)

    block_sparse_matmul_kernel[grid](
        A_f, B_f, C,
        block_rows_c, block_cols_c,
        num_blocks,
        M, K, N,
        A_f.stride(0), A_f.stride(1),
        B_f.stride(0), B_f.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=kernel_block_size,
    )

    return C


def make_block_sparse_ssa_forward(original_forward, ssa_module, block_size=16,
                                  min_seq_len=32, stats=None):
    """Create a block-sparse replacement for SSA.forward.

    Args:
        original_forward: The original SSA.forward method.
        ssa_module: The SSA module instance (for accessing sub-modules).
        block_size: Block tile size for sparse attention.
        min_seq_len: Minimum sequence length N to use block-sparse path.
        stats: Dict to accumulate statistics (mutated in place).

    Returns:
        New forward function with the same signature as SSA.forward.
    """
    if stats is None:
        stats = {}

    def block_sparse_forward(x):
        """Block-sparse SSA forward pass.

        Falls back to dense if:
        - N < min_seq_len (too small for block overhead)
        - Density is too high (most blocks would be active anyway)
        """
        T, B, N, C = x.shape
        num_heads = ssa_module.num_heads
        head_dim = C // num_heads

        # For small N, block-sparse overhead is not worthwhile
        if N < min_seq_len:
            stats['attn_dense_fallback'] = stats.get('attn_dense_fallback', 0) + 1
            return original_forward(x)

        # Compute q, k, v through the SSA sub-modules (same as original forward)
        x_for_qkv = x.flatten(0, 1)  # (T*B, N, C)

        q_linear_out = ssa_module.q_linear(x_for_qkv)
        q_linear_out = ssa_module.q_bn(
            q_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = ssa_module.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, num_heads, head_dim).permute(
            0, 1, 3, 2, 4
        ).contiguous()  # (T, B, H, N, D)

        k_linear_out = ssa_module.k_linear(x_for_qkv)
        k_linear_out = ssa_module.k_bn(
            k_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = ssa_module.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, num_heads, head_dim).permute(
            0, 1, 3, 2, 4
        ).contiguous()  # (T, B, H, N, D)

        v_linear_out = ssa_module.v_linear(x_for_qkv)
        v_linear_out = ssa_module.v_bn(
            v_linear_out.transpose(-1, -2)
        ).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = ssa_module.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, num_heads, head_dim).permute(
            0, 1, 3, 2, 4
        ).contiguous()  # (T, B, H, N, D)

        scale = ssa_module.scale

        # Block-sparse attention: process each (t, b, h) slice
        # attn = (q @ k^T) * scale, then out = attn @ v
        out = torch.zeros_like(q)  # (T, B, H, N, D)

        total_blocks_possible = 0
        total_blocks_computed = 0

        for t in range(T):
            for b_idx in range(B):
                for h in range(num_heads):
                    q_slice = q[t, b_idx, h]  # (N, D)
                    k_slice = k[t, b_idx, h]  # (N, D)
                    v_slice = v[t, b_idx, h]  # (N, D)

                    num_blocks_total = ((N + block_size - 1) // block_size) ** 2
                    total_blocks_possible += num_blocks_total

                    # Compute block mask for q @ k^T
                    block_rows, block_cols, nb_m, nb_n = _compute_block_mask(
                        q_slice, k_slice, block_size
                    )
                    num_nz = block_rows.shape[0]
                    total_blocks_computed += num_nz

                    if num_nz == 0:
                        # All-zero attention -- output is zero
                        continue

                    # Check if block sparsity is sufficient
                    block_density = num_nz / max(num_blocks_total, 1)
                    if block_density > 0.75:
                        # Most blocks active, use dense path for this slice
                        attn = (q_slice @ k_slice.t()) * scale
                        out[t, b_idx, h] = attn @ v_slice
                        continue

                    # Block-sparse: attn = q @ k^T (only non-zero blocks)
                    attn = _block_sparse_attn_matmul(
                        q_slice, k_slice.t().contiguous(),
                        block_rows, block_cols, block_size
                    )
                    attn = attn * scale

                    # For attn @ v, we use dense since attn is now partially filled
                    # and v is also sparse (binary spikes)
                    if attn.dtype != v_slice.dtype:
                        attn = attn.to(v_slice.dtype)
                    out[t, b_idx, h] = attn @ v_slice

        stats['attn_total_blocks'] = stats.get('attn_total_blocks', 0) + total_blocks_possible
        stats['attn_computed_blocks'] = stats.get('attn_computed_blocks', 0) + total_blocks_computed
        stats['attn_sparse_calls'] = stats.get('attn_sparse_calls', 0) + 1

        # Reshape back: (T, B, H, N, D) -> (T, B, N, C)
        x_out = out.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x_out = ssa_module.attn_lif(x_out)
        x_out = x_out.flatten(0, 1)
        x_out = ssa_module.proj_lif(
            ssa_module.proj_bn(
                ssa_module.proj_linear(x_out).transpose(-1, -2)
            ).transpose(-1, -2).reshape(T, B, N, C)
        )
        return x_out

    return block_sparse_forward


def apply_block_sparse_attention(model, block_size=16, min_seq_len=32, stats=None):
    """Find all SSA modules in the model and apply block-sparse attention.

    Args:
        model: The full model (e.g., Spikformer).
        block_size: Block tile size.
        min_seq_len: Minimum sequence length to use sparse path.
        stats: Shared statistics dict.

    Returns:
        List of (module, original_forward) tuples for cleanup.
    """
    if stats is None:
        stats = {}

    patched = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'SSA':
            original = module.forward
            new_forward = make_block_sparse_ssa_forward(
                original, module, block_size=block_size,
                min_seq_len=min_seq_len, stats=stats,
            )
            monkey_patch_forward(module, new_forward)
            patched.append((module, original))

    return patched
