"""Triton kernel for sparse activation gather-accumulate.

For binary spike tensors x (values 0 or 1), the matrix product y = x @ W^T
reduces to gathering and summing columns of W corresponding to non-zero
elements in each row of x:

    y[b, o] = sum(W[o, k] for k where x[b, k] != 0)

This kernel exploits SNN activation sparsity (~6-8% density) to skip
90%+ of multiply-accumulate operations.

Falls back to PyTorch implementation when Triton is not available.
"""

import torch
from typing import Optional

# Try to import Triton
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


def extract_nonzero_indices(
    x_binary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract non-zero column indices from a binary spike tensor.

    Args:
        x_binary: Binary tensor of shape (B, in_features) with values 0 or 1.

    Returns:
        nz_indices: (B, max_nnz) — non-zero column indices per batch element,
            padded with 0 to max_nnz (the maximum number of non-zeros across
            the batch).
        nz_counts: (B,) — actual number of non-zeros per batch element.
    """
    B, D = x_binary.shape
    device = x_binary.device

    # Find non-zero positions per row
    # nonzero() returns (row_idx, col_idx) pairs
    nz_mask = x_binary != 0  # (B, D) bool

    # Count non-zeros per batch element
    nz_counts = nz_mask.sum(dim=1)  # (B,)
    max_nnz = int(nz_counts.max().item())

    if max_nnz == 0:
        return (
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    # Build padded index tensor
    nz_indices = torch.zeros(B, max_nnz, dtype=torch.long, device=device)

    for b in range(B):
        cols = torch.where(nz_mask[b])[0]
        n = cols.shape[0]
        if n > 0:
            nz_indices[b, :n] = cols

    return nz_indices, nz_counts


def _extract_nonzero_indices_fast(
    x_binary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized extraction of non-zero indices (faster for large batches).

    Args:
        x_binary: Binary tensor of shape (B, in_features).

    Returns:
        Same as extract_nonzero_indices.
    """
    B, D = x_binary.shape
    device = x_binary.device

    nz_mask = x_binary != 0
    nz_counts = nz_mask.sum(dim=1)  # (B,)
    max_nnz = int(nz_counts.max().item())

    if max_nnz == 0:
        return (
            torch.zeros(B, 1, dtype=torch.long, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    # Use argsort trick: sort each row so True values come first
    # The indices of the sorted True values are the non-zero column indices
    sorted_indices = torch.argsort((~nz_mask).int(), dim=1, stable=True)  # (B, D)
    nz_indices = sorted_indices[:, :max_nnz].contiguous()

    return nz_indices, nz_counts


# ─── Triton kernel ──────────────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def gather_accumulate_kernel(
        # Pointers
        W_ptr,           # (out_features, in_features) — residual weight matrix
        nz_indices_ptr,  # (B, max_nnz) — non-zero column indices
        nz_counts_ptr,   # (B,) — actual nnz per batch element
        output_ptr,      # (B, out_features) — output buffer
        # Dimensions
        out_features: tl.constexpr,
        in_features: tl.constexpr,
        max_nnz: tl.constexpr,
        # Block sizes
        BLOCK_O: tl.constexpr,
    ):
        """Triton kernel: gather-accumulate for binary spike inputs.

        Each program instance handles one batch element and a block of output
        features. It iterates over the non-zero input indices, gathering the
        corresponding weight columns and accumulating.

        Grid: (B, cdiv(out_features, BLOCK_O))
        """
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)

        # Output feature range for this block
        o_start = block_id * BLOCK_O
        o_offsets = o_start + tl.arange(0, BLOCK_O)
        o_mask = o_offsets < out_features

        # Load the number of non-zero inputs for this batch element
        nnz = tl.load(nz_counts_ptr + batch_id)

        # Accumulator
        acc = tl.zeros([BLOCK_O], dtype=tl.float32)

        # Iterate over non-zero indices
        for k in range(max_nnz):
            # Only accumulate if k < actual nnz for this batch element
            k_mask = k < nnz

            # Load the column index
            nz_idx = tl.load(
                nz_indices_ptr + batch_id * max_nnz + k,
                mask=k_mask,
                other=0,
            )

            # Gather W[o_offsets, nz_idx] — one column of W for these output rows
            w_ptrs = W_ptr + o_offsets * in_features + nz_idx
            w_vals = tl.load(w_ptrs, mask=o_mask & k_mask, other=0.0)

            acc += w_vals.to(tl.float32)

        # Store result
        out_ptrs = output_ptr + batch_id * out_features + o_offsets
        tl.store(out_ptrs, acc.to(tl.float16), mask=o_mask)


def _gather_accumulate_triton(
    W_res: torch.Tensor,
    x_binary: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton-accelerated gather-accumulate.

    Args:
        W_res: Residual weight matrix, shape (out_features, in_features), fp16.
        x_binary: Binary spike input, shape (B, in_features), fp16.
        bias: Optional bias, shape (out_features,), fp16.

    Returns:
        Output tensor of shape (B, out_features), fp16.
    """
    B, in_features = x_binary.shape
    out_features = W_res.shape[0]

    # Extract non-zero indices
    nz_indices, nz_counts = _extract_nonzero_indices_fast(x_binary)
    max_nnz = nz_indices.shape[1]

    # Handle edge case: all zeros
    if max_nnz == 0 or nz_counts.max().item() == 0:
        output = torch.zeros(B, out_features, dtype=torch.float16, device=x_binary.device)
        if bias is not None:
            output = output + bias
        return output

    # Check density — if too dense, fall back to standard matmul
    density = nz_counts.float().mean().item() / in_features
    if density > 0.5:
        output = torch.nn.functional.linear(x_binary, W_res)
        if bias is not None:
            output = output + bias
        return output

    # Allocate output
    output = torch.empty(B, out_features, dtype=torch.float16, device=x_binary.device)

    # Launch kernel
    BLOCK_O = min(128, triton.next_power_of_2(out_features))
    grid = (B, triton.cdiv(out_features, BLOCK_O))

    gather_accumulate_kernel[grid](
        W_res,
        nz_indices,
        nz_counts,
        output,
        out_features=out_features,
        in_features=in_features,
        max_nnz=max_nnz,
        BLOCK_O=BLOCK_O,
    )

    if bias is not None:
        output = output + bias

    return output


def _gather_accumulate_pytorch(
    W_res: torch.Tensor,
    x_binary: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PyTorch fallback for gather-accumulate (no Triton required).

    For each batch element, gathers columns of W_res at non-zero input positions
    and sums them. Falls back to dense matmul if density is high.

    Args:
        W_res: Residual weight matrix, shape (out_features, in_features), fp16.
        x_binary: Binary spike input, shape (B, in_features), fp16.
        bias: Optional bias, shape (out_features,), fp16.

    Returns:
        Output tensor of shape (B, out_features), fp16.
    """
    B, in_features = x_binary.shape
    out_features = W_res.shape[0]
    device = x_binary.device

    # Extract non-zero indices
    nz_indices, nz_counts = _extract_nonzero_indices_fast(x_binary)
    max_nnz = nz_indices.shape[1]

    # Edge case: all zeros
    if max_nnz == 0 or nz_counts.max().item() == 0:
        output = torch.zeros(B, out_features, dtype=x_binary.dtype, device=device)
        if bias is not None:
            output = output + bias
        return output

    # Check density — if too dense, fall back to standard matmul
    density = nz_counts.float().mean().item() / in_features
    if density > 0.5:
        output = torch.nn.functional.linear(x_binary, W_res)
        if bias is not None:
            output = output + bias
        return output

    # Gather columns: W_res[:, nz_indices[b]] for each b
    # W_res is (O, D), nz_indices is (B, max_nnz)
    # We want gathered: (B, O, max_nnz) = W_res[:, nz_indices]
    # Then sum over the max_nnz dimension with masking

    # Index into W_res columns
    gathered = W_res[:, nz_indices.reshape(-1)]  # (O, B * max_nnz)
    gathered = gathered.reshape(out_features, B, max_nnz)  # (O, B, max_nnz)
    gathered = gathered.permute(1, 0, 2)  # (B, O, max_nnz)

    # Mask out padded positions
    # nz_counts: (B,), max_nnz: scalar
    mask = torch.arange(max_nnz, device=device).unsqueeze(0) < nz_counts.unsqueeze(1)
    # mask: (B, max_nnz)
    mask = mask.unsqueeze(1).expand_as(gathered)  # (B, O, max_nnz)

    # Zero out padding and sum (accumulate in fp32 to match Triton kernel)
    output = (gathered.float() * mask.float()).sum(dim=2).half()  # (B, O)

    if bias is not None:
        output = output + bias

    return output


def gather_accumulate(
    W_res: torch.Tensor,
    x_binary: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """High-level API for gather-accumulate sparse matmul.

    Dispatches to Triton kernel on CUDA when available, otherwise falls back
    to a PyTorch implementation.

    For binary spike inputs x (0/1), computes:
        y[b, o] = sum(W_res[o, k] for k where x[b, k] != 0) + bias[o]

    Args:
        W_res: Residual weight matrix, shape (out_features, in_features).
        x_binary: Binary spike input, shape (B, in_features).
        bias: Optional bias vector, shape (out_features,).

    Returns:
        Output tensor, shape (B, out_features).
    """
    if _HAS_TRITON and x_binary.is_cuda:
        return _gather_accumulate_triton(W_res, x_binary, bias)
    else:
        return _gather_accumulate_pytorch(W_res, x_binary, bias)
