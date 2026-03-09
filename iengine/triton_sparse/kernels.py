"""Raw Triton JIT kernel definitions for sparse SNN acceleration.

Kernels:
- sparse_matmul_kernel: SpMM that only processes non-zero columns of A.
- block_sparse_matmul_kernel: Block-sparse matmul operating on block_size tiles
  with a mask indicating which blocks are non-zero.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sparse_matmul_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions: A is (M, K_nz), B is (K_full, N), C is (M, N)
    # col_indices maps K_nz columns of A to rows of B
    col_indices_ptr,
    M, K_nz, N, K_full,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (constexpr for Triton compiler)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """SpMM kernel: C = A_filtered @ B[col_indices, :].

    A has been pre-filtered to only contain non-zero columns (shape M x K_nz).
    col_indices maps each column of A to the corresponding row in B (shape K_full x N).
    This avoids loading and multiplying zero columns entirely.

    Grid: (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K_nz in BLOCK_K chunks
    for k_start in range(0, K_nz, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_nz

        # Load col_indices for this chunk to get the actual B row indices
        b_row_indices = tl.load(col_indices_ptr + offs_k, mask=k_mask, other=0)

        # Load A tile: (BLOCK_M, BLOCK_K) from the filtered matrix
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: (BLOCK_K, BLOCK_N) using indirect row indexing
        b_ptrs = B_ptr + b_row_indices[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def block_sparse_matmul_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Block layout: nonzero_block_indices has pairs (block_row_in_A, block_col_in_B)
    # stored as separate arrays for row and col
    block_row_ptr, block_col_ptr,
    num_blocks,
    # Matrix dims: A is (M, K), B is (K, N), C is (M, N)
    M, K, N,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block size for the sparse block structure
    BLOCK_SIZE: tl.constexpr,
):
    """Block-sparse matmul: C += A[block_m, :] @ B[:, block_n] for non-zero blocks.

    Operates on block_size x block_size tiles. A block mask indicates which
    (block_row, block_col) output blocks are non-zero. Only those tiles are
    computed, the rest remain zero.

    Grid: (num_blocks,) -- one program per non-zero output block.
    Each program computes one BLOCK_SIZE x BLOCK_SIZE tile of C by iterating
    over the full K dimension.
    """
    pid = tl.program_id(0)

    # Which output block are we computing?
    block_m_idx = tl.load(block_row_ptr + pid)
    block_n_idx = tl.load(block_col_ptr + pid)

    offs_m = block_m_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = block_n_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over full K dimension in BLOCK_SIZE chunks
    for k_start in range(0, K, BLOCK_SIZE):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = offs_k < K

        # Load A tile (BLOCK_SIZE, BLOCK_SIZE)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile (BLOCK_SIZE, BLOCK_SIZE)
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store output tile
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
