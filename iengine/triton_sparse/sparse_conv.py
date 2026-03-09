"""Forward hook for Conv2d: im2col + Triton SpMM that skips zero columns.

Intercepts Conv2d forward passes. If the unfolded input is sufficiently sparse,
extracts only non-zero columns and dispatches the Triton sparse_matmul_kernel
instead of the standard dense convolution.

Conv2d hooks see (T*B, C_in, H, W) tensors after SeqToANNContainer flattening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from iengine.common.density import measure_density, should_use_sparse
from .kernels import sparse_matmul_kernel


def _compute_output_size(H, W, kernel_size, padding, stride, dilation=(1, 1)):
    """Compute output spatial dimensions for a Conv2d."""
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return H_out, W_out


def _select_block_sizes(M, K_nz, N):
    """Select Triton block sizes based on matrix dimensions.

    Returns (BLOCK_M, BLOCK_N, BLOCK_K) that are powers of 2 and fit the problem.
    Triton tl.dot requires minimum 16 for each dimension.
    """
    def _clamp_po2(val, lo=16, hi=128):
        # Find largest power of 2 <= val, clamped to [lo, hi]
        p = 1
        while p * 2 <= val:
            p *= 2
        return max(lo, min(p, hi))

    BLOCK_M = _clamp_po2(M, 16, 64)
    BLOCK_N = _clamp_po2(N, 16, 64)
    BLOCK_K = _clamp_po2(K_nz, 16, 64)
    return BLOCK_M, BLOCK_N, BLOCK_K


def sparse_conv2d_forward(x, conv_module, stats):
    """Sparse Conv2d using im2col + Triton SpMM.

    Args:
        x: Input tensor (TB, C_in, H, W) -- already flattened T*B.
        conv_module: The nn.Conv2d module.
        stats: Dict to accumulate statistics (mutated in place).

    Returns:
        Output tensor (TB, C_out, H_out, W_out) or None if falling back to dense.
    """
    TB, C_in, H, W = x.shape
    weight = conv_module.weight
    bias = conv_module.bias
    C_out = conv_module.out_channels

    kH, kW = conv_module.kernel_size
    padding = conv_module.padding
    stride = conv_module.stride
    dilation = conv_module.dilation
    groups = conv_module.groups

    # Only support groups=1 and dilation=1 for now
    if groups != 1 or dilation != (1, 1):
        return None

    H_out, W_out = _compute_output_size(H, W, (kH, kW), padding, stride, dilation)
    L = H_out * W_out  # number of output spatial positions
    K = C_in * kH * kW  # im2col column length

    # im2col: unfold input to (TB, K, L)
    x_unfolded = F.unfold(x, kernel_size=(kH, kW), padding=padding,
                          stride=stride, dilation=dilation)
    # x_unfolded shape: (TB, K, L)

    # Find non-zero columns: a column (along dim=1) is non-zero if any element is non-zero
    # Transpose to (TB, L, K) then check across K dimension
    # Actually, we want to find which of the K "input feature" rows have any non-zero
    # across the spatial dimension L and batch. This is "column sparsity" of x_unfolded.
    # For spike tensors, many rows (receptive field positions) will be all-zero.

    # Reshape for matmul: we want (TB*L, K) @ (K, C_out) = (TB*L, C_out)
    # x_col: (TB, K, L) -> (TB*L, K)
    x_col = x_unfolded.permute(0, 2, 1).reshape(TB * L, K)

    # Weight matrix: (C_out, K) -> need (K, C_out) for x_col @ W^T
    w_mat = weight.view(C_out, K).t().contiguous()  # (K, C_out)

    # Find non-zero columns of x_col (columns = K dimension features)
    # A column k is non-zero if any element across all (TB*L) rows is non-zero
    col_nonzero = (x_col != 0).any(dim=0)  # (K,)
    nz_indices = col_nonzero.nonzero(as_tuple=False).squeeze(1)  # (K_nz,)
    K_nz = nz_indices.shape[0]

    if K_nz == 0:
        # Entirely zero input -- output is just bias
        out = torch.zeros(TB, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
        if bias is not None:
            out += bias.view(1, C_out, 1, 1)
        stats['skipped_zero'] = stats.get('skipped_zero', 0) + 1
        stats['total_ops'] = stats.get('total_ops', 0) + TB * L * K * C_out
        stats['effective_ops'] = stats.get('effective_ops', 0)
        return out

    density = K_nz / K
    M = TB * L
    total_ops = M * K * C_out
    effective_ops = M * K_nz * C_out  # upper bound: we skip zero cols but not zero elements

    stats['total_ops'] = stats.get('total_ops', 0) + total_ops
    stats['effective_ops'] = stats.get('effective_ops', 0) + effective_ops

    # Check if sparse path is worthwhile
    # For very small tensors or high density, dense is faster due to kernel launch overhead
    min_elements = stats.get('min_tensor_size', 4096)
    density_threshold = stats.get('density_threshold', 0.5)

    if density > density_threshold or M * K_nz < min_elements:
        stats['dense_fallback'] = stats.get('dense_fallback', 0) + 1
        return None  # Signal caller to use dense path

    stats['sparse_launches'] = stats.get('sparse_launches', 0) + 1

    # Extract non-zero columns from x_col and corresponding rows from w_mat
    x_filtered = x_col[:, nz_indices].contiguous()  # (M, K_nz)
    # For the Triton kernel, we pass the full w_mat and col_indices
    # so the kernel gathers from B using indirect indexing
    col_indices = nz_indices.to(torch.int32).contiguous()

    # Output matrix
    C_out_mat = torch.empty(M, C_out, device=x.device, dtype=torch.float32)

    # Select block sizes
    BLOCK_M, BLOCK_N, BLOCK_K = _select_block_sizes(M, K_nz, C_out)

    # Launch Triton kernel
    grid = (
        ((M + BLOCK_M - 1) // BLOCK_M) * ((C_out + BLOCK_N - 1) // BLOCK_N),
    )

    # Ensure float32 for Triton dot product
    x_f = x_filtered.float()
    w_f = w_mat.float()

    sparse_matmul_kernel[grid](
        x_f, w_f, C_out_mat,
        col_indices,
        M, K_nz, C_out, K,
        x_f.stride(0), x_f.stride(1),
        w_f.stride(0), w_f.stride(1),
        C_out_mat.stride(0), C_out_mat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Cast back to input dtype if needed
    if C_out_mat.dtype != x.dtype:
        C_out_mat = C_out_mat.to(x.dtype)

    # Reshape output: (TB*L, C_out) -> (TB, C_out, H_out, W_out)
    out = C_out_mat.view(TB, L, C_out).permute(0, 2, 1).reshape(TB, C_out, H_out, W_out)

    if bias is not None:
        out = out + bias.view(1, C_out, 1, 1)

    return out


def make_sparse_conv2d_forward(module, original_forward, stats,
                                density_threshold=0.5, min_tensor_size=4096):
    """Create a replacement forward for nn.Conv2d using Triton sparse matmul.

    When sparse path is beneficial: uses im2col + Triton SpMM (skips dense entirely).
    Otherwise: calls original_forward (dense, no redundancy).

    Args:
        module: The nn.Conv2d module.
        original_forward: The original forward method to fall back to.
        stats: Dict to accumulate statistics across all hooked layers.
        density_threshold: Max column density ratio to use sparse path.
        min_tensor_size: Minimum M*K_nz to justify Triton launch overhead.

    Returns:
        New forward function that replaces module.forward.
    """
    stats['density_threshold'] = density_threshold
    stats['min_tensor_size'] = min_tensor_size

    def sparse_forward(x):
        # Only handle 4D CUDA inputs
        if x.ndim != 4 or not x.is_cuda:
            return original_forward(x)

        result = sparse_conv2d_forward(x, module, stats)
        if result is not None:
            return result
        # None means fall back to original dense forward
        return original_forward(x)

    return sparse_forward
