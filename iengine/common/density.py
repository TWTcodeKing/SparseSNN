"""Runtime density measurement and sparse execution gating."""

import torch


def measure_density(tensor: torch.Tensor) -> float:
    """Fast density measurement: fraction of non-zero elements.

    For binary spike tensors, this equals the firing rate.
    """
    return (tensor != 0).float().mean().item()


def should_use_sparse(tensor: torch.Tensor, threshold: float = 0.15,
                      min_elements: int = 4096) -> bool:
    """Decide at runtime whether to use sparse execution.

    Returns True if:
    1. Tensor density is below threshold (default 15%)
    2. Tensor has enough elements to amortize conversion overhead

    Args:
        tensor: Input activation tensor.
        threshold: Maximum density for sparse path (0.0-1.0).
        min_elements: Minimum tensor size to justify sparse overhead.
    """
    if tensor.numel() < min_elements:
        return False
    density = measure_density(tensor)
    return density < threshold


def to_sparse_csr_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a 2D dense tensor to CSR sparse format.

    If input is already sparse, returns as-is. Handles contiguity.
    """
    if tensor.is_sparse or tensor.is_sparse_csr:
        return tensor
    t = tensor.contiguous()
    return t.to_sparse_csr()


def sparse_matmul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sparse matrix multiplication: sparse(a) @ dense(b).

    Converts `a` to CSR if dense, then uses torch.sparse.mm.
    Falls back to dense matmul if conversion fails.
    """
    try:
        if not (a.is_sparse or a.is_sparse_csr):
            a = to_sparse_csr_2d(a)
        return torch.sparse.mm(a, b)
    except Exception:
        # Fallback to dense
        if a.is_sparse or a.is_sparse_csr:
            a = a.to_dense()
        return a @ b
