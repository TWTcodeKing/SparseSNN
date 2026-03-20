"""Triton-based sparse acceleration backend for SNNs.

Custom GPU kernels that exploit SNN binary spike sparsity:
  - Conv2d: im2col + SpMM skipping zero columns
  - SSA attention: block-sparse Q @ K^T

Note: Kernel launch overhead dominates for small tensors (CIFAR).
Best suited for larger inputs or very sparse activations (<15%).

Provides:
  - accelerator: TritonSparseAccelerator (SparseAccelerator interface)
  - kernels:     Raw Triton JIT kernel definitions
  - inference:   benchmark_triton_sparse() — single entry point
"""

from .accelerator import TritonSparseAccelerator
from .inference import benchmark_triton_sparse

__all__ = ['TritonSparseAccelerator', 'benchmark_triton_sparse']
