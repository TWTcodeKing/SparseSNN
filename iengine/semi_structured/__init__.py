"""Semi-structured 2:4 sparsity backend for NVIDIA Sparse Tensor Cores.

Provides:
  - conversion:  Linear + Conv2d → SparseSemiStructuredTensor / SparseConv2d
  - inference:   End-to-end benchmark (dense fp16 vs sparse 2:4 fp16)
  - accelerator: SemiStructuredAccelerator (SparseAccelerator interface)

Pruning primitives (prune_2_4, verify_2_4) live in sparse.pruning — import
them directly from there.
"""

from .accelerator import SemiStructuredAccelerator
from .conversion import (
    convert_linear_to_semi_structured,
    convert_to_semi_structured,
    restore_dense,
    SparseConv2d,
)
# from .inference import benchmark_semi_structured

# # Backward compatibility alias
# benchmark_structured_sparse = benchmark_semi_structured

__all__ = [
    'SemiStructuredAccelerator',
    'convert_linear_to_semi_structured',
    'convert_to_semi_structured',
    'restore_dense',
    'SparseConv2d',
    'benchmark_semi_structured',
    'benchmark_structured_sparse',
]
