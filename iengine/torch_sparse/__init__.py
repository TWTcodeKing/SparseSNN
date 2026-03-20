"""CSR sparse acceleration backend using torch.sparse.mm.

Exploits SNN binary spike sparsity by dynamically converting activations
to CSR format.  Supports Linear, Conv2d (im2col), and SSA attention.

Note: Per-call CSR conversion is slow — this is a research baseline.
For production, use semi_structured (2:4 Sparse Tensor Cores).

Provides:
  - accelerator:  TorchSparseAccelerator (SparseAccelerator interface)
  - inference:    benchmark_torch_sparse() — single entry point
"""

from .accelerator import TorchSparseAccelerator
from .inference import benchmark_torch_sparse

__all__ = ['TorchSparseAccelerator', 'benchmark_torch_sparse']
