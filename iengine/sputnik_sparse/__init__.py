"""Sputnik sparse acceleration backend using Google Research CUDA kernels.

Uses optimized Sputnik SpMM for sparse SNN activations through Linear
layers and SSA attention Q@K^T.

Requires building torch_sputnik from source:
    bash iengine/sputnik_sparse/build.sh

Provides:
  - accelerator: SputnikAccelerator (SparseAccelerator interface)
  - kernels:     Sputnik loading, CSR helpers, sputnik_spmm()
  - inference:   benchmark_sputnik_sparse() — single entry point
"""

from .accelerator import SputnikAccelerator
from .inference import benchmark_sputnik_sparse

__all__ = ['SputnikAccelerator', 'benchmark_sputnik_sparse']
