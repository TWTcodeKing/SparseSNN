"""Triton-based sparse acceleration backend for Spiking Neural Networks.

Provides GPU kernels that exploit spike sparsity:
- Sparse Conv2d: im2col + SpMM that skips zero columns via Triton kernel
- Block-sparse attention: block-sparse matmul for SSA (Spikformer)

Usage:
    from iengine.triton_sparse import TritonSparseAccelerator

    accel = TritonSparseAccelerator(config={
        'density_threshold': 0.5,
        'block_size': 16,
        'min_tensor_size': 4096,
    })
    model = accel.prepare(model)
    output = model(input)
    stats = accel.get_stats()
    model = accel.cleanup(model)
"""

from .accelerator import TritonSparseAccelerator
from .kernels import sparse_matmul_kernel, block_sparse_matmul_kernel

__all__ = [
    'TritonSparseAccelerator',
    'sparse_matmul_kernel',
    'block_sparse_matmul_kernel',
]
