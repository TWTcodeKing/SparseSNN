"""Gather-accumulate backend for sparse activation exploitation.

For binary spike tensors, instead of full matrix multiplication W @ x, we only
gather columns of W corresponding to non-zero elements in x and sum them.
This avoids all multiply-accumulate operations for zero activations.

Provides a Triton GPU kernel with PyTorch fallback.
"""

from .triton_gather_acc import gather_accumulate, extract_nonzero_indices

__all__ = ['gather_accumulate', 'extract_nonzero_indices']
