"""Semi-structured 2:4 sparsity backend for NVIDIA Sparse Tensor Cores.

Applies 2:4 magnitude-based weight pruning to nn.Linear layers and converts
to SparseSemiStructuredTensor for hardware-accelerated inference.
"""

from .accelerator import SemiStructuredAccelerator
from .pruning import prune_2_4, prune_model_linear
from .conversion import convert_linear_to_semi_structured, restore_dense

__all__ = [
    'SemiStructuredAccelerator',
    'prune_2_4',
    'prune_model_linear',
    'convert_linear_to_semi_structured',
    'restore_dense',
]
