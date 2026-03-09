"""torch_sparse — CSR sparse acceleration backend using torch.sparse.mm.

Accelerates nn.Linear layers and Spikformer SSA attention by converting
sparse spike activations to CSR format and using torch.sparse.mm for
matrix multiplication.
"""

from .accelerator import TorchSparseAccelerator

__all__ = ['TorchSparseAccelerator']
