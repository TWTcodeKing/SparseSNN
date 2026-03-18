"""Structured sparse inference path for SR-STE-trained SNN models.

This package provides the post-training inference accelerator for models
trained with SR-STE 2:4 regularization (models/structured_training.py).
It applies hard 2:4 projection and converts eligible nn.Linear layers to
SparseSemiStructuredTensor for hardware-accelerated inference on NVIDIA
Sparse Tensor Cores (RTX 4090 / A100 and newer).

Typical workflow:
    1. Train with SR-STE:
           python tengine/train.py --config ... --structured-sparse \
               --sr-lambda 0.01 --sr-start-epoch 50 --sr-end-epoch 150
    2. Run benchmark / inference via this package:
           python -m iengine.structured_sparse.semi_structured_path \
               --checkpoint output/.../best.pth --config ... --dataset cifar100

Entry point: iengine.structured_sparse.semi_structured_path
"""

from .semi_structured_path import StructuredSparseAccelerator

__all__ = ['StructuredSparseAccelerator']
