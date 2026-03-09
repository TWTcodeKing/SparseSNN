"""Sputnik sparse acceleration backend for SNN models.

Uses Google Research's Sputnik CUDA kernels (SpMM + SDDMM) via
Torch-Sputnik PyTorch bindings to accelerate sparse spike activations
through Linear layers and Spiking Self-Attention (SSA).

Requires building Sputnik and Torch-Sputnik from source.
See INSTALL.md or run `bash iengine/sputnik_sparse/build.sh` to build.
"""

from .accelerator import SputnikAccelerator

__all__ = ['SputnikAccelerator']
