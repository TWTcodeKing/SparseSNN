"""
iengine — Sparse inference acceleration backends for SNN models.

Each backend implements the SparseAccelerator interface from iengine.common.base.
Backends exploit the inherent spike sparsity (~6-8% density) in SNN activations
to skip zero-valued multiply-accumulate operations.

Available backends:
    - torch_sparse:    PyTorch CSR sparse tensors for Linear/attention
    - triton_sparse:   Custom Triton GPU kernels for Conv2d/block-sparse attention
    - semi_structured: NVIDIA 2:4 Sparse Tensor Cores for Linear weight pruning
    - sputnik_sparse:  Google Research SpMM/SDDMM CUDA kernels
"""
