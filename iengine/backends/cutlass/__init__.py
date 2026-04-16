"""CUTLASS backend for SNN inference.

Provides Conv2d+BN+IF/LIF fused kernels using CUTLASS 2.x implicit GEMM
with custom epilogue functors, plus a standalone compiler that generates
CUDA Graph + wavefront-scheduled binaries.

Submodules
----------
compile   -- Standalone CUTLASS compiler (PyTorch → .cu → binary)
backend   -- CUTLASS kernel template instantiation and management
kernels/  -- Custom epilogue headers (snn_epilogue.h, conv2d_if_kernel.cuh)
"""
