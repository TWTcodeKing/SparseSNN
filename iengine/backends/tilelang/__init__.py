"""TileLang backend: fused Conv+BN+IF/LIF kernels for SNN inference.

Uses TileLang's @tilelang.jit to compile pipelined tensor-core GEMM
kernels with register-resident BN + spiking neuron epilogues.

Kernels
-------
conv2d_bn_if_kernel  / conv2d_bn_lif_kernel   -- Dense Conv2d + BN + IF/LIF
conv2d_bn_if_sparse_kernel                    -- 2:4 sparse Conv2d + BN + IF
linear_bn_if_kernel  / linear_bn_lif_kernel   -- Dense Linear + BN + IF/LIF
"""

from iengine.backends.tilelang.conv2d_bn_neuron import (
    conv2d_bn_if_kernel,
    conv2d_bn_lif_kernel,
)
from iengine.backends.tilelang.conv2d_bn_neuron_sparse import (
    conv2d_bn_if_sparse_kernel,
    conv2d_bn_lif_sparse_kernel,
)
from iengine.backends.tilelang.linear_bn_neuron import (
    linear_bn_if_kernel,
    linear_bn_lif_kernel,
)
