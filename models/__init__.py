"""
SparseSNN Model Zoo

SNN foundation model implementations covering:
- SNN ResNet variants: SEW-ResNet, MS-ResNet
- SNN Transformer variants: Spikformer, Meta-SNN (Spike-Driven Transformer V2), QKFormer, MaxFormer

All models use standalone spiking neurons (models.neurons) instead of spikingjelly
where possible. Input format is standard (B, C, H, W) images; the temporal
dimension is handled internally.

Transformer models are built via `build_<arch>(config)` functions that accept a
config dict (loaded from YAML). ResNet models retain their direct factory functions.
"""

# Neuron primitives
from .neurons import (
    LIFNeuron, IFNeuron,
    MultiStepLIFNeuron, MultiStepIFNeuron,
    reset_net, heaviside,
)
from .layers import SeqToANNContainer, SeqToANNContainerT

# ---- SNN ResNet variants ----
from .sewresnet import (
    SEWResNet,
    sew_resnet18, sew_resnet34, sew_resnet50, sew_resnet101, sew_resnet152,
)
from .msresnet import (
    ms_resnet18, ms_resnet34, ms_resnet104,
)

# ---- SNN Transformer variants (config-based builders) ----
from .spikformer import Spikformer, build_spikformer
from .metaformer import SpikeDrivenTransformerV2, build_metaformer
from .qkformer import QKFormer, build_qkformer
from .maxformer import MaxFormer, build_maxformer

# arch name -> build function mapping
ARCH_BUILDERS = {
    'spikformer': build_spikformer,
    'metaformer': build_metaformer,
    'qkformer': build_qkformer,
    'maxformer': build_maxformer,
}
