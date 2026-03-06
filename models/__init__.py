"""
SparseSNN Model Zoo

SNN foundation model implementations covering:
- SNN ResNet variants: SEW-ResNet, MS-ResNet
- SNN Transformer variants: Spikformer, Spike-Driven Transformer V1/V2, QKFormer, MaxFormer

All models use standalone spiking neurons (models.neurons) instead of spikingjelly
where possible. Input format is standard (B, C, H, W) images; the temporal
dimension is handled internally.
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

# ---- SNN Transformer variants ----
from .spikformer import (
    Spikformer,
    spikformer_8_384, spikformer_8_512, spikformer_8_768,
)
from .sdformer import (
    SpikeDrivenTransformerV1,
    sdt_v1_8_384, sdt_v1_8_512, sdt_v1_8_768,
)
from .sdformer2 import (
    SpikeDrivenTransformerV2,
    meta_spikformer_8_384, meta_spikformer_8_512, meta_spikformer_8_768,
)
from .qkformer import (
    QKFormer,
    qkformer_10_384, qkformer_10_512, qkformer_10_768,
)
from .maxformer import (
    MaxFormer,
    maxformer_10_384, maxformer_10_512, maxformer_10_768,
)
