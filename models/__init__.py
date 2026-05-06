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
    LIFNeuron, IFNeuron, ILIFNeuron,
    MultiStepLIFNeuron, MultiStepIFNeuron, MultiStepILIFNeuron,
    reset_net, heaviside,
)
from .layers import SeqToANNContainer, SeqToANNContainerT

# ---- SNN ResNet variants ----
from .sewresnet import (
    SEWResNet,
    sew_resnet18, sew_resnet34, sew_resnet50, sew_resnet101, sew_resnet152,
    sew_resnet_cifar20, sew_resnet_cifar32, sew_resnet_cifar44,
    sew_resnet_cifar56, sew_resnet_cifar110,
)
from .msresnet import (
    ms_resnet18, ms_resnet34, ms_resnet50, ms_resnet104,
    ms_resnet_cifar20, ms_resnet_cifar32, ms_resnet_cifar44,
    ms_resnet_cifar56, ms_resnet_cifar110,
    ms_resnet_dvs20,
)
from .dvs_sewresnet import DVSSEWResNet, dvs_sew_resnet
from .snn_vgg import SNNVGG, snn_vgg9, snn_vgg11, snn_vgg16, snn_vgg19
from .spikingresformer import SpikingResformer, spikingresformer, build_spikingresformer

# ---- SNN Transformer variants (config-based builders) ----
from .spikformer import Spikformer, build_spikformer
from .metaformer import SpikeDrivenTransformerV2, build_metaformer
from .qkformer import QKFormer, build_qkformer
from .maxformer import MaxFormer, build_maxformer, MS_QKFormer, build_ms_qkformer

# ---- Detection models ----
from .spike_yolo import SpikeYOLO, build_spike_yolo, spike_yolo_n, spike_yolo_s, spike_yolo_m
from .ems_yolo import EMSYOLO, build_ems_yolo, ems_yolo_res34

# ---- NLP models ----
from .spike_bert import SpikeBERT, build_spike_bert

# arch name -> build function mapping
ARCH_BUILDERS = {
    'spikformer': build_spikformer,
    'metaformer': build_metaformer,
    'qkformer': build_qkformer,
    'maxformer': build_maxformer,
    'ms_qkformer': build_ms_qkformer,
    'spikingresformer': build_spikingresformer,
    'spike_yolo': build_spike_yolo,
    'ems_yolo': build_ems_yolo,
    'spike_bert': build_spike_bert,
}
