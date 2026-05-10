"""
SNN-VGG for DVS and static image datasets.

VGG-style sequential convolutional networks adapted for Spiking Neural Networks:
  - Conv2d + BatchNorm2d wrapped in SeqToANNContainer for temporal processing
  - Parametric LIF (PLIF) neurons with learnable decay parameter
  - Temporal mean pooling + adaptive spatial pooling before classifier

Four depth configurations: VGG-9, VGG-11, VGG-16, VGG-19.

Input: (B, T, C, H, W) for DVS or (B, C, H, W) for static images.
Output: (B, num_classes) logits.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.layers import SeqToANNContainer
from models.neurons import MultiStepLIFNeuron


# ── VGG feature configs ───────────────────────────────────────────────────
# Numbers are channel counts, 'M' is MaxPool2d(2,2).
# Layer count: conv layers + 1 FC = total named depth.

VGG_CFGS = {
    # 6 conv + 1 FC; 4 pools → spatial /16
    '9':  [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    # 8 conv + 1 FC; 5 pools → spatial /32
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 13 conv + 1 FC; 5 pools → spatial /32
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
           512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 16 conv + 1 FC; 5 pools → spatial /32
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
           512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# ── Feature extractor builder ─────────────────────────────────────────────

def _make_features(cfg_list, in_channels, init_tau=2.0):
    """Build VGG feature layers: Conv+BN+LIF blocks with MaxPool."""
    layers = []
    ch = in_channels
    for v in cfg_list:
        if v == 'M':
            layers.append(SeqToANNContainer(nn.MaxPool2d(2, 2)))
        else:
            layers.append(
                nn.Sequential(
                    SeqToANNContainer(
                        nn.Conv2d(ch, v, 3, padding=1, bias=False),
                        nn.BatchNorm2d(v),
                    ),
                    MultiStepLIFNeuron(tau=init_tau, detach_reset=True),
                )
            )
            ch = v
    return nn.Sequential(*layers), ch


# ── Main model ─────────────────────────────────────────────────────────────

class SNNVGG(nn.Module):
    """SNN-VGG: VGG-style spiking neural network.

    Args:
        cfg_name: VGG depth variant ('9', '11', '16', '19').
        num_classes: Number of output classes.
        in_channels: Input channels (2 for DVS, 3 for RGB).
        T: Number of timesteps (used when input is static 4D).
        init_tau: Initial PLIF tau value.
    """

    def __init__(self, cfg_name, num_classes=10, in_channels=2,
                 T=16, init_tau=2.0, **kwargs):
        super().__init__()
        self.T = T
        cfg_list = VGG_CFGS[cfg_name]
        self.features, last_ch = _make_features(cfg_list, in_channels, init_tau)
        self.classifier = nn.Linear(last_ch, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) DVS events or (B, C, H, W) static images.

        Returns:
            logits: (B, num_classes).
        """
        if x.dim() == 4:
            # Static image: repeat across T timesteps
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            # DVS: (B, T, C, H, W) → (T, B, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)

        x = self.features(x)              # (T, B, C', H', W')
        x = x.mean(dim=0)                 # temporal mean → (B, C', H', W')
        x = F.adaptive_avg_pool2d(x, 1)   # spatial pool  → (B, C', 1, 1)
        x = x.flatten(1)                  # (B, C')
        x = self.classifier(x)            # (B, num_classes)
        return x


# ── Factory functions ──────────────────────────────────────────────────────

def snn_vgg9(num_classes=10, in_channels=2, T=16, **kwargs):
    return SNNVGG('9', num_classes=num_classes, in_channels=in_channels,
                  T=T, **kwargs)


def snn_vgg11(num_classes=10, in_channels=2, T=16, **kwargs):
    return SNNVGG('11', num_classes=num_classes, in_channels=in_channels,
                  T=T, **kwargs)


def snn_vgg16(num_classes=10, in_channels=2, T=16, **kwargs):
    return SNNVGG('16', num_classes=num_classes, in_channels=in_channels,
                  T=T, **kwargs)


def snn_vgg19(num_classes=10, in_channels=2, T=16, **kwargs):
    return SNNVGG('19', num_classes=num_classes, in_channels=in_channels,
                  T=T, **kwargs)
