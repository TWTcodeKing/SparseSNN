"""
SEW-ResNet for DVS datasets (CIFAR-10-DVS, DVS128 Gesture).

Implements the DVS-specific architecture from:
  https://github.com/fangwei123456/Spike-Element-Wise-ResNet

This is structurally different from the ImageNet SEW-ResNet (sewresnet.py):
  - Flat nn.Sequential backbone (no layer1/2/3/4 grouping)
  - 1x1 conv transitions for channel changes, MaxPool2d for spatial downsampling
  - Parametric LIF (PLIF) neurons with learnable decay parameter `w`
  - SEWBlock: two conv3x3 with element-wise residual connection

Checkpoint key structure (matches spikingjelly training output):
  conv.0.0.module.0.weight     — stem conv1x1 (SeqToANNContainer)
  conv.0.1.w                   — stem PLIF neuron
  conv.1.conv.0.0.module.0.weight — block conv3x3 (SeqToANNContainer)
  conv.1.conv.0.1.w            — block PLIF neuron
  conv.9.0.module.0.weight     — channel transition conv1x1
  out.weight, out.bias         — classifier
"""

import math

import torch
import torch.nn as nn
from models.layers import SeqToANNContainer
from models.neurons import heaviside


# ── Parametric LIF neuron ───────────────────────────────────────────────

class PLIFNeuron(nn.Module):
    """Multi-step Parametric LIF neuron (flat, no nesting).

    Matches spikingjelly's MultiStepParametricLIFNode key structure:
    state_dict has a single `w` parameter (no `.neuron.w` nesting).

    Parameterization (matches spikingjelly):
        reciprocal_tau = sigmoid(w)   (i.e. 1/tau = sigmoid(w))
        v[t] = v[t-1] + reciprocal_tau * (x[t] - v[t-1])
        spike = Heaviside(v - v_threshold)

    Input: (T, B, C, ...) → Output: (T, B, C, ...) binary spikes.
    """

    def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0,
                 surrogate='atan', detach_reset=True):
        super().__init__()
        # w = -log(tau - 1), so sigmoid(w) = 1/tau
        # Shape [1] matches spikingjelly's ParametricLIFNode
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.tensor([init_w]))
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = surrogate
        self.detach_reset = detach_reset
        self.v = 0.0

    def reset(self):
        self.v = 0.0

    def _step(self, x):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x)

        reciprocal_tau = self.w.sigmoid()
        self.v = self.v + reciprocal_tau * (x - self.v)

        spike = heaviside(self.v - self.v_threshold, self.surrogate)

        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1.0 - spike_d) * self.v + spike_d * self.v_reset
        return spike

    def forward(self, x_seq):
        """x_seq: (T, B, C, ...) → (T, B, C, ...) spikes"""
        self.reset()
        spikes = []
        for t in range(x_seq.shape[0]):
            spikes.append(self._step(x_seq[t]))
        return torch.stack(spikes, dim=0)


# ── Building blocks ─────────────────────────────────────────────────────

def _conv3x3(in_channels, out_channels, init_tau=2.0):
    """Conv3x3 + BN + PLIF, matching spikingjelly's conv3x3."""
    return nn.Sequential(
        SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        PLIFNeuron(init_tau=init_tau, detach_reset=True),
    )


def _conv1x1(in_channels, out_channels, init_tau=2.0):
    """Conv1x1 + BN + PLIF, for channel transitions."""
    return nn.Sequential(
        SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        PLIFNeuron(init_tau=init_tau, detach_reset=True),
    )


class SEWBlock(nn.Module):
    """SEW residual block: two conv3x3 with element-wise skip connection.

    State dict keys:
        conv.0.0.module.{0,1}.*   — first conv3x3 (Conv2d, BN)
        conv.0.1.w                — first PLIF
        conv.1.0.module.{0,1}.*   — second conv3x3
        conv.1.1.w                — second PLIF
    """

    def __init__(self, in_channels, mid_channels, connect_f='ADD', init_tau=2.0):
        super().__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            _conv3x3(in_channels, mid_channels, init_tau),
            _conv3x3(mid_channels, in_channels, init_tau),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out = out + x
        elif self.connect_f == 'AND':
            out = out * x
        elif self.connect_f == 'IAND':
            out = x * (1.0 - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


# ── Main model ───────────────────────────────────────────────────────────

class DVSSEWResNet(nn.Module):
    """SEW-ResNet for DVS event-based datasets.

    Architecture: flat nn.Sequential backbone with SEWBlocks, 1x1 transitions,
    and MaxPool2d spatial downsampling. Uses PLIF neurons throughout.

    Constructed from a layer_list config, matching spikingjelly's ResNetN.

    Args:
        layer_list: List of stage dicts, each with keys:
            channels, up_kernel_size, mid_channels, num_blocks, block_type, k_pool
        num_classes: Number of output classes.
        in_channels: Input channels (2 for DVS).
        connect_f: Residual connection type ('ADD', 'AND', 'IAND').
        init_tau: Initial PLIF tau value (default 2.0).
        img_size: Spatial input size (default 128 for CIFAR-10-DVS).
    """

    def __init__(self, layer_list, num_classes, in_channels=2,
                 connect_f='ADD', init_tau=2.0, img_size=128):
        super().__init__()
        ch = in_channels
        conv_layers = []

        for cfg in layer_list:
            channels = cfg['channels']
            mid_channels = cfg.get('mid_channels', channels)

            # Channel transition (1x1 or 3x3 conv)
            if ch != channels:
                ks = cfg['up_kernel_size']
                if ks == 1:
                    conv_layers.append(_conv1x1(ch, channels, init_tau))
                elif ks == 3:
                    conv_layers.append(_conv3x3(ch, channels, init_tau))
                else:
                    raise NotImplementedError(f'up_kernel_size={ks}')
                ch = channels

            # Residual blocks
            num_blocks = cfg.get('num_blocks', 0)
            block_type = cfg.get('block_type', 'sew')
            for _ in range(num_blocks):
                if block_type == 'sew':
                    conv_layers.append(SEWBlock(ch, mid_channels, connect_f, init_tau))
                else:
                    raise NotImplementedError(f'block_type={block_type}')

            # Spatial downsampling
            if 'k_pool' in cfg:
                conv_layers.append(
                    SeqToANNContainer(nn.MaxPool2d(cfg['k_pool'], cfg['k_pool']))
                )

        conv_layers.append(nn.Flatten(2))
        self.conv = nn.Sequential(*conv_layers)

        # Compute output features by tracing spatial dims through MaxPool layers
        spatial = img_size
        for cfg in layer_list:
            if 'k_pool' in cfg:
                spatial = spatial // cfg['k_pool']
        out_features = cfg['channels'] * spatial * spatial
        self.out = nn.Linear(out_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: (T, B, C, H, W) temporal spike tensor.

        Returns:
            logits: (B, num_classes) — mean over timesteps.
        """
        x = x.transpose(0, 1).contiguous()  # (B, T, C, H, W) for nn.Sequential
        x = self.conv(x)  # (T, B, out_features)
        return self.out(x.mean(0))  # mean over T → (B, num_classes)


# ── Factory functions ────────────────────────────────────────────────────

# Default config: 4 stages × 64ch + 3 stages × 128ch, matching the original
_DVS_SEW_RESNET_CONFIG = [
    {'channels': 64,  'up_kernel_size': 1, 'mid_channels': 64,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 64,  'up_kernel_size': 1, 'mid_channels': 64,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 64,  'up_kernel_size': 1, 'mid_channels': 64,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 64,  'up_kernel_size': 1, 'mid_channels': 64,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128,
     'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
]


def _make_uniform_config(channels, num_stages=7):
    """Create a uniform layer_list with the same channel width at every stage."""
    return [
        {'channels': channels, 'up_kernel_size': 1, 'mid_channels': channels,
         'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2}
        for _ in range(num_stages)
    ]


def dvs_sew_resnet(num_classes=10, in_channels=2, connect_f='ADD',
                   img_size=128, channels=None, **kwargs):
    """DVS SEW-ResNet (configurable channel width, 7 stages).

    Args:
        channels: Channel width. If int, uniform width at all stages.
            If None, uses default 4x64ch + 3x128ch config.
            Matches checkpoints:
                channels=None → SEWResNet_ADD_T_16_*.pth (CIFAR-10-DVS)
                channels=32   → DVSGNetSEW32_*.pth (DVS Gesture)
    """
    if channels is not None:
        layer_list = _make_uniform_config(channels)
    else:
        layer_list = _DVS_SEW_RESNET_CONFIG
    return DVSSEWResNet(
        layer_list, num_classes=num_classes,
        in_channels=in_channels, connect_f=connect_f, img_size=img_size,
    )
