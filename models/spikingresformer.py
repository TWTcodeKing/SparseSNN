"""
SpikingResformer: Bridging ResNet and Vision Transformer
Paper: https://arxiv.org/abs/2403.14302
Source: https://github.com/xyshi2000/SpikingResformer

Standalone implementation matching the original exactly.
"""

import math
import torch
import torch.nn as nn
from models.neurons import MultiStepLIFNeuron


# ── Utility layers ──────────────────────────────────────────────────────

class BN(nn.Module):
    """BatchNorm wrapper for (T, B, C, H, W) input.
    Key: `.bn.weight` etc."""

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        T, B = x.shape[:2]
        out = self.bn(x.flatten(0, 1))
        return out.view(T, B, *out.shape[1:])


class _MultiStepConv2d(nn.Conv2d):
    """Conv2d that handles (T, B, C, H, W) multi-step input.
    spikingjelly layer.Conv2d equivalent with step_mode='m'."""

    def forward(self, x):
        if x.dim() == 5:
            T, B = x.shape[:2]
            out = nn.Conv2d.forward(self, x.flatten(0, 1))
            return out.view(T, B, *out.shape[1:])
        return nn.Conv2d.forward(self, x)


class Conv3x3(_MultiStepConv2d):
    def __init__(self, in_ch, out_ch, stride=1, groups=1):
        super().__init__(in_ch, out_ch, 3, stride=stride, padding=1,
                         bias=False, groups=groups)


class Conv1x1(_MultiStepConv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, 1, bias=False)


class _MultiStepMaxPool2d(nn.MaxPool2d):
    """MaxPool2d for (T, B, C, H, W) input."""

    def forward(self, x):
        T, B = x.shape[:2]
        out = nn.MaxPool2d.forward(self, x.flatten(0, 1))
        return out.view(T, B, *out.shape[1:])


class _MultiStepAvgPool2d(nn.AdaptiveAvgPool2d):
    """AdaptiveAvgPool2d for (T, B, C, H, W) input."""

    def forward(self, x):
        T, B = x.shape[:2]
        out = nn.AdaptiveAvgPool2d.forward(self, x.flatten(0, 1))
        return out.view(T, B, *out.shape[1:])


class _MultiStepLinear(nn.Linear):
    """Linear for (T, B, C) input."""

    def forward(self, x):
        if x.dim() == 3:
            T, B = x.shape[:2]
            out = nn.Linear.forward(self, x.flatten(0, 1))
            return out.view(T, B, *out.shape[1:])
        return nn.Linear.forward(self, x)


class LIF(MultiStepLIFNeuron):
    """LIF neuron matching spikingjelly: tau=2, v_threshold=1, ATan, decay_input=True."""

    def __init__(self):
        super().__init__(tau=2.0, v_threshold=1.0, v_reset=0.0,
                         surrogate='atan', detach_reset=True)


# ── DSSA: Deformable Spike-driven Self-Attention ────────────────────────

class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth

        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = LIF()

        self.W = _MultiStepConv2d(dim, dim * 2, kernel_size=patch_size,
                                  stride=patch_size, bias=False)
        self.norm = BN(dim * 2)
        self.activation_attn = LIF()
        self.activation_out = LIF()

        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = self.W(x)
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1 = y[:, :, :, :C // self.num_heads, :]
        y2 = y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        # EMA firing rate for x
        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (1 - self.momentum)

        scale1 = 1.0 / torch.sqrt(self.firing_rate_x.clamp(min=1e-8) * (self.dim // self.num_heads))
        attn = torch.matmul(y1.transpose(-2, -1), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        # EMA firing rate for attn
        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (1 - self.momentum)

        scale2 = 1.0 / torch.sqrt(self.firing_rate_attn.clamp(min=1e-8) * self.lenth)
        out = torch.matmul(y2, attn)
        out = out * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)

        out = self.Wproj(out)
        out = self.norm_proj(out)
        out = out + x_feat
        return out


# ── GWFFN: Group-Wise Feed-Forward Network ──────────────────────────────

class GWFFN(nn.Module):
    def __init__(self, in_channels, ratio=4, group_size=64):
        super().__init__()
        inner_channels = in_channels * ratio

        self.up = nn.Sequential(
            LIF(),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        # nn.ModuleList with one element → keys: conv.0.{0,1,2}.*
        self.conv = nn.ModuleList([
            nn.Sequential(
                LIF(),
                Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size),
                BN(inner_channels),
            )
        ])
        self.down = nn.Sequential(
            LIF(),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x):
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


# ── DownsampleLayer ─────────────────────────────────────────────────────

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.activation = LIF()
        self.conv = Conv3x3(in_channels, out_channels, stride=2)
        self.norm = BN(out_channels)

    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


# ── SpikingResformer ────────────────────────────────────────────────────

class SpikingResformer(nn.Module):
    def __init__(self, layer_names, planes, num_heads, patch_sizes,
                 img_size=224, T=4, in_channels=3, num_classes=1000,
                 group_size=64, **kwargs):
        super().__init__()
        self.T = T

        # Prologue: Conv7x7(stride=2) → BN → MaxPool(3, stride=2) → 4x downsample
        self.prologue = nn.Sequential(
            _MultiStepConv2d(in_channels, planes[0], 7, stride=2, padding=3, bias=False),
            BN(planes[0]),
            _MultiStepMaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        img_size = img_size // 4

        # Build stages
        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(DownsampleLayer(planes[idx - 1], planes[idx]))
                img_size = img_size // 2
            for name in layer_names[idx]:
                if name == 'DSSA':
                    lenth = (img_size // patch_sizes[idx]) ** 2
                    sub_layers.append(DSSA(planes[idx], num_heads[idx], lenth, patch_sizes[idx]))
                elif name == 'GWFFN':
                    sub_layers.append(GWFFN(planes[idx], group_size=group_size))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = _MultiStepAvgPool2d((1, 1))
        self.classifier = _MultiStepLinear(planes[-1], num_classes, bias=False)

    def forward(self, x):
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            # (B, T, C, H, W) → (T, B, C, H, W)
            x = x.transpose(0, 1)
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)  # (T, B, num_classes)
        return x.mean(0)  # average over T → (B, num_classes)


# ── Factory functions ────────────────────────────────────────────────────

_LAYER_PATTERN = [
    ['DSSA', 'GWFFN'] * 1,
    ['DSSA', 'GWFFN'] * 2,
    ['DSSA', 'GWFFN'] * 3,
]

_CONFIGS = {
    'ti': dict(planes=[64, 192, 384],   num_heads=[1, 3, 6],   patch_sizes=[4, 2, 1]),
    's':  dict(planes=[64, 256, 512],   num_heads=[1, 4, 8],   patch_sizes=[4, 2, 1]),
    'm':  dict(planes=[64, 384, 768],   num_heads=[1, 6, 12],  patch_sizes=[4, 2, 1]),
    'l':  dict(planes=[128, 512, 1024], num_heads=[2, 8, 16],  patch_sizes=[4, 2, 1]),
}


def spikingresformer(variant='s', num_classes=1000, in_channels=3, T=4,
                     img_size=224, **kwargs):
    cfg = _CONFIGS[variant]
    return SpikingResformer(
        _LAYER_PATTERN, num_classes=num_classes, in_channels=in_channels, T=T,
        img_size=img_size, **cfg, **kwargs,
    )


def build_spikingresformer(config):
    """Build from a config dict (for YAML-based pipeline)."""
    variant = config.get('variant', 's')
    return spikingresformer(
        variant=variant,
        num_classes=config.get('num_classes', 1000),
        in_channels=config.get('in_channels', 3),
        T=config.get('T', 4),
        img_size=config.get('img_size', 224),
    )
