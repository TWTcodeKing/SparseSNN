"""
SpikeYOLO: Spiking Neural Network Object Detection based on YOLOv8.

Uses I-LIF (Integer LIF) neurons with multi-level spike quantization.
Backbone is a simplified YOLOv8 with SNN-specific meta blocks.

Reference: https://github.com/BICLab/SpikeYOLO
"""

import torch
import torch.nn as nn

from models.layers import SeqToANNContainer
from models.neurons import MultiStepILIFNeuron
from models.detection_head import YOLOv8Detect


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpikeConv(nn.Module):
    """I-LIF → Conv2d + BN (pre-activation SNN convolution).

    Pattern: neuron fires first, then conv processes spikes.
    """

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 groups=1, decay=0.25, max_level=4):
        super().__init__()
        self.neuron = MultiStepILIFNeuron(decay=decay, max_level=max_level)
        self.conv_bn = SeqToANNContainer(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        """x: (T, B, C, H, W) → (T, B, C', H', W')"""
        x = self.neuron(x)
        return self.conv_bn(x)


class SpikeDWConv(nn.Module):
    """Depthwise separable SNN convolution: DW → PW."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 decay=0.25, max_level=4):
        super().__init__()
        self.dw = SpikeConv(in_ch, in_ch, kernel, stride, padding,
                            groups=in_ch, decay=decay, max_level=max_level)
        self.pw = SpikeConv(in_ch, out_ch, 1, 1, 0,
                            decay=decay, max_level=max_level)

    def forward(self, x):
        return self.pw(self.dw(x))


class MSConvBlock(nn.Module):
    """Meta SNN block with residual connections.

    Two parallel paths:
    1. SepConv path: depthwise separable conv
    2. MLP path: 1x1 expand → 1x1 reduce
    Both added to skip connection.
    """

    def __init__(self, ch, expand_ratio=2, decay=0.25, max_level=4):
        super().__init__()
        mid = int(ch * expand_ratio)

        # SepConv path
        self.sep_conv = SpikeDWConv(ch, ch, 3, 1, 1,
                                     decay=decay, max_level=max_level)

        # MLP path
        self.mlp_expand = SpikeConv(ch, mid, 1, 1, 0,
                                     decay=decay, max_level=max_level)
        self.mlp_reduce = SpikeConv(mid, ch, 1, 1, 0,
                                     decay=decay, max_level=max_level)

    def forward(self, x):
        """x: (T, B, C, H, W) → (T, B, C, H, W)"""
        sep = self.sep_conv(x)
        mlp = self.mlp_reduce(self.mlp_expand(x))
        return x + sep + mlp


class C2fSpike(nn.Module):
    """Simplified C2f block with SNN meta blocks (replaces YOLOv8 C2f).

    Split → N sequential MSConvBlocks → Concat → Conv1x1
    """

    def __init__(self, in_ch, out_ch, num_blocks=1, shortcut=True,
                 decay=0.25, max_level=4):
        super().__init__()
        mid = out_ch // 2
        self.cv1 = SpikeConv(in_ch, 2 * mid, 1, 1, 0,
                              decay=decay, max_level=max_level)
        self.blocks = nn.ModuleList([
            MSConvBlock(mid, decay=decay, max_level=max_level)
            for _ in range(num_blocks)
        ])
        # After concat: mid (skip) + mid (processed) = 2*mid → out_ch
        self.cv2 = SpikeConv(2 * mid + mid * num_blocks, out_ch, 1, 1, 0,
                              decay=decay, max_level=max_level)
        self.shortcut = shortcut

    def forward(self, x):
        """x: (T, B, C, H, W)"""
        x = self.cv1(x)
        T, B, C, H, W = x.shape
        # Split: first half as skip, second half through blocks
        x0, x1 = x[:, :, :C // 2], x[:, :, C // 2:]
        parts = [x0, x1]
        for block in self.blocks:
            x1 = block(x1)
            parts.append(x1)
        out = torch.cat(parts, dim=2)
        return self.cv2(out)


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------

class SpikeDownsample(nn.Module):
    """Strided convolution for spatial downsampling."""

    def __init__(self, in_ch, out_ch, decay=0.25, max_level=4):
        super().__init__()
        self.conv = SpikeConv(in_ch, out_ch, 3, 2, 1,
                               decay=decay, max_level=max_level)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Neck: PANet FPN
# ---------------------------------------------------------------------------

class SpikePANet(nn.Module):
    """PANet-style FPN neck with SNN blocks."""

    def __init__(self, channels, decay=0.25, max_level=4):
        super().__init__()
        c3, c4, c5 = channels

        # Top-down
        self.up5 = SpikeConv(c5, c4, 1, 1, 0, decay=decay, max_level=max_level)
        self.td4 = C2fSpike(c4 * 2, c4, num_blocks=1, decay=decay, max_level=max_level)
        self.up4 = SpikeConv(c4, c3, 1, 1, 0, decay=decay, max_level=max_level)
        self.td3 = C2fSpike(c3 * 2, c3, num_blocks=1, decay=decay, max_level=max_level)

        # Bottom-up
        self.down3 = SpikeDownsample(c3, c4, decay=decay, max_level=max_level)
        self.bu4 = C2fSpike(c4 * 2, c4, num_blocks=1, decay=decay, max_level=max_level)
        self.down4 = SpikeDownsample(c4, c5, decay=decay, max_level=max_level)
        self.bu5 = C2fSpike(c5 * 2, c5, num_blocks=1, decay=decay, max_level=max_level)

    def forward(self, features):
        """features: [p3, p4, p5] each (T, B, C, H, W)"""
        p3, p4, p5 = features

        # Top-down
        p5_up = self.up5(p5)
        T, B = p5_up.shape[:2]
        p5_flat = p5_up.flatten(0, 1)
        p5_flat = nn.functional.interpolate(p5_flat, size=p4.shape[3:], mode='nearest')
        p5_up = p5_flat.view(T, B, *p5_flat.shape[1:])
        p4 = self.td4(torch.cat([p4, p5_up], dim=2))

        p4_up = self.up4(p4)
        p4_flat = p4_up.flatten(0, 1)
        p4_flat = nn.functional.interpolate(p4_flat, size=p3.shape[3:], mode='nearest')
        p4_up = p4_flat.view(T, B, *p4_flat.shape[1:])
        p3 = self.td3(torch.cat([p3, p4_up], dim=2))

        # Bottom-up
        p3_down = self.down3(p3)
        p4 = self.bu4(torch.cat([p4, p3_down], dim=2))

        p4_down = self.down4(p4)
        p5 = self.bu5(torch.cat([p5, p4_down], dim=2))

        return [p3, p4, p5]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SpikeYOLO(nn.Module):
    """SpikeYOLO: YOLOv8-style spiking object detector with I-LIF neurons.

    Input: (B, C, H, W) single frame (model repeats T internally)
    Output: (B, num_boxes, 4 + num_classes) predictions

    Args:
        num_classes: number of detection classes
        in_channels: input image channels (default 3)
        T: number of temporal steps (default 4)
        img_size: input resolution (default 640)
        width_mult: channel width multiplier (0.25=nano, 0.5=small, 1.0=medium)
        depth_mult: block depth multiplier
        decay: I-LIF decay factor
        max_level: I-LIF maximum spike level
    """

    def __init__(self, num_classes=80, in_channels=3, T=4, img_size=640,
                 width_mult=0.25, depth_mult=0.33,
                 decay=0.25, max_level=4, **kwargs):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        # Channel widths (scaled by width_mult)
        base = [64, 128, 256, 512, 1024]
        channels = [max(int(c * width_mult), 16) for c in base]

        def n_blocks(n):
            return max(1, int(n * depth_mult))

        # Stem
        self.stem = SeqToANNContainer(
            nn.Conv2d(in_channels, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )
        self.stem_neuron = MultiStepILIFNeuron(decay=decay, max_level=max_level)

        # Backbone stages
        self.stage1 = nn.Sequential(
            SpikeDownsample(channels[0], channels[1], decay=decay, max_level=max_level),
            C2fSpike(channels[1], channels[1], n_blocks(3), decay=decay, max_level=max_level),
        )
        self.stage2 = nn.Sequential(
            SpikeDownsample(channels[1], channels[2], decay=decay, max_level=max_level),
            C2fSpike(channels[2], channels[2], n_blocks(6), decay=decay, max_level=max_level),
        )
        self.stage3 = nn.Sequential(
            SpikeDownsample(channels[2], channels[3], decay=decay, max_level=max_level),
            C2fSpike(channels[3], channels[3], n_blocks(6), decay=decay, max_level=max_level),
        )
        self.stage4 = nn.Sequential(
            SpikeDownsample(channels[3], channels[4], decay=decay, max_level=max_level),
            C2fSpike(channels[4], channels[4], n_blocks(3), decay=decay, max_level=max_level),
        )

        # Neck
        neck_channels = (channels[2], channels[3], channels[4])
        self.neck = SpikePANet(neck_channels, decay=decay, max_level=max_level)

        # Detection head (operates after temporal mean, standard 4D)
        self.detect = YOLOv8Detect(
            num_classes=num_classes,
            in_channels=neck_channels,
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image, or (T*B, C, H, W) if from TDL

        Returns:
            (B, total_boxes, 4 + num_classes) detection predictions
        """
        if x.dim() == 5:
            pass  # Already (T, B, C, H, W)
        else:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # Stem
        x = self.stem_neuron(self.stem(x))

        # Backbone
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = x                          # stride 8
        x = self.stage3(x)
        p4 = x                          # stride 16
        x = self.stage4(x)
        p5 = x                          # stride 32

        # Neck
        features = self.neck([p3, p4, p5])

        # Temporal mean: (T, B, C, H, W) → (B, C, H, W)
        features = [f.mean(dim=0) for f in features]

        # Detection head (no T dim)
        return self.detect(features)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def spike_yolo_n(num_classes=80, in_channels=3, T=4, **kwargs):
    """SpikeYOLO-Nano (width=0.25, depth=0.33)."""
    return SpikeYOLO(num_classes=num_classes, in_channels=in_channels, T=T,
                     width_mult=0.25, depth_mult=0.33, **kwargs)


def spike_yolo_s(num_classes=80, in_channels=3, T=4, **kwargs):
    """SpikeYOLO-Small (width=0.5, depth=0.33)."""
    return SpikeYOLO(num_classes=num_classes, in_channels=in_channels, T=T,
                     width_mult=0.5, depth_mult=0.33, **kwargs)


def spike_yolo_m(num_classes=80, in_channels=3, T=4, **kwargs):
    """SpikeYOLO-Medium (width=0.75, depth=0.67)."""
    return SpikeYOLO(num_classes=num_classes, in_channels=in_channels, T=T,
                     width_mult=0.75, depth_mult=0.67, **kwargs)


def build_spike_yolo(config):
    """Build SpikeYOLO from config dict."""
    return SpikeYOLO(
        num_classes=config.get('num_classes', 80),
        in_channels=config.get('in_channels', 3),
        T=config.get('T', 4),
        img_size=config.get('img_size', 640),
        width_mult=config.get('width_mult', 0.25),
        depth_mult=config.get('depth_mult', 0.33),
        decay=config.get('decay', 0.25),
        max_level=config.get('max_level', 4),
    )
