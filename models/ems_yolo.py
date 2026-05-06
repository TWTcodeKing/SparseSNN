"""
EMS-YOLO: Spiking Neural Network Object Detection based on YOLOv3.

Uses MSNeuron (decay=0.25, thresh=0.5) + TDBNContainer (Conv2d + BN3d)
from the MS-ResNet infrastructure, since EMS-YOLO's neuron dynamics are
identical (mem = mem * decay * (1-spike) + x).

Backbone: Spiking ResNet-34 (Darknet-style with residual blocks)
Neck: YOLOv3 FPN (3 scales)
Head: Anchor-based detection (YOLOv3Detect)

Reference: https://github.com/BICLab/EMS-YOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.msresnet import MSNeuron, TDBNContainer, _DECAY, _THRESH
from models.detection_head import YOLOv3Detect


def _make_conv(in_ch, out_ch, kernel=3, stride=1, padding=1, bn_init_thresh=True):
    """Helper: Conv2d wrapped in TDBNContainer + MSNeuron."""
    conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
    return TDBNContainer(conv, out_ch, bn_init_thresh=bn_init_thresh)


class EMSBasicBlock(nn.Module):
    """Residual block for EMS-YOLO backbone.

    Pattern: Conv3x3 → Neuron → Conv3x3 → Neuron + skip
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _make_conv(in_ch, out_ch, 3, stride, 1)
        self.sn1 = MSNeuron()
        self.conv2 = _make_conv(out_ch, out_ch, 3, 1, 1)
        self.sn2 = MSNeuron()
        self.downsample = downsample

    def forward(self, x):
        """x: (T, B, C, H, W)"""
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.sn2(out)
        return out


class EMSBackbone(nn.Module):
    """Spiking ResNet-34 backbone for EMS-YOLO.

    Returns multi-scale features [P3, P4, P5] from layer2, layer3, layer4.
    """

    def __init__(self, in_channels=3, T=4):
        super().__init__()
        self.T = T

        # Stem
        self.conv1 = _make_conv(in_channels, 64, 7, 2, 3)
        self.sn1 = MSNeuron()

        # Residual layers (ResNet-34: [3, 4, 6, 3])
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                _make_conv(in_ch, out_ch, 1, stride, 0),
                MSNeuron(),
            )
        layers = [EMSBasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(EMSBasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """x: (T, B, C, H, W) → list of 3 feature maps [(T,B,C_i,H_i,W_i)]"""
        x = self.sn1(self.conv1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        p3 = x                      # stride 8
        x = self.layer3(x)
        p4 = x                      # stride 16
        x = self.layer4(x)
        p5 = x                      # stride 32

        return [p3, p4, p5]


class EMSFPN(nn.Module):
    """YOLOv3-style FPN neck with lateral connections."""

    def __init__(self, in_channels=(128, 256, 512)):
        super().__init__()
        # Top-down pathway
        self.lateral5 = _make_conv(in_channels[2], in_channels[1], 1, 1, 0)
        self.sn5 = MSNeuron()
        self.smooth4 = _make_conv(in_channels[1], in_channels[1], 3, 1, 1)
        self.sn4 = MSNeuron()

        self.lateral4 = _make_conv(in_channels[1], in_channels[0], 1, 1, 0)
        self.sn4_lat = MSNeuron()
        self.smooth3 = _make_conv(in_channels[0], in_channels[0], 3, 1, 1)
        self.sn3 = MSNeuron()

    def forward(self, features):
        """features: [p3, p4, p5] each (T, B, C, H, W)"""
        p3, p4, p5 = features

        # Top-down from P5 → P4
        p5_up = self.sn5(self.lateral5(p5))
        T, B = p5_up.shape[:2]
        p5_up_flat = p5_up.flatten(0, 1)
        p5_up_flat = F.interpolate(p5_up_flat, size=p4.shape[3:], mode='nearest')
        p5_up = p5_up_flat.view(T, B, *p5_up_flat.shape[1:])
        p4 = self.sn4(self.smooth4(p4 + p5_up))

        # Top-down from P4 → P3
        p4_up = self.sn4_lat(self.lateral4(p4))
        p4_up_flat = p4_up.flatten(0, 1)
        p4_up_flat = F.interpolate(p4_up_flat, size=p3.shape[3:], mode='nearest')
        p4_up = p4_up_flat.view(T, B, *p4_up_flat.shape[1:])
        p3 = self.sn3(self.smooth3(p3 + p4_up))

        return [p3, p4, p5]


class EMSYOLO(nn.Module):
    """EMS-YOLO: Spiking object detection with MS-ResNet backbone.

    Input: (B, C, H, W) single frame (model repeats T internally)
    Output: (B, num_boxes, 5 + num_classes) predictions

    Args:
        num_classes: number of detection classes
        in_channels: input image channels (default 3)
        T: number of temporal steps (default 4)
        img_size: input resolution (for reference only)
    """

    def __init__(self, num_classes=80, in_channels=3, T=4, img_size=416, **kwargs):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        self.backbone = EMSBackbone(in_channels=in_channels, T=T)
        self.neck = EMSFPN(in_channels=(128, 256, 512))
        self.detect = YOLOv3Detect(
            num_classes=num_classes,
            in_channels=(128, 256, 512),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image, or (T*B, C, H, W) if from TDL

        Returns:
            (B, total_boxes, 5 + num_classes) detection predictions
        """
        if x.dim() == 5:
            pass  # Already (T, B, C, H, W)
        else:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # Backbone → multi-scale features
        features = self.backbone(x)

        # Neck → enhanced features
        features = self.neck(features)

        # Temporal mean: (T, B, C, H, W) → (B, C, H, W)
        features = [f.mean(dim=0) for f in features]

        # Detection head (operates on 4D, no T)
        return self.detect(features)


def ems_yolo_res34(num_classes=80, in_channels=3, T=4, **kwargs):
    """Factory function for EMS-YOLO with ResNet-34 backbone."""
    return EMSYOLO(num_classes=num_classes, in_channels=in_channels, T=T, **kwargs)


def build_ems_yolo(config):
    """Build EMS-YOLO from config dict."""
    return EMSYOLO(
        num_classes=config.get('num_classes', 80),
        in_channels=config.get('in_channels', 3),
        T=config.get('T', 4),
        img_size=config.get('img_size', 416),
    )
