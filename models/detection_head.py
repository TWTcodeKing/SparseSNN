"""
Detection head utilities for spiking YOLO models.

Provides anchor-free (YOLOv8-style DFL) and anchor-based (YOLOv3-style)
detection heads that operate AFTER temporal mean (no T dimension).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL(nn.Module):
    """Distribution Focal Loss layer (YOLOv8 box regression).

    Converts discrete distribution over reg_max bins to continuous coordinates.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        # Fixed weights: [0, 1, 2, ..., reg_max-1]
        self.conv.weight.data[:] = torch.arange(reg_max, dtype=torch.float32).view(
            1, reg_max, 1, 1)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        """x: (B, 4*reg_max, H, W) → (B, 4, H, W)"""
        B, _, H, W = x.shape
        x = x.view(B, 4, self.reg_max, H, W)
        x = x.softmax(2)
        x = x.view(B * 4, self.reg_max, H, W)
        x = self.conv(x)
        return x.view(B, 4, H, W)


class YOLOv8Detect(nn.Module):
    """YOLOv8-style anchor-free detection head with DFL.

    Takes multi-scale feature maps, predicts boxes + classes per scale.
    Operates on standard 4D tensors (B, C, H, W) — no temporal dimension.

    Args:
        num_classes: number of object classes
        in_channels: list of input channels for each scale [P3, P4, P5]
        reg_max: DFL distribution bins (default 16)
    """

    def __init__(self, num_classes=80, in_channels=(256, 512, 1024), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_scales = len(in_channels)

        # Per-scale box and class convolutions
        self.box_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        box_out = 4 * reg_max
        cls_out = num_classes

        for ch in in_channels:
            self.box_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
                nn.Conv2d(ch, box_out, 1),
            ))
            self.cls_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
                nn.Conv2d(ch, cls_out, 1),
            ))

        self.dfl = DFL(reg_max)

    def forward(self, features):
        """
        Args:
            features: list of (B, C_i, H_i, W_i) feature maps from neck

        Returns:
            (B, total_boxes, 4 + num_classes) concatenated predictions
        """
        all_boxes = []
        all_cls = []

        for i, feat in enumerate(features):
            box = self.box_convs[i](feat)       # (B, 4*reg_max, H, W)
            cls = self.cls_convs[i](feat)       # (B, num_classes, H, W)

            B, _, H, W = box.shape
            box = self.dfl(box)                 # (B, 4, H, W)
            box = box.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)

            all_boxes.append(box)
            all_cls.append(cls)

        boxes = torch.cat(all_boxes, dim=1)     # (B, total, 4)
        classes = torch.cat(all_cls, dim=1)     # (B, total, num_classes)
        return torch.cat([boxes, classes], dim=-1)  # (B, total, 4+num_classes)


class YOLOv3Detect(nn.Module):
    """YOLOv3-style anchor-based detection head.

    Takes multi-scale feature maps, predicts (x, y, w, h, obj, classes) per anchor.
    Operates on standard 4D tensors (B, C, H, W) — no temporal dimension.

    Args:
        num_classes: number of object classes
        in_channels: list of input channels for each scale
        anchors_per_scale: number of anchors per scale (default 3)
    """

    def __init__(self, num_classes=80, in_channels=(256, 512, 1024),
                 anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.num_scales = len(in_channels)
        out_ch = anchors_per_scale * (5 + num_classes)

        self.heads = nn.ModuleList([
            nn.Conv2d(ch, out_ch, 1) for ch in in_channels
        ])

    def forward(self, features):
        """
        Args:
            features: list of (B, C_i, H_i, W_i) feature maps

        Returns:
            (B, total_boxes, 5 + num_classes) concatenated predictions
        """
        all_preds = []
        na = self.anchors_per_scale
        out_dim = 5 + self.num_classes

        for i, feat in enumerate(features):
            pred = self.heads[i](feat)          # (B, na*(5+cls), H, W)
            B, _, H, W = pred.shape
            pred = pred.view(B, na, out_dim, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).reshape(B, na * H * W, out_dim)
            all_preds.append(pred)

        return torch.cat(all_preds, dim=1)  # (B, total, 5+num_classes)
