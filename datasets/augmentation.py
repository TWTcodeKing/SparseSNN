"""
Data augmentation utilities for SNN training.

Implements MixUp and CutMix as batch-level transforms (applied after dataloader).
These are used by the training loop, not by the dataloader transforms.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF


class SNNAugmentWide(nn.Module):
    """TrivialAugmentWide variant for neuromorphic/SNN data.

    Only geometric + cutout ops (no color/brightness/contrast since DVS
    has no intensity). Applied per-sample on (T, C, H, W) tensors.

    Source: https://github.com/bic-L/MaxFormer event/snn_aug.py
    """

    def __init__(self, num_bins=31):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, img):
        # img: (T, C, H, W)
        fill = [0.0] * img.shape[-3]
        op_meta = self._augmentation_space(self.num_bins)
        op_name = list(op_meta.keys())[torch.randint(len(op_meta), (1,)).item()]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,))].item())
            if len(magnitudes) > 0 else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        # Apply to each timestep
        frames = []
        for t in range(img.shape[0]):
            frames.append(self._apply_op(img[t], op_name, magnitude, fill))
        return torch.stack(frames)

    def _augmentation_space(self, num_bins):
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True),
        }

    def _apply_op(self, img, op_name, magnitude, fill):
        if op_name == "Identity":
            return img
        elif op_name == "ShearX":
            return TF.affine(img, angle=0.0, translate=[0, 0],
                             scale=1.0, shear=[math.degrees(math.atan(magnitude)), 0.0],
                             fill=fill)
        elif op_name == "TranslateX":
            return TF.affine(img, angle=0.0, translate=[int(magnitude), 0],
                             scale=1.0, shear=[0.0, 0.0], fill=fill)
        elif op_name == "TranslateY":
            return TF.affine(img, angle=0.0, translate=[0, int(magnitude)],
                             scale=1.0, shear=[0.0, 0.0], fill=fill)
        elif op_name == "Rotate":
            return TF.rotate(img, magnitude, fill=fill)
        elif op_name == "Cutout":
            # RandomErasing-style cutout
            return transforms.RandomErasing(
                p=1.0, scale=(0.001, 0.11), ratio=(1, 1))(img)
        return img


def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation to a batch.

    Args:
        x: input images (B, C, H, W)
        y: labels (B,)
        alpha: Beta distribution parameter. 0 disables mixup.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to a batch.

    Args:
        x: input images (B, C, H, W)
        y: labels (B,)
        alpha: Beta distribution parameter. 0 disables cutmix.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    H, W = x.shape[-2], x.shape[-1]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Uniform random center
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[..., y1:y2, x1:x2] = x[index][..., y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def snn_aug_batch(images, hflip=True, snn_aug=None):
    """Apply per-sample DVS augmentation on a (B, T, C, H, W) batch.

    Args:
        images: (B, T, C, H, W) tensor
        hflip: apply random horizontal flip per sample
        snn_aug: SNNAugmentWide instance (or None to skip)
    """
    augmented = []
    for i in range(images.shape[0]):
        x = images[i]  # (T, C, H, W)
        if hflip and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        if snn_aug is not None:
            x = snn_aug(x)
        augmented.append(x)
    return torch.stack(augmented)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp/CutMix augmented batch."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
