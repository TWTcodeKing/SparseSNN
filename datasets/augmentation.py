"""
Data augmentation utilities for SNN training.

Implements MixUp and CutMix as batch-level transforms (applied after dataloader).
These are used by the training loop, not by the dataloader transforms.
"""

import torch
import numpy as np


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

    _, _, H, W = x.shape
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
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp/CutMix augmented batch."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
