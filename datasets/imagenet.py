"""
ImageNet (ILSVRC2012) dataset loader.

Expects the standard directory layout:
    data_root/
        train/
            n01440764/
                *.JPEG
            ...
        val/
            n01440764/
                *.JPEG
            ...

Standard augmentation pipeline:
- Train: RandomResizedCrop(224) + RandomHorizontalFlip + Normalize
- Val: Resize(256) + CenterCrop(224) + Normalize
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_dataloaders(data_root, batch_size, img_size=224, num_workers=8,
                         auto_aug=False, distributed=False):
    """
    Returns (train_loader, val_loader) for ImageNet.

    Args:
        data_root: path to ImageNet root (containing train/ and val/)
        batch_size: batch size per GPU
        img_size: crop size (default 224)
        num_workers: dataloader workers
        auto_aug: use AutoAugment (ImageNet policy)
        distributed: use DistributedSampler
    """
    train_t = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_aug:
        train_t.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    train_t += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    resize_size = int(img_size / 0.875)  # 256 for img_size=224
    val_t = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    train_set = datasets.ImageFolder(
        os.path.join(data_root, 'train'),
        transform=transforms.Compose(train_t),
    )
    val_set = datasets.ImageFolder(
        os.path.join(data_root, 'val'),
        transform=transforms.Compose(val_t),
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
