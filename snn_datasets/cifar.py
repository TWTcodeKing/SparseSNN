"""
CIFAR-10 / CIFAR-100 dataset loaders.

Two augmentation modes:
- Standard SNN: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- SNN RandAugment (auto_aug=True): RandomHorizontalFlip + SNN-reduced RandAugment
  (Equalize, Rotate, ShearX, TranslateX/Y only) + RandomErasing(p=0.25)
  Matching the MaxFormer/QKFormer official training pipeline.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class SNNRandAugment:
    """RandAugment with SNN-friendly ops only (no color/brightness/contrast).

    Matches MaxFormer official: rand-m9-n1-mstd0.4-inc1 with only
    Equalize, Rotate, ShearX, TranslateXRel, TranslateYRel kept.
    """

    def __init__(self, magnitude=9, num_ops=1, magnitude_std=0.4):
        self.magnitude = magnitude
        self.num_ops = num_ops
        self.magnitude_std = magnitude_std
        self.ops = [
            self._equalize,
            self._rotate,
            self._shear_x,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img):
        import random
        for _ in range(self.num_ops):
            mag = self.magnitude
            if self.magnitude_std > 0:
                mag = max(0, random.gauss(mag, self.magnitude_std))
            op = random.choice(self.ops)
            img = op(img, mag)
        return img

    def _equalize(self, img, mag):
        from torchvision.transforms.functional import equalize
        return equalize(img)

    def _rotate(self, img, mag):
        from torchvision.transforms.functional import rotate
        angle = (mag / 10.0) * 30.0  # max 30 degrees at mag=10
        import random
        if random.random() < 0.5:
            angle = -angle
        return rotate(img, angle)

    def _shear_x(self, img, mag):
        from torchvision.transforms.functional import affine
        import random
        shear = (mag / 10.0) * 0.3 * (1 if random.random() > 0.5 else -1)
        import math
        return affine(img, angle=0.0, translate=[0, 0], scale=1.0,
                      shear=[math.degrees(math.atan(shear)), 0.0])

    def _translate_x(self, img, mag):
        from torchvision.transforms.functional import affine
        import random
        pixels = int((mag / 10.0) * 10 * (1 if random.random() > 0.5 else -1))
        return affine(img, angle=0.0, translate=[pixels, 0], scale=1.0, shear=[0.0, 0.0])

    def _translate_y(self, img, mag):
        from torchvision.transforms.functional import affine
        import random
        pixels = int((mag / 10.0) * 10 * (1 if random.random() > 0.5 else -1))
        return affine(img, angle=0.0, translate=[0, pixels], scale=1.0, shear=[0.0, 0.0])


def _build_cifar_transforms(img_size, mean, std, auto_aug=False, cutout=False):
    if auto_aug:
        # SNN-reduced RandAugment pipeline (matches MaxFormer/QKFormer official)
        train_t = [
            transforms.RandomHorizontalFlip(),
            SNNRandAugment(magnitude=9, num_ops=1, magnitude_std=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, value=0, inplace=True),
        ]
    else:
        # Standard SNN augmentation
        train_t = [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    if img_size != 32:
        train_t.insert(0, transforms.Resize(img_size))

    test_t = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if img_size != 32:
        test_t.insert(0, transforms.Resize(img_size))

    return transforms.Compose(train_t), transforms.Compose(test_t)


def cifar10_dataloaders(data_root, batch_size, img_size=32, num_workers=4,
                        auto_aug=False, cutout=False, distributed=False):
    """
    Returns (train_loader, test_loader) for CIFAR-10.

    Args:
        data_root: path to store/load dataset
        batch_size: batch size per GPU
        img_size: resize images to this size (default 32, no resize)
        num_workers: dataloader workers
        auto_aug: use SNN-reduced RandAugment + RandomErasing
        cutout: apply extra Cutout (16x16 erasing)
        distributed: use DistributedSampler
    """
    train_transform, test_transform = _build_cifar_transforms(
        img_size, CIFAR10_MEAN, CIFAR10_STD, auto_aug)

    if cutout and not auto_aug:
        train_transform = transforms.Compose([
            *train_transform.transforms,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.08)),
        ])

    train_set = datasets.CIFAR10(root=data_root, train=True,
                                 download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False,
                                download=True, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def cifar100_dataloaders(data_root, batch_size, img_size=32, num_workers=4,
                         auto_aug=False, cutout=False, distributed=False):
    """
    Returns (train_loader, test_loader) for CIFAR-100.
    """
    train_transform, test_transform = _build_cifar_transforms(
        img_size, CIFAR100_MEAN, CIFAR100_STD, auto_aug)

    if cutout and not auto_aug:
        train_transform = transforms.Compose([
            *train_transform.transforms,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.08)),
        ])

    train_set = datasets.CIFAR100(root=data_root, train=True,
                                  download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_root, train=False,
                                 download=True, transform=test_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              num_workers=num_workers, pin_memory=True, sampler=train_sampler,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
