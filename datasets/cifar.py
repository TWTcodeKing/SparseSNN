"""
CIFAR-10 / CIFAR-100 dataset loaders.

Standard augmentation pipeline following common SNN training practice:
- Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- Test: Normalize only
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2470, 0.2435, 0.2616)


def _build_cifar_transforms(img_size, mean, std, auto_aug=False):
    train_t = [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
    ]
    if auto_aug:
        train_t.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    train_t += [
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
        auto_aug: use AutoAugment
        cutout: apply Cutout (16x16 erasing)
        distributed: use DistributedSampler
    """
    train_transform, test_transform = _build_cifar_transforms(
        img_size, CIFAR10_MEAN, CIFAR10_STD, auto_aug)

    if cutout:
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

    if cutout:
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
