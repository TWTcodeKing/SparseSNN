"""
CIFAR10-DVS neuromorphic dataset loader.

Uses spikingjelly's CIFAR10DVS to load event data integrated into frames.
The dataset has no official train/test split, so we split 90/10 per class
following Spikformer (https://github.com/ZK-Zhou/spikformer).

Output shape per sample: (T, C, H, W) where T=frames_number, C=2, H=W=128.
"""

import torch
from torch.utils.data import DataLoader, Subset
from spikingjelly.datasets import cifar10_dvs
from spikingjelly.datasets import split_to_train_test_set



def cifar10dvs_dataloaders(data_root, batch_size, frames_number=16,
                           split_by='number', train_ratio=0.9,
                           num_workers=4, distributed=False, 
                           auto_aug=False,cutout=False):
    """
    Returns (train_loader, test_loader) for CIFAR10-DVS.

    Args:
        data_root: path to store/load CIFAR10-DVS dataset
        batch_size: batch size per GPU
        frames_number: number of frames to integrate events into (T)
        split_by: 'number' (equal event count per frame) or 'time'
        train_ratio: fraction of data used for training (default 0.9)
        num_workers: dataloader workers
        distributed: use DistributedSampler
    """
    origin_set = cifar10_dvs.CIFAR10DVS(
        root=data_root, data_type='frame',
        frames_number=frames_number, split_by=split_by,
    )

    train_set, test_set = split_to_train_test_set(
        train_ratio, origin_set, num_classes=10,
    )

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_set)
                     if distributed else None)

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )

    return train_loader, test_loader
