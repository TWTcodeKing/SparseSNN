"""
DVS128 Gesture neuromorphic dataset loader.

Uses spikingjelly's DVS128Gesture to load event data integrated into frames.
Unlike CIFAR10-DVS, this dataset has an official train/test split.

11 gesture classes, output shape per sample: (T, C, H, W)
where T=frames_number, C=2, H=W=128.
"""

import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import dvs128_gesture


def dvs128gesture_dataloaders(data_root, batch_size, frames_number=16,
                              split_by='number', num_workers=4,
                              distributed=False, auto_aug=False,
                              cutout=False):
    """
    Returns (train_loader, test_loader) for DVS128 Gesture.

    Args:
        data_root: path to store/load DVS128 Gesture dataset
        batch_size: batch size per GPU
        frames_number: number of frames to integrate events into (T)
        split_by: 'number' (equal event count per frame) or 'time'
        num_workers: dataloader workers
        distributed: use DistributedSampler
    """
    train_set = dvs128_gesture.DVS128Gesture(
        root=data_root, train=True, data_type='frame',
        frames_number=frames_number, split_by=split_by,
    )
    test_set = dvs128_gesture.DVS128Gesture(
        root=data_root, train=False, data_type='frame',
        frames_number=frames_number, split_by=split_by,
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
