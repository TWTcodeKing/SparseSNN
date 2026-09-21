"""
UT-HAR: WiFi CSI-based human activity recognition (SenseFi preprocessed version).

Reference:
  S. Yousefi, H. Narui, S. Dayal, S. Ermon, S. Valaee, "A Survey on Behavior
  Recognition Using WiFi Channel State Information", IEEE Communications
  Magazine, 2017.  Preprocessed split from the SenseFi benchmark
  (https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).

7 activities (lie down, fall, walk, pick up, run, sit down, stand up).
Each sample is a CSI amplitude window of 250 packets x 90 subcarriers
(3 antennas x 30 subcarriers, Intel 5300 NIC), used as a single-channel
static "image" (1, 250, 90).  Splits: train 3977 / val 496 / test 500.

Data organization (the .csv files are actually numpy .npy binaries):
  <data_root>/UT_HAR/data/X_{train,val,test}.csv    float64 (N, 250, 90)
  <data_root>/UT_HAR/label/y_{train,val,test}.csv   int64   (N,)

Normalization follows SenseFi: per-split min-max scaling to [0, 1].
The model repeats the static input across T timesteps (direct encoding).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

UT_HAR_CLASSES = ['lie down', 'fall', 'walk', 'pick up', 'run', 'sit down', 'stand up']


def _load_split(root, split):
    x_path = os.path.join(root, 'data', f'X_{split}.csv')
    y_path = os.path.join(root, 'label', f'y_{split}.csv')
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"UT-HAR split '{split}' not found under {root}. Expected "
            f"{x_path} and {y_path} (SenseFi UT_HAR.zip layout).")
    with open(x_path, 'rb') as f:
        x = np.load(f)
    with open(y_path, 'rb') as f:
        y = np.load(f)
    x = x.reshape(len(x), 1, 250, 90).astype(np.float32)
    # SenseFi normalization: per-split min-max to [0, 1]
    x = (x - x.min()) / (x.max() - x.min())
    return x, y.astype(np.int64)


class UTHARDataset(Dataset):
    """In-memory UT-HAR split. Returns ((1, 250, 90) float tensor, label)."""

    def __init__(self, data_root, split='train'):
        root = os.path.join(data_root, 'UT_HAR')
        self.x, self.y = _load_split(root, split)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(self.y[idx])


def ut_har_dataloaders(data_root, batch_size=64, num_workers=4,
                       distributed=False, test_split='test', **kwargs):
    """(train_loader, test_loader) for UT-HAR.

    test_split: 'test' (500 samples, default) or 'val' (496 samples).
    Extra kwargs (auto_aug, cutout, ...) are accepted and ignored.
    """
    train_set = UTHARDataset(data_root, 'train')
    test_set = UTHARDataset(data_root, test_split)

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_set)
                     if distributed else None)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler,
        drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
