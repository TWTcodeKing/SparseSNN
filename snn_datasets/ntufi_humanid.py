"""
NTU-Fi-HumanID: WiFi CSI-based human gait identification dataset.

14 subjects, CSI amplitude from Atheros WiFi (3 antenna pairs × 114 subcarriers).
Each .mat file contains one gait sample: CSIamp field with shape (3*114, 2000).
Downsampled 4x to (3, 114, 500), then reshaped to 2D spatial input for CNN.

For SNN-VGG-9 with T timesteps:
  - CSI time dimension (500) is chunked into T frames of length 500//T
  - Each frame: (3, 114, 500//T) → resize to (3, H, W) for VGG
  - Output per sample: (T, 3, H, W)

Data organization:
  <data_root>/NTU-Fi-HumanID/train_amp/<class_id>/*.mat
  <data_root>/NTU-Fi-HumanID/test_amp/<class_id>/*.mat

Reference: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
"""

import os
import glob
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader


# Normalization constants from the benchmark
CSI_MEAN = 42.3199
CSI_STD = 4.9802


class NTUFiHumanIDDataset(Dataset):
    """NTU-Fi-HumanID WiFi CSI dataset.

    Args:
        root_dir: path to train_amp/ or test_amp/ directory
        T: number of SNN timesteps (splits CSI time dim into T frames)
        spatial_size: (H, W) resize target for VGG spatial input
        modal: 'CSIamp' or 'CSIphase'
    """

    def __init__(self, root_dir, T=4, spatial_size=(32, 32), modal='CSIamp'):
        self.root_dir = root_dir
        self.T = T
        self.spatial_size = spatial_size
        self.modal = modal

        self.data_list = sorted(glob.glob(os.path.join(root_dir, '*', '*.mat')))
        self.folders = sorted(glob.glob(os.path.join(root_dir, '*/')))
        self.category = {
            os.path.basename(f.rstrip('/')): i
            for i, f in enumerate(self.folders)
        }

        if not self.data_list:
            raise FileNotFoundError(
                f"No .mat files found in {root_dir}. "
                f"Download NTU-Fi-HumanID from: "
                f"https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample_path = self.data_list[idx]
        # Label from parent folder name
        class_name = os.path.basename(os.path.dirname(sample_path))
        label = self.category[class_name]

        # Load CSI amplitude from .mat
        mat = sio.loadmat(sample_path)
        x = mat[self.modal].astype(np.float32)  # (342, 2000) or (3*114, 2000)

        # Normalize
        x = (x - CSI_MEAN) / CSI_STD

        # Downsample time: 2000 → 500
        x = x[:, ::4]  # (342, 500)

        # Reshape to (3, 114, 500)
        x = x.reshape(3, 114, 500)

        # Split time dimension into T frames: (3, 114, 500) → (T, 3, 114, 500//T)
        frame_len = 500 // self.T
        frames = []
        for t in range(self.T):
            frame = x[:, :, t * frame_len : (t + 1) * frame_len]  # (3, 114, frame_len)
            # Resize to spatial_size using simple interpolation
            frame = self._resize(frame, self.spatial_size)  # (3, H, W)
            frames.append(frame)

        # Stack: (T, 3, H, W)
        x_tensor = torch.FloatTensor(np.stack(frames, axis=0))
        return x_tensor, label

    @staticmethod
    def _resize(frame, size):
        """Resize (C, H_in, W_in) to (C, H, W) via bilinear interpolation."""
        C, H_in, W_in = frame.shape
        H, W = size
        # Use torch for resize then back to numpy
        t = torch.FloatTensor(frame).unsqueeze(0)  # (1, C, H_in, W_in)
        t = torch.nn.functional.interpolate(t, size=(H, W), mode='bilinear',
                                            align_corners=False)
        return t.squeeze(0).numpy()  # (C, H, W)


def ntufi_humanid_dataloaders(data_root, batch_size=64, T=4,
                               spatial_size=(32, 32),
                               num_workers=4, distributed=False):
    """Create train/test dataloaders for NTU-Fi-HumanID.

    Args:
        data_root: root directory containing NTU-Fi-HumanID/
        batch_size: batch size per GPU
        T: SNN timesteps
        spatial_size: (H, W) for VGG input
        num_workers: dataloader workers
        distributed: use DistributedSampler

    Returns:
        (train_loader, test_loader)
    """
    train_dir = os.path.join(data_root, 'NTU-Fi-HumanID', 'train_amp')
    test_dir = os.path.join(data_root, 'NTU-Fi-HumanID', 'test_amp')

    train_set = NTUFiHumanIDDataset(train_dir, T=T, spatial_size=spatial_size)
    test_set = NTUFiHumanIDDataset(test_dir, T=T, spatial_size=spatial_size)

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
        pin_memory=True,
    )

    return train_loader, test_loader
