"""
Prophesee Gen1 Automotive Detection Dataset loader.

Gen1 is an event-camera (DVS) object detection dataset with 2 classes
(car, pedestrian) at 304x240 resolution. Events are preprocessed into
temporal frame stacks stored as .npy files with YOLO-format .txt labels.

Preprocessing (run once):
    python datasets/gen1_preprocess.py --raw-dir /data/gen1_raw --out-dir /data/gen1

Expected directory structure after preprocessing:
    {data_root}/gen1/
        train/
            images/     (*.npy files, shape [T, H, W, 3] uint8)
            labels/     (*.txt files, YOLO format: cls cx cy w h)
        val/
            images/
            labels/
        test/
            images/
            labels/

Raw Gen1 format (before preprocessing):
    - Events: .dat files (Prophesee format, loaded via PSEELoader)
    - Annotations: .npy structured arrays with fields (t, x, y, w, h, class_id, track_id)
    - 2 classes: 0=car, 1=pedestrian
    - Resolution: 304 width x 240 height
    - Events accumulated into T frames per sample (default T=5, 50ms per frame)
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


GEN1_CLASSES = ('car', 'pedestrian')
GEN1_NUM_CLASSES = 2
GEN1_WIDTH = 304
GEN1_HEIGHT = 240


class Gen1DetectionDataset(Dataset):
    """Gen1 DVS detection dataset (preprocessed .npy frames + .txt labels).

    Returns:
        image: (C, H, W) float tensor where C = T * 3 (T event frames stacked as channels)
               OR (T, C, H, W) if return_temporal=True
        target: dict with 'boxes' (N, 4) [cx, cy, w, h] normalized, 'labels' (N,) long
    """

    def __init__(self, root, split='train', img_size=320, augment=True,
                 return_temporal=True):
        """
        Args:
            root: path to gen1/ directory (with train/val/test subdirs)
            split: 'train', 'val', or 'test'
            img_size: resize target (square), default 320
            augment: apply data augmentation
            return_temporal: if True, return (T, 3, H, W); else (T*3, H, W)
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.return_temporal = return_temporal

        img_dir = os.path.join(root, split, 'images')
        label_dir = os.path.join(root, split, 'labels')

        # Find all .npy image files
        self.img_files = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.endswith('.npy')
        ])
        self.label_dir = label_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # Derive label path: images/xxx.npy → labels/xxx.txt
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, basename + '.txt')

        # Load event frames: (T, H, W, 3) or (T, W, H, 3) uint8
        frames = np.load(img_path)  # typically (T, 304, 240, 3) or (T, 240, 304, 3)

        # Normalize to [0, 1] float
        frames = frames.astype(np.float32) / 255.0

        # Handle both (T, W, H, 3) and (T, H, W, 3) — standardize to (T, H, W, 3)
        if frames.shape[1] == GEN1_WIDTH and frames.shape[2] == GEN1_HEIGHT:
            # (T, W=304, H=240, 3) → transpose to (T, H=240, W=304, 3)
            frames = frames.transpose(0, 2, 1, 3)

        T = frames.shape[0]

        # Convert to torch: (T, H, W, 3) → (T, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        # Resize to target size
        if self.img_size != frames.shape[2] or self.img_size != frames.shape[3]:
            frames = torch.nn.functional.interpolate(
                frames, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False)

        # Load labels (YOLO format: class_id cx cy w h, normalized)
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), \
                                       float(parts[3]), float(parts[4])
                        boxes.append([cx, cy, w, h])
                        labels.append(cls_id)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                frames = frames.flip(-1)  # horizontal flip
                boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32).clamp(0, 1)
            labels_t = torch.tensor(labels, dtype=torch.long)

        target = {'boxes': boxes_t, 'labels': labels_t}

        if self.return_temporal:
            # (T, 3, H, W) — compatible with SNN temporal input
            return frames, target
        else:
            # (T*3, H, W) — flatten temporal into channels
            return frames.reshape(-1, self.img_size, self.img_size), target


def gen1_collate_fn(batch):
    """Custom collate for variable-length box annotations."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def gen1_dataloaders(data_root, batch_size, img_size=320,
                     num_workers=4, distributed=False, **kwargs):
    """Build Gen1 train/val dataloaders.

    Args:
        data_root: path containing 'gen1/' subdirectory
        batch_size: batch size
        img_size: resize target (default 320)
        num_workers: dataloader workers
        distributed: use DistributedSampler

    Returns:
        (train_loader, val_loader)
    """
    gen1_root = os.path.join(data_root, 'gen1')

    train_ds = Gen1DetectionDataset(gen1_root, 'train', img_size, augment=True)
    val_ds = Gen1DetectionDataset(gen1_root, 'val', img_size, augment=False)

    train_sampler = None
    val_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=gen1_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=gen1_collate_fn,
    )
    return train_loader, val_loader
