"""
COCO dataset loader for spiking object detection models.

Provides train/val dataloaders with standard YOLO-style augmentations.
Requires: pycocotools, installed via `uv pip install pycocotools`

Dataset structure expected:
    {data_root}/coco/
        train2017/       (118k images)
        val2017/         (5k images)
        annotations/
            instances_train2017.json
            instances_val2017.json
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)


class COCODetectionDataset(Dataset):
    """COCO detection dataset with YOLO-style augmentations.

    Returns:
        image: (3, H, W) float tensor (normalized)
        target: dict with 'boxes' (N, 4) [cx, cy, w, h] normalized,
                'labels' (N,) long tensor
    """

    def __init__(self, root, split='train', img_size=640, augment=True):
        from pycocotools.coco import COCO

        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.split = split

        ann_file = os.path.join(root, 'annotations',
                                f'instances_{split}2017.json')
        self.coco = COCO(ann_file)
        self.img_dir = os.path.join(root, f'{split}2017')

        # Filter images with annotations
        self.ids = sorted(self.coco.getImgIds())
        # Map category IDs to contiguous 0-79
        cats = sorted(self.coco.getCatIds())
        self.cat_to_label = {c: i for i, c in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: (x, y, w, h) absolute
            if w < 1 or h < 1:
                continue
            # Convert to (cx, cy, w, h) normalized
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_to_label[ann['category_id']])

        # Resize to target size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

            # HSV jitter
            img = TF.adjust_hue(img, random.uniform(-0.015, 0.015))
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))

        # To tensor + normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, COCO_MEAN, COCO_STD)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32).clamp(0, 1)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels}
        return img, target


def coco_collate_fn(batch):
    """Custom collate for variable-length box annotations."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def coco_dataloaders(data_root, batch_size, img_size=640,
                     num_workers=4, distributed=False, **kwargs):
    """Build COCO train/val dataloaders.

    Args:
        data_root: path containing 'coco/' subdirectory
        batch_size: batch size
        img_size: input image size
        num_workers: dataloader workers
        distributed: use DistributedSampler

    Returns:
        (train_loader, val_loader)
    """
    coco_root = os.path.join(data_root, 'coco')

    train_ds = COCODetectionDataset(coco_root, 'train', img_size, augment=True)
    val_ds = COCODetectionDataset(coco_root, 'val', img_size, augment=False)

    train_sampler = None
    val_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=coco_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=coco_collate_fn,
    )
    return train_loader, val_loader
