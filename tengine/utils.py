"""
Training utilities: metrics tracking, checkpointing, seed, model builder.
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np


# ---- Metrics ----

class AverageMeter:
    """Tracks mean and count of a metric."""

    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ---- Checkpointing ----

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(output_dir, 'best.pth')
        torch.save(state, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return start_epoch, best_acc


# ---- Seed ----

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- Model builder ----

_MODEL_REGISTRY = {}


def _populate_registry():
    """Lazy-populate from models package."""
    if _MODEL_REGISTRY:
        return
    import models as M

    # ResNet variants
    for name in ['sew_resnet18', 'sew_resnet34', 'sew_resnet50',
                 'sew_resnet101', 'sew_resnet152',
                 'ms_resnet18', 'ms_resnet34', 'ms_resnet104']:
        fn = getattr(M, name, None)
        if fn:
            _MODEL_REGISTRY[name] = fn

    # Transformer variants
    for name in ['spikformer_8_384', 'spikformer_8_512', 'spikformer_8_768',
                 'sdt_v1_8_384', 'sdt_v1_8_512', 'sdt_v1_8_768',
                 'meta_spikformer_8_384', 'meta_spikformer_8_512', 'meta_spikformer_8_768',
                 'qkformer_10_384', 'qkformer_10_512', 'qkformer_10_768',
                 'maxformer_10_384', 'maxformer_10_512', 'maxformer_10_768']:
        fn = getattr(M, name, None)
        if fn:
            _MODEL_REGISTRY[name] = fn


def list_models():
    _populate_registry()
    return sorted(_MODEL_REGISTRY.keys())


def build_model(model_name, **kwargs):
    """Build a model by name. kwargs are forwarded to the factory function."""
    _populate_registry()
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list_models()}"
        )
    return _MODEL_REGISTRY[model_name](**kwargs)


# ---- Dataset builder ----

_DATASET_CONFIG = {
    'cifar10':    {'num_classes': 10,  'img_size': 32},
    'cifar100':   {'num_classes': 100, 'img_size': 32},
    'imagenet':   {'num_classes': 1000, 'img_size': 224},
    'cifar10dvs': {'num_classes': 10,  'img_size': 128},
}


def get_dataset_config(dataset_name):
    if dataset_name not in _DATASET_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {list(_DATASET_CONFIG.keys())}")
    return _DATASET_CONFIG[dataset_name]


def build_dataloaders(dataset_name, data_root, batch_size, img_size=None,
                      num_workers=4, distributed=False, **kwargs):
    """Build train/test dataloaders by dataset name."""
    from datasets import (cifar10_dataloaders, cifar100_dataloaders,
                          imagenet_dataloaders, cifar10dvs_dataloaders)

    cfg = get_dataset_config(dataset_name)
    if img_size is None:
        img_size = cfg['img_size']

    if dataset_name == 'cifar10':
        return cifar10_dataloaders(
            data_root, batch_size, img_size=img_size,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'cifar100':
        return cifar100_dataloaders(
            data_root, batch_size, img_size=img_size,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'imagenet':
        return imagenet_dataloaders(
            data_root, batch_size, img_size=img_size,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'cifar10dvs':
        frames = kwargs.pop('frames_number', 16)
        return cifar10dvs_dataloaders(
            data_root, batch_size, frames_number=frames,
            num_workers=num_workers, distributed=distributed, **kwargs)
