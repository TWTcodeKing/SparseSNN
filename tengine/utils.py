"""
Training utilities: metrics tracking, checkpointing, seed, model/config builder.
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import yaml


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


# ---- YAML config loading ----

def load_training_recipe(recipe_path):
    """Load a training recipe YAML file.

    Returns a flat dict suitable for merging with argparse defaults.
    Nested keys like optimizer.lr become top-level: {'opt': ..., 'lr': ..., ...}
    """
    with open(recipe_path, 'r') as f:
        recipe = yaml.safe_load(f)

    flat = {}
    # Optimizer
    opt = recipe.get('optimizer', {})
    if 'type' in opt:
        flat['opt'] = opt['type']
    if 'lr' in opt:
        flat['lr'] = float(opt['lr'])
    if 'weight_decay' in opt:
        flat['weight_decay'] = float(opt['weight_decay'])
    if 'momentum' in opt:
        flat['momentum'] = float(opt['momentum'])

    # Scheduler
    sched = recipe.get('scheduler', {})
    if 'type' in sched:
        flat['sched'] = sched['type']
    if 'warmup_epochs' in sched:
        flat['warmup_epochs'] = int(sched['warmup_epochs'])
    if 'min_lr' in sched:
        flat['min_lr'] = float(sched['min_lr'])
    if 'step_size' in sched:
        flat['step_size'] = int(sched['step_size'])
    if 'decay_rate' in sched:
        flat['decay_rate'] = float(sched['decay_rate'])
    if 'decay_epochs' in sched:
        de = sched['decay_epochs']
        if isinstance(de, list):
            flat['decay_epochs'] = ','.join(str(e) for e in de)
        else:
            flat['decay_epochs'] = str(de)

    # Top-level
    for key in ('epochs', 'batch_size', 'img_size'):
        if key in recipe:
            flat[key.replace('-', '_')] = recipe[key]

    # Augmentation
    aug = recipe.get('augmentation', {})
    for key in ('auto_aug', 'cutout', 'mixup_alpha', 'cutmix_alpha', 'random_erasing',
                'snn_aug', 'mixup_off_epoch'):
        if key in aug:
            flat[key] = aug[key]

    # Regularization
    reg = recipe.get('regularization', {})
    if 'label_smoothing' in reg:
        flat['label_smoothing'] = reg['label_smoothing']
    if 'drop_path_rate' in reg:
        flat['drop_path_rate'] = reg['drop_path_rate']

    # SNN
    snn = recipe.get('snn', {})
    if 'T' in snn:
        flat['T'] = snn['T']
    if 'learnable_params' in snn:
        flat['learnable_params'] = bool(snn['learnable_params'])
    if 'amp' in snn:
        flat['amp'] = bool(snn['amp'])

    # Structured sparse (SR-STE 2:4)
    ss = recipe.get('structured_sparsity', recipe.get('structured_sparse', {}))
    if ss.get('enabled', False):
        flat['structured_sparse'] = True
    if 'sr_lambda' in ss:
        flat['sr_lambda'] = float(ss['sr_lambda'])
    if 'sr_n' in ss:
        flat['sr_n'] = int(ss['sr_n'])
    if 'sr_m' in ss:
        flat['sr_m'] = int(ss['sr_m'])
    if 'start_epoch' in ss:
        flat['sr_start_epoch'] = int(ss['start_epoch'])
    if 'end_epoch' in ss:
        flat['sr_end_epoch'] = int(ss['end_epoch'])

    # Dynamic N:M-ceiling sparse training
    dyn = recipe.get('dynamic_sparsity', {})
    if dyn.get('enabled', False):
        flat['dynamic_sparse'] = True
    for key in ('dyn_n', 'dyn_m', 'dyn_target_sparsity', 'dyn_grow_ratio',
                'dyn_alpha', 'dyn_delta_T', 'dyn_T_end_fraction',
                'dyn_density_exponent', 'dyn_ema_decay'):
        if key in dyn:
            flat[key] = dyn[key]

    return flat


def load_model_config(config_path):
    """Load a model config from a YAML file.

    Returns a dict with at least an 'arch' key identifying the model family.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'arch' not in config:
        raise ValueError(f"Config {config_path} must contain an 'arch' field")
    return config


# ---- Model builder ----

_RESNET_REGISTRY = {}


def _populate_resnet_registry():
    """Lazy-populate ResNet factory functions."""
    if _RESNET_REGISTRY:
        return
    import models as M
    for name in ['sew_resnet18', 'sew_resnet34', 'sew_resnet50',
                 'sew_resnet101', 'sew_resnet152',
                 'sew_resnet_cifar20', 'sew_resnet_cifar32', 'sew_resnet_cifar44',
                 'sew_resnet_cifar56', 'sew_resnet_cifar110',
                 'ms_resnet18', 'ms_resnet34', 'ms_resnet50', 'ms_resnet104',
                 'ms_resnet_cifar20', 'ms_resnet_cifar32', 'ms_resnet_cifar44',
                 'ms_resnet_cifar56', 'ms_resnet_cifar110',
                 'ms_resnet_dvs20',
                 'dvs_sew_resnet',
                 'snn_vgg9', 'snn_vgg11', 'snn_vgg16', 'snn_vgg19',
                 'ems_yolo_res34',
                 'spike_yolo_n', 'spike_yolo_s', 'spike_yolo_m']:
        fn = getattr(M, name, None)
        if fn:
            _RESNET_REGISTRY[name] = fn


def list_models():
    _populate_resnet_registry()
    resnet_names = sorted(_RESNET_REGISTRY.keys())
    transformer_note = ['(transformer models: use --config <yaml>)']
    return resnet_names + transformer_note


def build_model_from_config(config):
    """Build a transformer model from a merged config dict.

    The config must contain 'arch' (matching a key in models.ARCH_BUILDERS)
    plus all required model parameters (including runtime keys like
    num_classes, T, img_size, in_channels).
    """
    from models import ARCH_BUILDERS
    arch = config['arch']
    if arch not in ARCH_BUILDERS:
        raise ValueError(
            f"Unknown arch '{arch}'. Available: {list(ARCH_BUILDERS.keys())}"
        )
    return ARCH_BUILDERS[arch](config)


def build_model(model_name, **kwargs):
    """Build a ResNet model by name. kwargs are forwarded to the factory function."""
    _populate_resnet_registry()
    if model_name not in _RESNET_REGISTRY:
        raise ValueError(
            f"Unknown ResNet model '{model_name}'. "
            f"Available: {sorted(_RESNET_REGISTRY.keys())}. "
            f"For transformer models, use --config <yaml>."
        )
    return _RESNET_REGISTRY[model_name](**kwargs)


# ---- Dataset builder ----

_DATASET_CONFIG = {
    'cifar10':    {'num_classes': 10,   'img_size': 32,  'in_channels': 3},
    'cifar100':   {'num_classes': 100,  'img_size': 32,  'in_channels': 3},
    'imagenet':   {'num_classes': 1000, 'img_size': 224, 'in_channels': 3},
    'cifar10dvs': {'num_classes': 10,   'img_size': 128, 'in_channels': 2},
    'dvs128gesture': {'num_classes': 11, 'img_size': 128, 'in_channels': 2},
    'coco':       {'num_classes': 80,   'img_size': 640, 'in_channels': 3, 'task': 'detection'},
    'gen1':       {'num_classes': 2,    'img_size': 320, 'in_channels': 3, 'task': 'detection'},
    'sst2':       {'num_classes': 2,    'img_size': 1,   'in_channels': 1, 'task': 'nlp', 'seq_len': 128},
    'mrpc':       {'num_classes': 2,    'img_size': 1,   'in_channels': 1, 'task': 'nlp', 'seq_len': 128},
    'ntufi_humanid': {'num_classes': 14, 'img_size': 32, 'in_channels': 3},
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
                          imagenet_dataloaders, cifar10dvs_dataloaders,
                          dvs128gesture_dataloaders)

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
    elif dataset_name == 'dvs128gesture':
        frames = kwargs.pop('frames_number', 16)
        return dvs128gesture_dataloaders(
            data_root, batch_size, frames_number=frames,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'coco':
        from datasets import coco_dataloaders
        return coco_dataloaders(
            data_root, batch_size, img_size=img_size,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'gen1':
        from datasets import gen1_dataloaders
        return gen1_dataloaders(
            data_root, batch_size, img_size=img_size,
            num_workers=num_workers, distributed=distributed, **kwargs)
    elif dataset_name == 'ntufi_humanid':
        from datasets import ntufi_humanid_dataloaders
        T = kwargs.pop('T', kwargs.pop('frames_number', 4))
        return ntufi_humanid_dataloaders(
            data_root, batch_size, T=T, spatial_size=(img_size, img_size),
            num_workers=num_workers, distributed=distributed)
    elif dataset_name in ('sst2', 'mrpc', 'cola', 'qnli'):
        from datasets import glue_dataloaders
        seq_len = kwargs.pop('seq_len', cfg.get('seq_len', 128))
        return glue_dataloaders(
            data_root, batch_size, task=dataset_name, seq_len=seq_len,
            num_workers=num_workers, distributed=distributed, **kwargs)
