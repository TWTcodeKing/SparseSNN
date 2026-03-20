"""
SparseSNN Training Script.

Supports single-GPU and multi-GPU (DDP) training for all SNN models.

Single GPU (ResNet, legacy):
    python tengine/train.py --model sew_resnet18 --dataset cifar10 \
        --data-root ./data --gpu-ids 0

Single GPU (Transformer, config-based):
    python tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar100 --data-root ./data --gpu-ids 0

Multi-GPU (Transformer, config-based):
    torchrun --nproc_per_node=4 tengine/train.py \
        --config configs/spikformer/spikformer_8_384.yaml \
        --dataset imagenet --data-root /data/imagenet --gpu-ids 0,1,2,3
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path so that `models` and `datasets` are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.dist import setup_distributed, cleanup_distributed, is_main_process, reduce_tensor
from tengine.logger import setup_logger, TrainLogger
from tengine.utils import (
    AverageMeter, accuracy, set_seed,
    save_checkpoint, load_checkpoint,
    build_model, build_model_from_config, load_model_config,
    load_training_recipe,
    build_dataloaders, get_dataset_config, list_models,
)
from datasets.augmentation import mixup_data, cutmix_data, mixup_criterion


def parse_args():
    parser = argparse.ArgumentParser(description='SparseSNN Training')

    # ---- Model (two modes: --model for ResNets, --config for Transformers) ----
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None,
                       help='ResNet model name (e.g. sew_resnet18)')
    group.add_argument('--config', type=str, default=None,
                       help='Path to model YAML config (for transformer models)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps for SNN (default: 4)')
    parser.add_argument('--zero-init-residual', action='store_true', default=False,
                        help='Zero-initialize last BN in each residual branch (ResNets only)')
    parser.add_argument('--connect-f', type=str, default="ADD",
                        help='Add extra connection from block input to block output (ResNets only)')
    # ---- Dataset ----
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'],
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Override input image size (default: dataset native)')

    # ---- Training ----
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # ---- Optimizer ----
    parser.add_argument('--opt', type=str, default='adamw',
                        choices=['adamw', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9)

    # ---- Scheduler ----
    parser.add_argument('--sched', type=str, default='cosine',
                        choices=['cosine', 'step', 'multistep'],
                        help='LR scheduler type')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Epoch interval for StepLR decay')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='LR decay factor for step/multistep')
    parser.add_argument('--decay-epochs', type=str, default='',
                        help='Comma-separated milestone epochs for multistep (e.g. 100,150)')

    # ---- Augmentation ----
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='MixUp alpha (0 = disabled)')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0,
                        help='CutMix alpha (0 = disabled)')
    parser.add_argument('--auto-aug', action='store_true', default=False,
                        help='Use AutoAugment')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='Use Cutout (RandomErasing)')
    parser.add_argument('--random-erasing', type=float, default=0.0,
                        help='Random erasing probability')

    # ---- AMP ----
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')

    # ---- Recipe ----
    parser.add_argument('--recipe', type=str, default='',
                        help='Path to training recipe YAML (values serve as defaults)')

    # ---- Infrastructure ----
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='Comma-separated GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    # ---- Structured Sparse Training (SR-STE 2:4) ----
    parser.add_argument('--structured-sparse', action='store_true', default=False,
                        help='Enable SR-STE 2:4 structured sparsity regularization')
    parser.add_argument('--sr-lambda', type=float, default=0.01,
                        help='SR-STE regularization strength (default: 0.01)')
    parser.add_argument('--sr-start-epoch', type=int, default=50,
                        help='Epoch at which SR-STE regularization begins (default: 50)')
    parser.add_argument('--sr-end-epoch', type=int, default=150,
                        help='Epoch at which SR-STE lambda reaches target (default: 150)')
    # ---- Learnable neuron parameters ----
    parser.add_argument('--learnable-params', action='store_true', default=False,
                        help='Make tau and v_threshold of LIF neurons learnable')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args):
    """Build LR scheduler with optional linear warmup.

    Supported scheduler types (args.sched):
        - 'cosine': CosineAnnealingLR (default for all SNN models)
        - 'step':   StepLR with fixed step_size and gamma
        - 'multistep': MultiStepLR with milestone epochs and gamma

    When warmup_epochs > 0, a LinearLR warmup phase is prepended via
    SequentialLR so the two phases compose cleanly.
    """
    import math
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    sched_type = getattr(args, 'sched', 'cosine')

    # ---- Main scheduler (operates over post-warmup epochs) ----
    after_epochs = max(1, total_epochs - warmup_epochs)

    if sched_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=after_epochs, eta_min=args.min_lr)
    elif sched_type == 'step':
        step_size = getattr(args, 'step_size', 30)
        decay_rate = getattr(args, 'decay_rate', 0.1)
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=decay_rate)
    elif sched_type == 'multistep':
        decay_epochs = getattr(args, 'decay_epochs', [])
        if isinstance(decay_epochs, str):
            decay_epochs = [int(e) for e in decay_epochs.split(',') if e.strip()]
        # Shift milestones by warmup offset since SequentialLR resets epoch count
        milestones = [max(0, m - warmup_epochs) for m in decay_epochs]
        decay_rate = getattr(args, 'decay_rate', 0.1)
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=decay_rate)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    # ---- Warmup phase ----
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs])
    else:
        scheduler = main_scheduler

    return scheduler


# ---------------------------------------------------------------------------
# Train / Evaluate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, tlog, world_size, mixup_alpha=0.0, cutmix_alpha=0.0,
                    structured_sparse=False, sr_lambda=0.0):
    """Train one epoch.

    Args:
        structured_sparse: If True, adds SR-STE 2:4 regularization to task loss.
        sr_lambda: Current regularization coefficient (computed by caller from
            ProgressiveSparsityScheduler). Only used when structured_sparse=True.
    """
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    total_steps = len(loader)
    use_mix = mixup_alpha > 0 or cutmix_alpha > 0

    # Import SR-STE helper only when needed (avoids import cost for normal runs)
    if structured_sparse and sr_lambda > 0:
        from sparse.st_train import sr_ste_regularizer
    else:
        sr_ste_regularizer = None

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp / CutMix
        mix_active = False
        if use_mix:
            import random as _rnd
            if mixup_alpha > 0 and cutmix_alpha > 0:
                if _rnd.random() < 0.5:
                    images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                else:
                    images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
                mix_active = True
            elif mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                mix_active = True
            elif cutmix_alpha > 0:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
                mix_active = True

        if scaler is not None:
            with amp.autocast():
                output = model(images)
                if mix_active:
                    task_loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    task_loss = criterion(output, targets)
        else:
            output = model(images)
            if mix_active:
                task_loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                task_loss = criterion(output, targets)

        # SR-STE regularization: adds ||W - project_2_4(W)||^2 penalty
        if sr_ste_regularizer is not None:
            sr_loss = sr_ste_regularizer(model, sr_lambda)
            loss = task_loss + sr_loss
        else:
            loss = task_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)

        if world_size > 1:
            loss = reduce_tensor(loss.detach(), world_size)
            acc1 = reduce_tensor(acc1, world_size)
            acc5 = reduce_tensor(acc5, world_size)

        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)

        if is_main_process():
            tlog.step(epoch, step, total_steps,
                      loss=losses.val, acc1=top1.val, lr=optimizer.param_groups[0]['lr'])

    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, world_size):
    model.eval()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)

        if world_size > 1:
            loss = reduce_tensor(loss.detach(), world_size)
            acc1 = reduce_tensor(acc1, world_size)
            acc5 = reduce_tensor(acc5, world_size)

        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)

    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Apply recipe defaults (CLI args override recipe values) ----
    if args.recipe:
        recipe = load_training_recipe(args.recipe)
        parser_defaults = {
            'opt': 'adamw', 'lr': 1e-3, 'weight_decay': 0, 'momentum': 0.9,
            'sched': 'cosine', 'warmup_epochs': 0, 'min_lr': 1e-5,
            'step_size': 30, 'decay_rate': 0.1, 'decay_epochs': '',
            'epochs': 200, 'batch_size': 64,
            'label_smoothing': 0.1, 'T': 4, 'mixup_alpha': 0.0, 'cutmix_alpha': 0.0,
            'auto_aug': False, 'cutout': False, 'random_erasing': 0.0,
            # structured sparse defaults
            'structured_sparse': False, 'sr_lambda': 0.01,
            'sr_start_epoch': 50, 'sr_end_epoch': 150,
            'learnable_params': False,
        }
        for key, recipe_val in recipe.items():
            if hasattr(args, key):
                # Only apply recipe value if user didn't explicitly set this arg
                current = getattr(args, key)
                default = parser_defaults.get(key)
                if current == default:
                    setattr(args, key, recipe_val)

    # ---- Distributed setup ----
    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1

    # GPU selection (for single-GPU mode, use the first ID)
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    if not distributed:
        torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')

    set_seed(args.seed + rank)

    # ---- Dataset ----
    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=args.workers,
        distributed=distributed,
        auto_aug=args.auto_aug, cutout=args.cutout,
    )

    # ---- Model ----
    learnable_params = getattr(args, 'learnable_params', False)
    if args.config:
        # Transformer models: load YAML config and merge runtime params
        model_cfg = load_model_config(args.config)
        model_cfg.update({
            'num_classes': num_classes,
            'T': args.T,
            'img_size': img_size,
            'in_channels': in_channels,
        })
        if learnable_params:
            model_cfg['learnable_params'] = True
        model = build_model_from_config(model_cfg)
        model_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        # ResNet models: direct factory
        model_kwargs = {'in_channels': in_channels}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
            model_kwargs['zero_init_residual'] = args.zero_init_residual
            model_kwargs['connect_f'] = args.connect_f
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, num_classes=num_classes,**model_kwargs)
        model_name = args.model

    model = model.to(device)

    # ---- Output directory ----
    run_name = f"{model_name}_{args.dataset}_bs{args.batch_size}_lr{args.lr}"
    output_dir = os.path.join(args.output_dir, run_name)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    # ---- Logger (main process only) ----
    log_file = os.path.join(output_dir, 'train.log') if is_main_process() else None
    logger = setup_logger('train', log_file)
    tlog = TrainLogger(logger, args.epochs, args.print_freq)

    if is_main_process():
        tlog.banner(f"Config: {vars(args)}")

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {model_name}  |  Params: {n_params:,}")

    # ---- Optimizer ----
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = build_scheduler(optimizer, args)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = amp.GradScaler() if args.amp else None

    # ---- Resume ----
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model.module if distributed else model,
            optimizer, scheduler)
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}")

    # ---- Structured sparse scheduler (SR-STE) ----
    structured_sparse = getattr(args, 'structured_sparse', False)
    sr_scheduler = None
    if structured_sparse:
        from sparse.st_train import ProgressiveSparsityScheduler
        sr_scheduler = ProgressiveSparsityScheduler(
            start_epoch=getattr(args, 'sr_start_epoch', 50),
            end_epoch=getattr(args, 'sr_end_epoch', 150),
            target_lambda=getattr(args, 'sr_lambda', 0.01),
        )
        if is_main_process():
            logger.info(f"SR-STE enabled: {sr_scheduler}")

    # ---- Training loop ----
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        if is_main_process():
            tlog.epoch_start(epoch)

        # Compute current SR-STE lambda (0.0 if not using structured sparse)
        current_sr_lambda = 0.0
        if sr_scheduler is not None:
            current_sr_lambda = sr_scheduler.get_lambda(epoch)

        train_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                  scaler, device, epoch, tlog, world_size,
                                  mixup_alpha=args.mixup_alpha,
                                  cutmix_alpha=args.cutmix_alpha,
                                  structured_sparse=structured_sparse,
                                  sr_lambda=current_sr_lambda)
        val_m = evaluate(model, val_loader, criterion, device, world_size)

        scheduler.step()

        is_best = val_m['acc1'] > best_acc
        if is_best:
            best_acc = val_m['acc1']

        if is_main_process():
            tlog.epoch_end(epoch, train_m, val_m)
            if is_best:
                tlog.best(epoch, best_acc)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': (model.module if distributed else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
                'args': vars(args),
            }, is_best, output_dir)

    # ---- Apply hard 2:4 projection after training ----
    if structured_sparse and is_main_process():
        from sparse.st_train import apply_hard_n_m_projection
        raw_model = model.module if distributed else model
        proj_stats = apply_hard_n_m_projection(raw_model)
        n_projected = sum(1 for v in proj_stats.values())
        logger.info(f"Hard N:M projection applied to {n_projected} Linear layers")
        # Save the sparse model
        sparse_path = os.path.join(output_dir, 'best_sparse.pth')
        torch.save({
            'model': raw_model.state_dict(),
            'args': vars(args),
            'projection_stats': proj_stats,
        }, sparse_path)
        logger.info(f"Sparse model saved to {sparse_path}")

    total_time = time.time() - t0
    if is_main_process():
        tlog.finish(best_acc, total_time)

    cleanup_distributed()


if __name__ == '__main__':
    main()
