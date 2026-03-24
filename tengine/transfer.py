"""
SparseSNN Transfer Learning Script.

Finetune a pretrained ImageNet model on a downstream dataset (CIFAR-10/100).
Replaces the classifier head with a randomly initialized one, loads pretrained
weights for the backbone, and finetunes the full model.

Usage:
    uv run tengine/transfer.py \
        --config configs/spikingresformer/spikingresformer_ti.yaml \
        --pretrained checkpoints/spikingresformer/ImageNet_spikingresformer_ti.pth \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --img-size 128 --epochs 100 --lr 1e-4 --gpu-ids 0
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.cuda import amp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.logger import setup_logger, TrainLogger
from tengine.utils import (
    AverageMeter, accuracy, set_seed,
    save_checkpoint,
    build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)
from datasets.augmentation import mixup_data, cutmix_data, mixup_criterion


def parse_args():
    parser = argparse.ArgumentParser(description='SparseSNN Transfer Learning')

    # Model
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None,
                       help='ResNet model name')
    group.add_argument('--config', type=str, default=None,
                       help='Path to model YAML config')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained ImageNet checkpoint')
    parser.add_argument('--recipe', type=str, default=None,
                        help='Path to training recipe YAML (overrides defaults)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'cifar10dvs', 'dvs128gesture'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=None,
                        help='Input image size (default: from dataset config)')
    parser.add_argument('--frames-number', type=int, default=None,
                        help='Number of frames for DVS datasets (default: same as --T)')

    # Training (defaults; overridden by recipe if provided)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.5)
    parser.add_argument('--cutmix-alpha', type=float, default=0.0)
    parser.add_argument('--auto-aug', action='store_true', default=True)
    parser.add_argument('--amp', action='store_true', default=True)

    # Infra
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--output-dir', type=str, default='output')

    # Extra model kwargs
    parser.add_argument('--connect-f', type=str, default='ADD')
    parser.add_argument('--neuron-type', type=str, default=None)

    args = parser.parse_args()

    # ---- Apply recipe overrides (recipe > argparse defaults, CLI > recipe) ----
    if args.recipe:
        from tengine.utils import load_training_recipe
        recipe_defaults = load_training_recipe(args.recipe)
        # Recipe sets defaults; explicit CLI args still take priority
        for k, v in recipe_defaults.items():
            if hasattr(args, k) and k not in _cli_explicit_args():
                setattr(args, k, v)

    return args


def _cli_explicit_args():
    """Return set of arg names explicitly passed on the command line."""
    import sys as _sys
    explicit = set()
    for tok in _sys.argv[1:]:
        if tok.startswith('--'):
            name = tok.lstrip('-').split('=')[0].replace('-', '_')
            explicit.add(name)
    return explicit


@torch.no_grad()
def evaluate(model, loader, criterion, device, dvs=False):
    model.eval()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # DVS data: (B, T, C, H, W) — models handle permutation internally
        output = model(images)
        loss = criterion(output, targets)
        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, logger, mixup_alpha=0.0, cutmix_alpha=0.0, dvs=False):
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # DVS data: (B, T, C, H, W) — models handle permutation internally

        # Mixup / CutMix
        mixed = False
        if mixup_alpha > 0 and torch.rand(1).item() < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
            mixed = True
        elif cutmix_alpha > 0:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
            mixed = True

        optimizer.zero_grad()
        if scaler is not None:
            with amp.autocast():
                output = model(images)
                if mixed:
                    loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    loss = criterion(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images)
            if mixed:
                loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            else:
                loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        reset_net(model)
        losses.update(loss.item(), images.size(0))
        if not mixed:
            acc1, _ = accuracy(output, targets, topk=(1, 5))
            top1.update(acc1.item(), images.size(0))

        if (step + 1) % 50 == 0:
            logger.info(
                f"  Epoch [{epoch}] Step [{step+1}/{len(loader)}] "
                f"Loss: {losses.avg:.4f}  Acc@1: {top1.avg:.2f}"
            )

    return losses.avg


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(args.seed)

    logger = setup_logger('transfer')
    dash = "-" * 72

    # ---- Dataset ----
    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']
    dvs = args.dataset in ('cifar10dvs', 'dvs128gesture')

    dl_kwargs = {}
    if dvs:
        dl_kwargs['frames_number'] = args.frames_number or args.T
    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=args.workers,
        auto_aug=getattr(args, 'auto_aug', False) if not dvs else False,
        **dl_kwargs,
    )

    # ---- Build model with target num_classes ----
    if args.config:
        model_cfg = load_model_config(args.config)
        model_cfg.update({
            'num_classes': num_classes,
            'T': args.T,
            'img_size': img_size,
            'in_channels': in_channels,
        })
        model = build_model_from_config(model_cfg)
        model_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        extra = {}
        if args.connect_f:
            extra['connect_f'] = args.connect_f
        if args.neuron_type:
            extra['neuron_type'] = args.neuron_type
        model = build_model(args.model, num_classes=num_classes,
                            in_channels=in_channels, T=args.T, **extra)
        model_name = args.model

    # ---- Load pretrained weights (skip classifier) ----
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    pretrained_sd = ckpt
    for key in ('model', 'state_dict', 'net'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            pretrained_sd = ckpt[key]
            break

    # Filter out classifier/head keys (shape mismatch due to num_classes change)
    model_sd = model.state_dict()
    loaded, skipped = {}, []
    for k, v in pretrained_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            loaded[k] = v
        else:
            skipped.append(k)

    result = model.load_state_dict(loaded, strict=False)
    logger.info(f"Loaded {len(loaded)} pretrained params, skipped {len(skipped)}: {skipped}")
    logger.info(f"Missing (randomly initialized): {result.missing_keys}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(dash)
    logger.info(f"Transfer: {model_name} → {args.dataset}")
    logger.info(f"Pretrained: {args.pretrained}")
    logger.info(f"Params: {n_params:,} (trainable: {n_trainable:,})")
    logger.info(f"img_size: {img_size}, T: {args.T}, epochs: {args.epochs}, lr: {args.lr}")
    logger.info(dash)

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    warmup_epochs = args.warmup_epochs
    after_epochs = max(1, args.epochs - warmup_epochs)
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=after_epochs, eta_min=args.min_lr)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched],
            milestones=[warmup_epochs])
    else:
        scheduler = main_sched

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = amp.GradScaler() if args.amp else None

    # ---- Output ----
    run_name = f"transfer_{model_name}_{args.dataset}_lr{args.lr}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, logger, args.mixup_alpha, args.cutmix_alpha, dvs=dvs)

        scheduler.step()
        results = evaluate(model, val_loader, criterion, device, dvs=dvs)

        is_best = results['acc1'] > best_acc
        if is_best:
            best_acc = results['acc1']

        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {train_loss:.4f}  Val Acc@1: {results['acc1']:.2f}  "
            f"Best: {best_acc:.2f}  LR: {lr:.6f}  "
            f"Time: {time.time()-t0:.1f}s"
        )

        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
        }, is_best, output_dir)

    logger.info(dash)
    logger.info(f"Transfer complete. Best Acc@1: {best_acc:.2f}")
    logger.info(dash)


if __name__ == '__main__':
    main()
