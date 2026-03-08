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
    build_dataloaders, get_dataset_config, list_models,
)


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

    # ---- AMP ----
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision')

    # ---- Infrastructure ----
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='Comma-separated GPU IDs, e.g. 0,1,2,3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cosine LR scheduler with linear warmup
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / Evaluate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    epoch, tlog, world_size):
    model.train()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    total_steps = len(loader)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if scaler is not None:
            with amp.autocast():
                output = model(images)
                loss = criterion(output, targets)
        else:
            output = model(images)
            loss = criterion(output, targets)

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
    )

    # ---- Model ----
    if args.config:
        # Transformer models: load YAML config and merge runtime params
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
        # ResNet models: direct factory
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
            model_kwargs['zero_init_residual'] = args.zero_init_residual
            model_kwargs['connect_f'] = args.connect_f
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, **model_kwargs)
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

    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs,
                                args.min_lr, args.lr)

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

    # ---- Training loop ----
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        if is_main_process():
            tlog.epoch_start(epoch)

        train_m = train_one_epoch(model, train_loader, criterion, optimizer,
                                  scaler, device, epoch, tlog, world_size)
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

    total_time = time.time() - t0
    if is_main_process():
        tlog.finish(best_acc, total_time)

    cleanup_distributed()


if __name__ == '__main__':
    main()
