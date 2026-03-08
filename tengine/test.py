"""
SparseSNN Evaluation Script.

Evaluate a trained model checkpoint on a dataset.

Usage (ResNet):
    python tengine/test.py --model sew_resnet18 --dataset cifar10 \
        --data-root ./data --checkpoint ./output/.../best.pth --gpu-ids 0

Usage (Transformer, config-based):
    python tengine/test.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar10 --data-root ./data --checkpoint ./output/.../best.pth --gpu-ids 0
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.logger import setup_logger
from tengine.utils import (
    AverageMeter, accuracy, set_seed,
    build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description='SparseSNN Evaluation')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None,
                       help='ResNet model name (e.g. sew_resnet18)')
    group.add_argument('--config', type=str, default=None,
                       help='Path to model YAML config (for transformer models)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps for SNN')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, logger):
    model.eval()
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    total_steps = len(loader)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)

        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)

        if (step + 1) % 20 == 0 or step == total_steps - 1:
            logger.info(
                f"  Step [{step + 1}/{total_steps}]  "
                f"Loss: {losses.avg:.4f}  Acc@1: {top1.avg:.4f}  Acc@5: {top5.avg:.4f}"
            )

    return {'loss': losses.avg, 'acc1': top1.avg, 'acc5': top5.avg}


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')

    set_seed(args.seed)

    logger = setup_logger('test')
    dash = "-" * 72

    # ---- Dataset ----
    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=args.workers,
    )

    # ---- Model ----
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
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, **model_kwargs)
        model_name = args.model

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(dash)
    logger.info(f"Model: {model_name}  |  Params: {n_params:,}")
    logger.info(f"Dataset: {args.dataset}  |  Checkpoint: {args.checkpoint}")
    logger.info(dash)

    criterion = nn.CrossEntropyLoss()
    results = evaluate(model, val_loader, criterion, device, logger)

    logger.info(dash)
    logger.info(
        f"Results >>  Loss: {results['loss']:.4f}  "
        f"Acc@1: {results['acc1']:.4f}  Acc@5: {results['acc5']:.4f}"
    )
    logger.info(dash)


if __name__ == '__main__':
    main()
