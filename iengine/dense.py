"""Dense baseline inference benchmark.

Evaluates a dense SNN model in fp16 with optional torch.compile and
fused Triton LIF/IF kernels. Reports accuracy and latency (ms/batch).

Usage:
    python -m iengine.dense \
        --model sew_resnet_cifar32 --dataset cifar100 --data-root /home/twt/datasets \
        --checkpoint output/sew_resnet_cifar32_cifar100_bs100_lr0.1/best.pth \
        --T 4 --batch-size 64 --compile --fuse-neurons
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
def maybe_fuse_neurons(model, fuse=True, verbose=True):
    """Optionally replace neuron forwards with fused Triton kernels."""
    if not fuse:
        return 0
    try:
        from iengine.triton_sparse.neuron_kernel import replace_neuron_forward
    except ImportError as e:
        if verbose:
            print(f"  [Triton neurons] Not available: {e}")
        return 0
    return replace_neuron_forward(model, verbose=verbose)
from tengine.utils import (
    set_seed,
    build_model_from_config,
    load_model_config,
    build_dataloaders,
    get_dataset_config,
    AverageMeter,
    accuracy,
)


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None, input_dtype=None):
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    n_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        n_seen += images.size(0)
        if max_samples and n_seen >= max_samples:
            break

    return {'acc1': top1.avg, 'acc5': top5.avg}


@torch.no_grad()
def benchmark_latency(model, loader, device, n_warmup=10, n_measure=50,
                      input_dtype=None):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    if input_dtype is not None:
        images = images.to(dtype=input_dtype)

    for _ in range(n_warmup):
        model(images)
        reset_net(model)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_measure):
        model(images)
        reset_net(model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / n_measure) * 1000  # ms/batch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dense baseline inference benchmark (fp16)')
    parser.add_argument('--config', type=str, default=None,
                        help='Model config YAML path')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet_cifar32)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs',
                                 'dvs128gesture'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--img-size', type=int, default=None,
                        help='Override image size (e.g. 128 for transfer models)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile')
    parser.add_argument('--fuse-neurons', action='store_true', default=False,
                        help='Replace LIF/IF neurons with fused Triton kernels')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    ds_config = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_config['img_size']

    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=4, distributed=False,
    )

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_config)
        config['img_size'] = img_size
        if args.T is not None:
            config['T'] = args.T
        elif 'T' not in config:
            config['T'] = 4
        model = build_model_from_config(config)
    elif args.model:
        from tengine.utils import build_model
        model = build_model(args.model, num_classes=ds_config['num_classes'],
                            in_channels=ds_config['in_channels'], T=args.T or 4)
    else:
        raise ValueError("Must provide --config or --model")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device).half().eval()

    n_params = sum(p.numel() for p in model.parameters())

    compile_tag = " + torch.compile" if args.compile else ""
    neuron_tag = " + fused neurons" if args.fuse_neurons else ""

    maybe_fuse_neurons(model, fuse=args.fuse_neurons)
    if args.compile:
        model = torch.compile(model)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Model: {args.config or args.model}  |  Params: {n_params:,}")
    print(f"Mode: Dense fp16{compile_tag}{neuron_tag}")

    acc = evaluate(model, test_loader, device, args.max_samples,
                   input_dtype=torch.float16)
    latency = benchmark_latency(model, test_loader, device,
                                input_dtype=torch.float16)

    print(f"\n{'='*50}")
    print(f"  Acc@1: {acc['acc1']:.2f}%  Acc@5: {acc['acc5']:.2f}%")
    print(f"  Latency: {latency:.3f} ms/batch  (bs={args.batch_size})")
    print(f"{'='*50}")

    return {'acc': acc, 'latency_ms_per_batch': latency}


if __name__ == '__main__':
    main()
