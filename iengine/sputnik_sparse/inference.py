"""Sputnik sparse inference benchmark for SNN models.

Uses Google Research's Sputnik CUDA SpMM kernels to accelerate sparse
SNN activations through Linear layers and SSA attention.

Requires torch_sputnik — build from source:
    bash iengine/sputnik_sparse/build.sh

Usage:
    python -m iengine.sputnik_sparse.inference \
        --config configs/spikformer/spikformer_cifar.yaml \
        --checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets
"""

import argparse
import copy
import sys
import os
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config,
    load_model_config, get_dataset_config, build_dataloaders,
    AverageMeter, accuracy,
)
from .accelerator import SputnikAccelerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model, loader, device, max_samples=None):
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    n_seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
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
def _benchmark_latency(model, loader, device, n_warmup=50, n_measure=200):
    model.eval()
    images, _ = next(iter(loader))
    images = images.to(device)
    for _ in range(n_warmup):
        model(images); reset_net(model)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        model(images); reset_net(model)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_measure / images.size(0) * 1000


# ---------------------------------------------------------------------------
# Main benchmark API
# ---------------------------------------------------------------------------

def benchmark_sputnik_sparse(
    model: nn.Module,
    test_loader,
    device: torch.device,
    max_samples: int = 500,
    density_threshold: float = 0.15,
    use_compile: bool = True,
    n_warmup: int = 50,
    n_measure: int = 200,
) -> dict:
    """Benchmark dense vs Sputnik-sparse inference.

    Args:
        model:             Trained SNN model (on CUDA, fp32).
        test_loader:       Validation data loader.
        device:            CUDA device.
        max_samples:       Max samples for accuracy eval.
        density_threshold: Max density for sparse path.
        use_compile:       torch.compile for dense baseline.
        n_warmup:          Warmup iterations.
        n_measure:         Timed iterations.

    Returns:
        Dict with keys: dense, sparse, comparison, exec_stats.
    """
    results = {}
    compile_tag = " + torch.compile" if use_compile else ""

    # ---- Dense baseline ----
    print(f"--- Dense Baseline{compile_tag} ---")
    model_dense = copy.deepcopy(model)
    if use_compile:
        model_dense = torch.compile(model_dense)
    dense_acc = _evaluate(model_dense, test_loader, device, max_samples)
    dense_lat = _benchmark_latency(model_dense, test_loader, device,
                                   n_warmup=n_warmup, n_measure=n_measure)
    results['dense'] = {**dense_acc, 'latency_ms': dense_lat}
    print(f"  Acc@1: {dense_acc['acc1']:.2f}%  |  Latency: {dense_lat:.3f} ms/sample")
    del model_dense
    torch.cuda.empty_cache()

    # ---- Sputnik sparse ----
    print(f"\n--- Sputnik Sparse (threshold={density_threshold}) ---")
    accel = SputnikAccelerator(config={
        'density_threshold': density_threshold,
    })

    if not accel.available:
        print("  [SKIP] torch_sputnik not available")
        results['sparse'] = {'acc1': 0, 'acc5': 0, 'latency_ms': 0}
        results['comparison'] = {'speedup': 0, 'acc_drop': 0,
                                 'error': 'torch_sputnik not installed'}
        return results

    model_sparse = copy.deepcopy(model)
    accel.prepare(model_sparse)

    sparse_acc = _evaluate(model_sparse, test_loader, device, max_samples)
    sparse_lat = _benchmark_latency(model_sparse, test_loader, device,
                                    n_warmup=n_warmup, n_measure=n_measure)
    results['sparse'] = {**sparse_acc, 'latency_ms': sparse_lat}
    print(f"  Acc@1: {sparse_acc['acc1']:.2f}%  |  Latency: {sparse_lat:.3f} ms/sample")

    exec_stats = accel.get_stats()
    results['exec_stats'] = exec_stats

    if exec_stats['per_layer']:
        print(f"\n  {'Layer':<42} {'Type':<10} {'Density':>8} {'Sparse/Total':>14}")
        print("  " + "-" * 78)
        for name, ls in sorted(exec_stats['per_layer'].items()):
            sparse_n = ls.get('sparse_calls', ls.get('sparse_qk_calls', 0))
            total_n = ls['total_calls']
            print(f"  {name:<42} {ls['type']:<10} "
                  f"{ls['density']:>8.4f} {sparse_n:>6}/{total_n:<6}")

    accel.cleanup(model_sparse)
    del model_sparse
    torch.cuda.empty_cache()

    # ---- Comparison ----
    speedup = dense_lat / sparse_lat if sparse_lat > 0 else float('inf')
    acc_drop = dense_acc['acc1'] - sparse_acc['acc1']

    results['comparison'] = {
        'speedup': speedup,
        'acc_drop': acc_drop,
        'overall_density': exec_stats['density'],
    }

    print(f"\n{'='*60}")
    print(f"  Results (Sputnik)")
    print(f"{'='*60}")
    print(f"  Dense{compile_tag}:  {dense_lat:.3f} ms/sample  Acc: {dense_acc['acc1']:.2f}%")
    print(f"  Sputnik:       {sparse_lat:.3f} ms/sample  Acc: {sparse_acc['acc1']:.2f}%")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  Acc drop:      {acc_drop:+.2f}%")
    print(f"  Density:       {exec_stats['density']:.4f}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sputnik sparse inference benchmark for SNNs')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--n-warmup', type=int, default=20)
    parser.add_argument('--n-measure', type=int, default=50)
    parser.add_argument('--density-threshold', type=float, default=0.15)
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no-compile', dest='compile', action='store_false')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    ds_cfg = get_dataset_config(args.dataset)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T if args.T else config.get('T', 4)
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T or 4)
    else:
        raise ValueError("Must provide --config or --model")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(device).eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=0)

    return benchmark_sputnik_sparse(
        model, test_loader, device,
        max_samples=args.max_samples,
        density_threshold=args.density_threshold,
        use_compile=args.compile,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )


if __name__ == '__main__':
    main()
