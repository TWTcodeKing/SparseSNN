"""Triton sparse inference benchmark for SNN models.

Exploits SNN activation sparsity via custom Triton GPU kernels:
  - Conv2d: im2col + SpMM that skips zero columns
  - SSA attention: block-sparse Q @ K^T

Note: Triton kernel launch overhead dominates for small CIFAR tensors.
Best suited for larger inputs (ImageNet) or very low density (<15%).

Usage:
    python -m iengine.triton_sparse.inference \
        --config configs/spikformer/spikformer_cifar.yaml \
        --checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets

    python -m iengine.triton_sparse.inference \
        --model ms_resnet34 \
        --checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets --T 6
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
from .accelerator import TritonSparseAccelerator


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

def benchmark_triton_sparse(
    model: nn.Module,
    test_loader,
    device: torch.device,
    max_samples: int = 500,
    density_threshold: float = 0.5,
    block_size: int = 16,
    min_tensor_size: int = 4096,
    enable_conv2d: bool = True,
    enable_attention: bool = True,
    use_compile: bool = True,
    n_warmup: int = 50,
    n_measure: int = 200,
    fuse_neurons: bool = False,
) -> dict:
    """Benchmark dense vs Triton-sparse inference.

    Args:
        model:             Trained SNN model (on CUDA, fp32).
        test_loader:       Validation data loader.
        device:            CUDA device.
        max_samples:       Max samples for accuracy eval.
        density_threshold: Max density for sparse Conv2d path.
        block_size:        Tile size for block-sparse attention.
        min_tensor_size:   Min M*K_nz for Triton launch.
        enable_conv2d:     Enable Triton sparse Conv2d.
        enable_attention:  Enable block-sparse attention.
        use_compile:       torch.compile for dense baseline.
        n_warmup:          Warmup iterations.
        n_measure:         Timed iterations.

    Returns:
        Dict with keys: dense, sparse, comparison, exec_stats.
    """
    from iengine.common.neuron_utils import maybe_fuse_neurons
    results = {}
    compile_tag = " + torch.compile" if use_compile else ""
    neuron_tag = " + fused neurons" if fuse_neurons else ""

    # ---- Dense baseline ----
    print(f"--- Dense Baseline{compile_tag}{neuron_tag} ---")
    model_dense = copy.deepcopy(model)
    maybe_fuse_neurons(model_dense, fuse=fuse_neurons)
    if use_compile:
        model_dense = torch.compile(model_dense)
    dense_acc = _evaluate(model_dense, test_loader, device, max_samples)
    dense_lat = _benchmark_latency(model_dense, test_loader, device,
                                   n_warmup=n_warmup, n_measure=n_measure)
    results['dense'] = {**dense_acc, 'latency_ms': dense_lat}
    print(f"  Acc@1: {dense_acc['acc1']:.2f}%  |  Latency: {dense_lat:.3f} ms/sample")
    del model_dense
    torch.cuda.empty_cache()

    # ---- Triton sparse ----
    print(f"\n--- TritonSparse (threshold={density_threshold}, block={block_size}) ---")
    accel = TritonSparseAccelerator(config={
        'density_threshold': density_threshold,
        'block_size': block_size,
        'min_tensor_size': min_tensor_size,
        'enable_conv2d': enable_conv2d,
        'enable_attention': enable_attention,
    })
    model_sparse = copy.deepcopy(model)
    maybe_fuse_neurons(model_sparse, fuse=fuse_neurons)
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
    print(f"  Results (TritonSparse)")
    print(f"{'='*60}")
    print(f"  Dense{compile_tag}:  {dense_lat:.3f} ms/sample  Acc: {dense_acc['acc1']:.2f}%")
    print(f"  Triton sparse: {sparse_lat:.3f} ms/sample  Acc: {sparse_acc['acc1']:.2f}%")
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
        description='Triton sparse inference benchmark for SNNs')
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
    parser.add_argument('--density-threshold', type=float, default=0.5)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--min-tensor-size', type=int, default=4096)
    parser.add_argument('--no-conv2d', action='store_true')
    parser.add_argument('--no-attention', action='store_true')
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no-compile', dest='compile', action='store_false')
    parser.add_argument('--fuse-neurons', action='store_true', default=False,
                        help='Replace LIF/IF neurons with fused Triton kernels')
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

    return benchmark_triton_sparse(
        model, test_loader, device,
        max_samples=args.max_samples,
        density_threshold=args.density_threshold,
        block_size=args.block_size,
        min_tensor_size=args.min_tensor_size,
        enable_conv2d=not args.no_conv2d,
        enable_attention=not args.no_attention,
        use_compile=args.compile,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        fuse_neurons=args.fuse_neurons,
    )


if __name__ == '__main__':
    main()
