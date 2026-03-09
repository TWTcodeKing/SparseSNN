"""Benchmark script for Triton sparse acceleration.

Loads a trained SNN model, runs dense vs Triton-sparse inference comparison,
and reports latency, accuracy, and sparsity statistics.

Works with all three model families:
- Spikformer (Conv2d in SPS + SSA attention)
- SEW-ResNet34 (Conv2d heavy)
- MS-ResNet34 (Conv2d heavy)

Usage:
    uv run python -m iengine.triton_sparse.benchmark \
        --model sew_resnet34 \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/sew_resnet34_cifar100_bs100_lr0.1/best.pth \
        --gpu-ids 0

    uv run python -m iengine.triton_sparse.benchmark \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0

Performance caveat: For small CIFAR tensors (32x32), Triton kernel launch
overhead may negate sparsity savings. The sparse path is most beneficial for
larger inputs or when input density is very low (<15%).
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)
from iengine.common.benchmark import SparseBenchmark, profile_layer_density
from iengine.triton_sparse.accelerator import TritonSparseAccelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark Triton sparse acceleration for SNN models'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None,
                       help='ResNet model name (e.g. sew_resnet34, ms_resnet34)')
    group.add_argument('--config', type=str, default=None,
                       help='YAML config path for transformer models')

    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples for benchmarking')

    # Sparse config
    parser.add_argument('--density-threshold', type=float, default=0.5,
                        help='Max column density for sparse conv path')
    parser.add_argument('--block-size', type=int, default=16,
                        help='Block size for block-sparse attention')
    parser.add_argument('--min-tensor-size', type=int, default=4096,
                        help='Min M*K_nz to justify Triton launch')

    # Extra options
    parser.add_argument('--profile-density', action='store_true',
                        help='Profile per-layer density before benchmarking')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_model(args, device):
    """Build and load model from checkpoint."""
    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    if args.config:
        model_cfg = load_model_config(args.config)
        model_cfg.update({
            'num_classes': num_classes, 'T': args.T,
            'img_size': img_size, 'in_channels': in_channels,
        })
        model = build_model_from_config(model_cfg)
        model_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
            model_kwargs['connect_f'] = 'ADD'
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, **model_kwargs)
        model_name = args.model

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, model_name


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(args.seed)

    ds_cfg = get_dataset_config(args.dataset)
    img_size = ds_cfg['img_size']

    print(f"\n{'='*60}")
    print(f"  Triton Sparse Benchmark")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Device: cuda:{gpu_ids[0]}")

    # Build dataloader
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size,
        img_size=img_size, num_workers=4,
    )

    # Load model
    model, model_name = load_model(args, device)
    print(f"  Model: {model_name}")

    # Optional density profiling
    if args.profile_density:
        print(f"\n--- Per-layer density profile ---")
        density_info = profile_layer_density(
            model, val_loader, device, max_samples=50, reset_fn=reset_net,
        )
        print(f"  {'Layer':<50} {'Type':<12} {'Density':>8} {'Shape'}")
        print(f"  {'-'*90}")
        for name, info in sorted(density_info.items()):
            print(f"  {name[:49]:<50} {info['type']:<12} "
                  f"{info['density']:>7.4f}  {info['shape']}")
        print()

    # Dense benchmark
    print(f"\n--- Dense baseline ---")
    bench = SparseBenchmark()
    dense_stats = bench.run(
        model, val_loader, device,
        max_samples=args.max_samples, reset_fn=reset_net,
    )
    print(f"  Latency: {dense_stats['per_sample_ms']:.3f} ms/sample  "
          f"Acc: {dense_stats['accuracy']:.2f}%")

    # Sparse benchmark
    print(f"\n--- Triton sparse ---")
    sparse_config = {
        'density_threshold': args.density_threshold,
        'block_size': args.block_size,
        'min_tensor_size': args.min_tensor_size,
    }
    accel = TritonSparseAccelerator(config=sparse_config)
    model = accel.prepare(model)

    sparse_stats = bench.run(
        model, val_loader, device,
        max_samples=args.max_samples, reset_fn=reset_net,
    )
    print(f"  Latency: {sparse_stats['per_sample_ms']:.3f} ms/sample  "
          f"Acc: {sparse_stats['accuracy']:.2f}%")

    # Get sparse execution stats
    exec_stats = accel.get_stats()

    # Compare
    comparison = bench.compare(dense_stats, sparse_stats, backend_name='TritonSparse')

    # Print detailed sparse stats
    print(f"--- Sparse execution details ---")
    print(f"  Conv2d: {exec_stats['sparse_launches']} Triton launches, "
          f"{exec_stats['dense_fallback']} dense fallbacks, "
          f"{exec_stats['skipped_zero']} zero-input skips")
    if exec_stats['total_ops'] > 0:
        print(f"  Conv2d ops: {exec_stats['effective_ops']:,} / {exec_stats['total_ops']:,} "
              f"({exec_stats['density']:.4f} density)")
    if exec_stats['attn_sparse_calls'] > 0:
        print(f"  SSA: {exec_stats['attn_sparse_calls']} calls, "
              f"{exec_stats['attn_computed_blocks']} / {exec_stats['attn_total_blocks']} blocks "
              f"({exec_stats['attn_block_density']:.4f} block density)")
        print(f"  SSA dense fallbacks: {exec_stats['attn_dense_fallback']}")

    # Honest assessment
    if comparison['speedup'] < 1.0:
        print(f"\n  NOTE: Sparse path is {1/comparison['speedup']:.2f}x SLOWER than dense.")
        print(f"  This is expected for small tensors (CIFAR {img_size}x{img_size}) where")
        print(f"  Triton kernel launch overhead exceeds sparsity savings.")
        print(f"  Sparse kernels are more beneficial for larger inputs (e.g. ImageNet 224x224)")
        print(f"  or models with very low activation density (<15%).")

    # Cleanup
    model = accel.cleanup(model)


if __name__ == '__main__':
    main()
