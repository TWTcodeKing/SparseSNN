"""Benchmark script comparing sparse vs dense latency for SNN models.

Usage:
    python -m iengine.torch_sparse.benchmark \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth \
        --gpu-ids 0
"""

import argparse
import sys
import torch

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config,
    load_model_config, build_dataloaders, get_dataset_config,
)
from iengine.common.benchmark import SparseBenchmark, profile_layer_density
from .accelerator import TorchSparseAccelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark TorchSparse acceleration vs dense baseline'
    )
    # Model specification (one of --config or --model)
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer model')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar10, cifar100, imagenet, cifar10dvs)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    # Runtime
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU id(s) to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for benchmarking')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples to benchmark')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--density-threshold', type=float, default=0.15,
                        help='Density threshold for sparse execution')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--profile-density', action='store_true',
                        help='Profile per-layer density before benchmarking')
    parser.add_argument('--no-linear', action='store_true',
                        help='Disable sparse Linear acceleration')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable sparse attention acceleration')

    return parser.parse_args()


def build_model_from_args(args):
    """Build model from either --config or --model args."""
    ds_config = get_dataset_config(args.dataset)

    if args.config is not None:
        config = load_model_config(args.config)
        # Merge dataset config
        config['num_classes'] = ds_config['num_classes']
        config['img_size'] = ds_config['img_size']
        config['in_channels'] = ds_config['in_channels']
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model is not None:
        model = build_model(
            args.model,
            num_classes=ds_config['num_classes'],
            T=args.T,
        )
    else:
        print("Error: must specify --config or --model")
        sys.exit(1)

    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device setup
    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Density threshold: {args.density_threshold}")

    # Build model
    model = build_model_from_args(args)
    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        print("Checkpoint loaded.")

    model.eval()

    # Build dataloader (val only)
    ds_config = get_dataset_config(args.dataset)
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_config['img_size'], num_workers=4,
    )

    # Optional density profiling
    if args.profile_density:
        print("\nProfiling per-layer density...")
        density_profile = profile_layer_density(
            model, val_loader, device,
            max_samples=min(args.max_samples, 50),
            reset_fn=reset_net,
        )
        print(f"\n{'Layer':<50} {'Type':<10} {'Density':>8} {'Shape'}")
        print("-" * 90)
        for name, info in sorted(density_profile.items(),
                                  key=lambda x: x[1]['density']):
            print(f"{name:<50} {info['type']:<10} {info['density']:>8.4f} "
                  f"{info['shape']}")
        print()

    # Dense baseline benchmark
    print("Running dense baseline...")
    bench = SparseBenchmark()
    dense_stats = bench.run(
        model, val_loader, device,
        max_samples=args.max_samples,
        reset_fn=reset_net,
    )
    print(f"  Dense: {dense_stats['per_sample_ms']:.3f} ms/sample, "
          f"Acc: {dense_stats['accuracy']:.2f}%")

    # Apply sparse acceleration
    accel_config = {
        'density_threshold': args.density_threshold,
        'enable_linear': not args.no_linear,
        'enable_attention': not args.no_attention,
    }
    accel = TorchSparseAccelerator(config=accel_config)
    accel.prepare(model)

    # Sparse benchmark
    print("Running sparse benchmark...")
    sparse_stats = bench.run(
        model, val_loader, device,
        max_samples=args.max_samples,
        reset_fn=reset_net,
    )
    print(f"  Sparse: {sparse_stats['per_sample_ms']:.3f} ms/sample, "
          f"Acc: {sparse_stats['accuracy']:.2f}%")

    # Comparison
    comparison = bench.compare(dense_stats, sparse_stats,
                               backend_name=accel.name)

    # Sparse execution stats
    exec_stats = accel.get_stats()
    print(f"Sparse execution stats:")
    print(f"  Overall density: {exec_stats['density']:.4f}")
    print(f"  Total ops: {exec_stats['total_ops']:,}")
    print(f"  Effective ops: {exec_stats['effective_ops']:,}")

    if exec_stats['per_layer']:
        print(f"\n{'Layer':<50} {'Type':<10} {'Density':>8} "
              f"{'Sparse/Total':>14}")
        print("-" * 90)
        for name, layer_stats in sorted(exec_stats['per_layer'].items()):
            ltype = layer_stats['type']
            ldensity = layer_stats['density']
            total_calls = layer_stats['total_calls']
            if ltype == 'linear':
                sparse_calls = layer_stats['sparse_calls']
            else:
                sparse_calls = layer_stats.get('sparse_qk_calls', 0)
            print(f"{name:<50} {ltype:<10} {ldensity:>8.4f} "
                  f"{sparse_calls:>6}/{total_calls:<6}")

    # Cleanup
    accel.cleanup(model)

    return comparison


if __name__ == '__main__':
    main()
