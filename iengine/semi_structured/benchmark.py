"""Benchmark semi-structured 2:4 sparsity: accuracy and latency vs dense baseline.

Usage:
    uv run python -m iengine.semi_structured.benchmark \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth \
        --gpu-ids 0
"""

import argparse
import sys
import os

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tengine.utils import (
    set_seed,
    build_model_from_config,
    load_model_config,
    build_dataloaders,
    get_dataset_config,
)
from models import reset_net
from iengine.common.benchmark import SparseBenchmark, profile_layer_density
from .accelerator import SemiStructuredAccelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark 2:4 semi-structured sparsity on SNN models'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Model config YAML path')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root path for dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples for benchmark')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-head', action='store_true', default=True,
                        help='Exclude classification head (default: True)')
    parser.add_argument('--no-exclude-head', dest='exclude_head',
                        action='store_false')
    parser.add_argument('--profile-density', action='store_true', default=True,
                        help='Profile activation density per layer')
    parser.add_argument('--T', type=int, default=None,
                        help='Override number of timesteps')
    return parser.parse_args()


def load_trained_model(config_path, dataset_name, checkpoint_path, device, T=None):
    """Load a trained model from config + checkpoint."""
    config = load_model_config(config_path)
    ds_config = get_dataset_config(dataset_name)
    config.update(ds_config)
    if T is not None:
        config['T'] = T
    elif 'T' not in config:
        config['T'] = 4

    model = build_model_from_config(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def print_layer_report(stats, density_profile=None):
    """Print per-layer report of pruning and density."""
    per_layer = stats.get('per_layer', {})
    if not per_layer:
        print("  No layers to report.")
        return

    print(f"\n{'Layer':<45} {'Shape':<18} {'Converted':<10} "
          f"{'W.Dens':<8} {'A.Dens':<8} {'Compound':<10}")
    print('-' * 105)

    for name, info in sorted(per_layer.items()):
        shape_str = f"{info['shape'][0]}x{info['shape'][1]}"
        conv_str = 'Yes' if info['converted'] else 'No'
        w_dens = f"{info['weight_density']:.3f}"
        a_dens = f"{info['activation_density']:.3f}"
        compound = f"{info['density']:.3f}"
        print(f"  {name:<43} {shape_str:<18} {conv_str:<10} "
              f"{w_dens:<8} {a_dens:<8} {compound:<10}")

    print(f"\n  Total layers converted: {stats['layers_converted']}")
    print(f"  Total layers skipped:   {stats['layers_skipped']}")
    print(f"  Overall density:        {stats['density']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Build dataloader (test split only)
    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        num_workers=4, distributed=False,
    )

    # Load model
    model = load_trained_model(
        args.config, args.dataset, args.checkpoint, device, T=args.T
    )
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    bench = SparseBenchmark()
    reset_fn = lambda m: reset_net(m)

    # ---- Dense baseline ----
    print("\n--- Dense Baseline ---")
    dense_stats = bench.run(
        model, test_loader, device,
        max_samples=args.max_samples, reset_fn=reset_fn,
    )
    print(f"  Accuracy: {dense_stats['accuracy']:.2f}%")
    print(f"  Latency:  {dense_stats['per_sample_ms']:.3f} ms/sample")

    # ---- Profile activation density (before conversion) ----
    density_profile = None
    if args.profile_density:
        print("\n--- Profiling Activation Density ---")
        density_profile = profile_layer_density(
            model, test_loader, device,
            max_samples=50, reset_fn=reset_fn,
        )
        linear_densities = {
            k: v for k, v in density_profile.items() if v['type'] == 'Linear'
        }
        if linear_densities:
            avg_act = sum(v['density'] for v in linear_densities.values()) / len(linear_densities)
            print(f"  Linear layers profiled: {len(linear_densities)}")
            print(f"  Average activation density: {avg_act:.4f}")

    # ---- Apply semi-structured sparsity ----
    print("\n--- Applying 2:4 Semi-Structured Sparsity ---")
    accel = SemiStructuredAccelerator({
        'exclude_head': args.exclude_head,
    })

    if density_profile:
        accel.set_activation_density(density_profile)

    model = accel.prepare(model)

    # ---- Sparse benchmark ----
    print("\n--- Sparse (2:4) Benchmark ---")
    sparse_stats = bench.run(
        model, test_loader, device,
        max_samples=args.max_samples, reset_fn=reset_fn,
    )
    print(f"  Accuracy: {sparse_stats['accuracy']:.2f}%")
    print(f"  Latency:  {sparse_stats['per_sample_ms']:.3f} ms/sample")

    # ---- Comparison ----
    comparison = bench.compare(dense_stats, sparse_stats, backend_name=accel.name)

    # ---- Per-layer report ----
    stats = accel.get_stats()
    print_layer_report(stats, density_profile)

    # ---- Cleanup ----
    model = accel.cleanup(model)
    print("\nModel restored to dense fp32.")

    return comparison


if __name__ == '__main__':
    main()
