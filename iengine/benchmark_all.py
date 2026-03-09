"""
Benchmark all sparse acceleration backends on all SNN models.

Compares dense baseline against each available backend, reporting latency,
accuracy, and speedup for every (model, backend) pair.

Usage:
    uv run python -m iengine.benchmark_all \
        --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 0

    # Specific backends only:
    uv run python -m iengine.benchmark_all \
        --dataset cifar100 --data-root /home/twt/datasets/ --gpu-ids 0 \
        --backends torch_sparse semi_structured
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from iengine.common.benchmark import SparseBenchmark, profile_layer_density
from tengine.utils import (
    set_seed, build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)

# Model configurations: (name, config_or_model, checkpoint, is_config)
MODEL_CONFIGS = [
    (
        'Spikformer',
        'configs/spikformer/spikformer_cifar.yaml',
        'output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth',
        True,
    ),
    (
        'SEW-ResNet34',
        'sew_resnet34',
        'output/sew_resnet34_cifar100_bs100_lr0.1/best.pth',
        False,
    ),
    (
        'MS-ResNet34',
        'ms_resnet34',
        'output/ms_resnet34_cifar100_bs100_lr0.1/best.pth',
        False,
    ),
]

BACKEND_REGISTRY = {
    'torch_sparse': {
        'module': 'iengine.torch_sparse',
        'class': 'TorchSparseAccelerator',
        'config': {'density_threshold': 0.15},
    },
    'triton_sparse': {
        'module': 'iengine.triton_sparse',
        'class': 'TritonSparseAccelerator',
        'config': {'density_threshold': 0.5, 'block_size': 16},
    },
    'semi_structured': {
        'module': 'iengine.semi_structured',
        'class': 'SemiStructuredAccelerator',
        'config': {'exclude_head': True},
    },
    'sputnik_sparse': {
        'module': 'iengine.sputnik_sparse',
        'class': 'SputnikAccelerator',
        'config': {'density_threshold': 0.15},
    },
}


def load_backend(name):
    """Dynamically import a backend accelerator class."""
    info = BACKEND_REGISTRY[name]
    try:
        mod = __import__(info['module'], fromlist=[info['class']])
        cls = getattr(mod, info['class'])
        return cls(config=info['config'])
    except (ImportError, Exception) as e:
        print(f"  [SKIP] {name}: {e}")
        return None


def build_model_for_benchmark(model_info, dataset, data_root, T=4):
    """Build and load a model from config or name."""
    name, config_or_model, checkpoint, is_config = model_info
    ds_cfg = get_dataset_config(dataset)
    num_classes = ds_cfg['num_classes']
    img_size = ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    if not os.path.exists(checkpoint):
        print(f"  [SKIP] {name}: checkpoint not found at {checkpoint}")
        return None

    if is_config:
        model_cfg = load_model_config(config_or_model)
        model_cfg.update({
            'num_classes': num_classes, 'T': T,
            'img_size': img_size, 'in_channels': in_channels,
        })
        model = build_model_from_config(model_cfg)
    else:
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in config_or_model:
            model_kwargs['T'] = T
            model_kwargs['connect_f'] = 'ADD'
        else:
            model_kwargs['time_window'] = T
        model = build_model(config_or_model, **model_kwargs)

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model


def reset_fn(model):
    reset_net(model)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark all sparse backends')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--backends', nargs='+', default=None,
                        choices=list(BACKEND_REGISTRY.keys()),
                        help='Backends to test (default: all available)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model names to test (default: all with checkpoints)')
    parser.add_argument('--profile-density', action='store_true',
                        help='Profile per-layer density before benchmarking')
    return parser.parse_args()


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(42)

    ds_cfg = get_dataset_config(args.dataset)
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, batch_size=16,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    backend_names = args.backends or list(BACKEND_REGISTRY.keys())
    model_configs = MODEL_CONFIGS
    if args.models:
        model_configs = [m for m in MODEL_CONFIGS if m[0] in args.models]

    # Results table
    all_results = []

    print(f"\n{'='*80}")
    print(f"  Sparse Acceleration Benchmark — All Backends")
    print(f"  Dataset: {args.dataset}  |  Max samples: {args.max_samples}")
    print(f"  Backends: {', '.join(backend_names)}")
    print(f"{'='*80}\n")

    for model_info in model_configs:
        model_name = model_info[0]
        print(f"\n{'─'*80}")
        print(f"  Model: {model_name}")
        print(f"{'─'*80}")

        model = build_model_for_benchmark(model_info, args.dataset, args.data_root, args.T)
        if model is None:
            continue

        model = model.to(device)
        model.eval()

        # Optional density profiling
        if args.profile_density:
            print(f"\n  Profiling per-layer density...")
            density_profile = profile_layer_density(
                model, val_loader, device, max_samples=50, reset_fn=reset_fn
            )
            for lname, info in sorted(density_profile.items()):
                print(f"    {lname:<50} {info['type']:<12} density={info['density']:.4f}")
            reset_net(model)

        # Dense baseline
        print(f"\n  Running dense baseline...")
        dense_stats = SparseBenchmark.run(
            model, val_loader, device,
            max_samples=args.max_samples, reset_fn=reset_fn
        )
        print(f"    Dense: {dense_stats['per_sample_ms']:.3f} ms/sample  "
              f"Acc: {dense_stats['accuracy']:.2f}%")

        # Test each backend
        for backend_name in backend_names:
            print(f"\n  Testing backend: {backend_name}...")

            # Reload model fresh for each backend (semi-structured modifies weights)
            if backend_name == 'semi_structured':
                model_fresh = build_model_for_benchmark(
                    model_info, args.dataset, args.data_root, args.T
                )
                if model_fresh is None:
                    continue
                model_fresh = model_fresh.to(device)
                model_fresh.eval()
            else:
                model_fresh = model

            accel = load_backend(backend_name)
            if accel is None:
                continue

            try:
                model_sparse = accel.prepare(model_fresh)

                sparse_stats = SparseBenchmark.run(
                    model_sparse, val_loader, device,
                    max_samples=args.max_samples, reset_fn=reset_fn
                )

                result = SparseBenchmark.compare(dense_stats, sparse_stats, backend_name)
                result['model'] = model_name

                # Get backend-specific stats
                try:
                    backend_stats = accel.get_stats()
                    result['backend_stats'] = backend_stats
                except Exception:
                    pass

                all_results.append(result)

                accel.cleanup(model_sparse)

            except Exception as e:
                print(f"    [ERROR] {backend_name}: {e}")
                import traceback
                traceback.print_exc()

        # Reset model state
        reset_net(model)

    # Summary table
    if all_results:
        print(f"\n\n{'='*90}")
        print(f"  SUMMARY: All Backends × All Models")
        print(f"{'='*90}")
        print(f"  {'Model':<18} {'Backend':<18} {'Dense ms':>10} {'Sparse ms':>10} "
              f"{'Speedup':>8} {'Dense Acc':>10} {'Sparse Acc':>10} {'Acc Δ':>8}")
        print(f"  {'─'*86}")

        for r in all_results:
            print(f"  {r['model']:<18} {r['backend']:<18} "
                  f"{r['dense_ms']:>10.3f} {r['sparse_ms']:>10.3f} "
                  f"{r['speedup']:>7.2f}x "
                  f"{r['dense_acc']:>9.2f}% {r['sparse_acc']:>9.2f}% "
                  f"{r['acc_delta']:>+7.2f}%")

        print(f"{'='*90}\n")

    print("Done.")


if __name__ == '__main__':
    main()
