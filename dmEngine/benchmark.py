"""Weight-sparse inference benchmark orchestrator.

Benchmarks inference speedup across a grid of (model x backend x sparsity).
Uses random sparse weights with dense inputs — no training needed.

Usage:
    python -m dmEngine.benchmark \
        --models resnet18 resnet50 \
        --backends torch_csr semi_structured triton_wsparse \
        --sparsities 0.5 0.7 0.9 0.95 \
        --batch-size 32 --gpu-ids 0 \
        --output-md dmEngine/results.md
"""

import argparse
import copy
import sys

import torch

from dmModels import build_resnet, RESNET_REGISTRY, make_dense_input
from dmEngine.backends import BACKEND_REGISTRY, load_backend
from dmEngine.common.timer import InferenceTimer, compare_latency
from dmEngine.common.report import export_results_md, print_result_row


def parse_args():
    p = argparse.ArgumentParser(description='Weight-sparse inference benchmark')
    p.add_argument('--models', nargs='+',
                   default=['resnet18', 'resnet50'],
                   choices=list(RESNET_REGISTRY.keys()),
                   help='Model architectures to benchmark')
    p.add_argument('--backends', nargs='+',
                   default=['torch_csr'],
                   choices=list(BACKEND_REGISTRY.keys()),
                   help='Sparse backends to benchmark')
    p.add_argument('--sparsities', nargs='+', type=float,
                   default=[0.5, 0.7, 0.9, 0.95],
                   help='Weight sparsity levels (0.0=dense, 1.0=all zeros)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-warmup', type=int, default=10)
    p.add_argument('--num-iters', type=int, default=100)
    p.add_argument('--num-classes', type=int, default=1000)
    p.add_argument('--gpu-ids', type=str, default='0',
                   help='GPU device id')
    p.add_argument('--output-md', type=str, default='dmEngine/results.md',
                   help='Output markdown file path')
    return p.parse_args()


def benchmark_one(model_name, backend_name, sparsity, args, device, timer):
    """Benchmark a single (model, backend, sparsity) configuration.

    Returns a result dict or None if the backend doesn't support this sparsity.
    """
    # Build fresh model each time (random init)
    model = build_resnet(model_name, num_classes=args.num_classes).to(device)
    model.eval()

    # Dense input generator
    def input_fn():
        return make_dense_input(
            batch_size=args.batch_size, channels=3,
            height=224, width=224, device=device,
        )

    # --- Dense baseline ---
    dense_stats = timer.run(model, input_fn,
                            num_warmup=args.num_warmup,
                            num_iters=args.num_iters, device=device)

    # --- Sparse run ---
    backend = load_backend(backend_name)

    # Check sparsity support
    supported = backend.supported_sparsities()
    if supported and sparsity not in supported:
        # Use the closest supported sparsity (e.g., semi_structured only does 0.5)
        print(f"    [SKIP] {backend.name} does not support sparsity={sparsity:.2f} "
              f"(supported: {supported})")
        return None

    # Work on a fresh copy so dense baseline isn't contaminated
    sparse_model = copy.deepcopy(model)
    try:
        backend.prepare(sparse_model, sparsity)
    except Exception as e:
        print(f"    [ERROR] {backend.name} prepare failed: {e}")
        return None

    try:
        sparse_stats = timer.run(sparse_model, input_fn,
                                 num_warmup=args.num_warmup,
                                 num_iters=args.num_iters, device=device)
    except Exception as e:
        print(f"    [ERROR] {backend.name} inference failed: {e}")
        backend.cleanup(sparse_model)
        return None

    backend.cleanup(sparse_model)

    # Compare
    cmp = compare_latency(dense_stats, sparse_stats)

    return {
        'model': model_name,
        'backend': backend_name,
        'sparsity': sparsity,
        'dense_mean_ms': cmp['dense_mean_ms'],
        'sparse_mean_ms': cmp['sparse_mean_ms'],
        'speedup': cmp['speedup'],
        'config': {
            'input_shape': '3x224x224',
            'batch_size': args.batch_size,
            'device': str(device),
            'num_warmup': args.num_warmup,
            'num_iters': args.num_iters,
        },
    }


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available()
                          else 'cpu')
    timer = InferenceTimer()
    results = []

    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Backends: {args.backends}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Header
    print(f"  {'Model':>12s} | {'Backend':>16s} | Spar | "
          f"{'Dense':>8s}    | {'Sparse':>8s}    | Speedup")
    print("  " + "-" * 78)

    for model_name in args.models:
        for backend_name in args.backends:
            for sparsity in args.sparsities:
                result = benchmark_one(
                    model_name, backend_name, sparsity,
                    args, device, timer,
                )
                if result is not None:
                    results.append(result)
                    print_result_row(
                        result['model'], result['backend'],
                        result['sparsity'],
                        result['dense_mean_ms'],
                        result['sparse_mean_ms'],
                        result['speedup'],
                    )

    # Export results
    if results:
        export_results_md(results, args.output_md)
        print(f"\nResults exported to {args.output_md}")
    else:
        print("\nNo results to export.")


if __name__ == '__main__':
    main()
