"""Benchmark script for Sputnik sparse acceleration.

Runs two types of benchmarks:
1. Micro-benchmark: Sputnik SpMM vs torch.sparse.mm vs dense at various densities
2. Full model benchmark: compare dense vs Sputnik-accelerated inference

Usage:
    # Micro-benchmark only (no model/data needed)
    uv run python -m iengine.sputnik_sparse.benchmark --micro-only

    # Full model benchmark
    uv run python -m iengine.sputnik_sparse.benchmark \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth \
        --gpu-ids 0
"""

import argparse
import sys
import time

import torch
import torch.nn as nn

# Check torch_sputnik availability before proceeding
_sputnik_available = False
try:
    import torch_sputnik
    _sputnik_available = True
except ImportError:
    pass


def print_build_instructions():
    """Print build instructions and exit."""
    print("=" * 60)
    print("  torch_sputnik is not installed")
    print("=" * 60)
    print()
    print("Build from source:")
    print("  bash iengine/sputnik_sparse/build.sh")
    print()
    print("Or manually:")
    print("  1. git clone https://github.com/google-research/sputnik")
    print("  2. cd sputnik && mkdir build && cd build")
    print('  3. cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="89" -DBUILD_TEST=OFF')
    print("  4. make -j$(nproc)")
    print("  5. git clone https://github.com/mabdullahsoyturk/Torch-Sputnik")
    print("  6. cd Torch-Sputnik")
    print("  7. SPUTNIK_BUILD_DIR=<path/to/sputnik/build> pip install -e .")
    print()
    print("See iengine/sputnik_sparse/INSTALL.md for full instructions.")
    print("=" * 60)


def micro_benchmark(device: torch.device, warmup: int = 10, repeats: int = 100):
    """Micro-benchmark: Sputnik SpMM vs torch.sparse.mm vs dense matmul.

    Tests at 1%, 5%, 10%, 15% density for (4096, 384) x (384, 384) matrices,
    which approximates typical SNN Linear layer dimensions.
    """
    print("\n" + "=" * 70)
    print("  Micro-Benchmark: SpMM at Various Densities")
    print("=" * 70)

    if not _sputnik_available:
        print("\n  [SKIP] torch_sputnik not available — only dense and torch.sparse")
        print()

    densities = [0.01, 0.05, 0.10, 0.15]
    shapes = [
        (4096, 384, 384),   # Typical MLP: (T*B*N, C) @ (C, C)
        (1024, 384, 384),   # Smaller batch
        (256, 64, 64),      # Attention head: (N, D) @ (D, N)
    ]

    for M, K, N in shapes:
        print(f"\n  Shape: ({M}, {K}) @ ({K}, {N})")
        print(f"  {'Density':>8s}  {'Dense (ms)':>11s}  {'torch.sparse':>13s}  ", end="")
        if _sputnik_available:
            print(f"{'Sputnik (ms)':>13s}  {'Speedup':>8s}")
        else:
            print(f"{'Speedup (ts)':>13s}")
        print("  " + "-" * 66)

        weight = torch.randn(K, N, device=device, dtype=torch.float32)

        for density in densities:
            # Create sparse input with specified density
            nnz = int(M * K * density)
            input_dense = torch.zeros(M, K, device=device, dtype=torch.float32)
            indices = torch.randint(0, M * K, (nnz,), device=device)
            input_dense.view(-1)[indices] = 1.0

            # --- Dense matmul ---
            for _ in range(warmup):
                _ = input_dense @ weight
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = input_dense @ weight
            torch.cuda.synchronize()
            dense_ms = (time.perf_counter() - t0) / repeats * 1000

            # --- torch.sparse.mm ---
            input_csr = input_dense.to_sparse_csr()
            for _ in range(warmup):
                _ = torch.sparse.mm(input_csr, weight)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _ = torch.sparse.mm(input_csr, weight)
            torch.cuda.synchronize()
            torch_sparse_ms = (time.perf_counter() - t0) / repeats * 1000

            ts_speedup = dense_ms / max(torch_sparse_ms, 1e-6)

            # --- Sputnik SpMM ---
            if _sputnik_available:
                from .sparse_linear import _to_sputnik_csr
                values, row_indices, row_offsets, col_indices, nnz_val = \
                    _to_sputnik_csr(input_dense)

                for _ in range(warmup):
                    _ = torch_sputnik.spmm(
                        M, K, N, nnz_val,
                        row_indices, values, row_offsets, col_indices,
                        weight
                    )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(repeats):
                    _ = torch_sputnik.spmm(
                        M, K, N, nnz_val,
                        row_indices, values, row_offsets, col_indices,
                        weight
                    )
                torch.cuda.synchronize()
                sputnik_ms = (time.perf_counter() - t0) / repeats * 1000
                sp_speedup = dense_ms / max(sputnik_ms, 1e-6)

                print(f"  {density:>7.0%}  {dense_ms:>11.3f}  {torch_sparse_ms:>13.3f}  "
                      f"{sputnik_ms:>13.3f}  {sp_speedup:>7.2f}x")
            else:
                print(f"  {density:>7.0%}  {dense_ms:>11.3f}  {torch_sparse_ms:>13.3f}  "
                      f"{ts_speedup:>12.2f}x")

    print()


def full_model_benchmark(args):
    """Full model benchmark: dense vs Sputnik-accelerated inference."""
    from tengine.utils import (
        set_seed, build_model_from_config, load_model_config,
        build_dataloaders, get_dataset_config,
    )
    from iengine.common.benchmark import SparseBenchmark
    from models import reset_net
    from .accelerator import SputnikAccelerator

    set_seed(42)

    # Setup device
    gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}')

    # Load config and build model
    config = load_model_config(args.config)
    ds_config = get_dataset_config(args.dataset)
    config.update(ds_config)
    config['T'] = config.get('T', 4)

    model = build_model_from_config(config)
    model = model.to(device)

    # Load checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # Build dataloader
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size,
        img_size=config.get('img_size'),
    )

    bench = SparseBenchmark()

    # --- Dense baseline ---
    print("\n[1/3] Running dense baseline...")
    dense_stats = bench.run(model, val_loader, device,
                            max_samples=args.max_samples,
                            reset_fn=reset_net)
    print(f"  Dense: {dense_stats['per_sample_ms']:.3f} ms/sample, "
          f"Acc: {dense_stats['accuracy']:.2f}%")

    # --- Sputnik accelerated ---
    accel = SputnikAccelerator({
        'density_threshold': args.density_threshold,
        'min_elements': args.min_elements,
    })

    if not accel.available:
        print("\n[SKIP] Sputnik not available — cannot run sparse benchmark")
        print_build_instructions()
        return

    print("\n[2/3] Running Sputnik-accelerated inference...")
    accel.prepare(model)
    sparse_stats = bench.run(model, val_loader, device,
                             max_samples=args.max_samples,
                             reset_fn=reset_net)
    print(f"  Sparse: {sparse_stats['per_sample_ms']:.3f} ms/sample, "
          f"Acc: {sparse_stats['accuracy']:.2f}%")

    # --- Comparison ---
    print("\n[3/3] Comparison:")
    bench.compare(dense_stats, sparse_stats, backend_name='Sputnik')

    # --- Per-layer stats ---
    exec_stats = accel.get_stats()
    print("\nPer-layer statistics:")
    print(f"  {'Layer':<40s}  {'Density':>8s}  {'Sparse/Total':>13s}")
    print("  " + "-" * 65)
    for name, layer_stats in exec_stats['per_layer'].items():
        d = layer_stats.get('density', 1.0)
        sparse_c = layer_stats.get('sparse_calls', 0)
        dense_c = layer_stats.get('dense_calls', 0)
        total_c = sparse_c + dense_c
        print(f"  {name:<40s}  {d:>7.1%}  {sparse_c:>5d}/{total_c:<5d}")

    # Cleanup
    accel.cleanup(model)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Sputnik sparse acceleration for SNN models')

    # Micro-benchmark
    parser.add_argument('--micro-only', action='store_true',
                        help='Run micro-benchmark only (no model needed)')

    # Model config
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for transformer model')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to dataset root')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs (comma-separated)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for benchmarking')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='Max samples to benchmark')

    # Sparse config
    parser.add_argument('--density-threshold', type=float, default=0.15,
                        help='Max density for sparse execution')
    parser.add_argument('--min-elements', type=int, default=4096,
                        help='Min elements to justify sparse overhead')

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for benchmarking.")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu_ids.split(",")[0]}')

    # Always run micro-benchmark
    micro_benchmark(device)

    if args.micro_only:
        if not _sputnik_available:
            print()
            print_build_instructions()
        return

    # Full model benchmark requires config + checkpoint
    if args.config is None:
        print("ERROR: --config is required for full model benchmark.")
        print("Use --micro-only for micro-benchmark without a model.")
        sys.exit(1)

    full_model_benchmark(args)


if __name__ == '__main__':
    main()
