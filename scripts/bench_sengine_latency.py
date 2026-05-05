#!/usr/bin/env python3
"""Benchmark sengine inference latency (C++ executor) for SNN models.

Exports plugin-mode ONNX (TDL + FusedIF/LIF custom ops), builds sengine
with TileLang kernels and BA-MTTS scheduling, benchmarks via C++ CUDA
Graph executor.

Usage:
    # ResNet
    python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset cifar100

    # Transformer
    python scripts/bench_sengine_latency.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar100

    # With checkpoint + custom batch sizes
    python scripts/bench_sengine_latency.py --model sew_resnet34 --dataset cifar100 \
        --checkpoint obc_pt/sew_resnet34_cifar100_sbc_2_4_global.pth \
        --batch-sizes 1,4,8,16,32
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

import torch


def export_plugin_onnx(args, ds_cfg, img_size, output_dir):
    """Export plugin-mode ONNX for sengine."""
    from sengine.scripts.export_onnx import export_model

    model_name = args.model or os.path.splitext(os.path.basename(args.config))[0]
    plugin_path = os.path.join(output_dir,
                               f"{model_name}_{args.dataset}_plugin.onnx")

    if os.path.exists(plugin_path):
        print(f"  Plugin ONNX exists: {plugin_path}")
        return plugin_path

    print(f"  Exporting plugin ONNX: {model_name}...")
    result = export_model(
        model_name=args.model or model_name,
        output_dir=output_dir,
        T=args.T,
        dataset=args.dataset,
        img_size=img_size,
        config=args.config,
        checkpoint=args.checkpoint,
    )
    if result and os.path.exists(result):
        print(f"  Saved: {result}")
    else:
        raise RuntimeError(f"ONNX export failed for {model_name}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sengine inference latency (C++ executor)")

    # Model
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')
    parser.add_argument('--config', type=str, default=None,
                        help='Transformer YAML config path')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar100, imagenet, etc.)')
    parser.add_argument('--T', type=int, default=4,
                        help='Temporal steps (default: 4)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (auto-detected from dataset if omitted)')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (random weights if omitted)')

    # Benchmark
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16',
                        help='Comma-separated batch sizes (default: 1,4,8,16)')
    parser.add_argument('--warmup', type=int, default=200,
                        help='Warmup iterations (default: 200)')
    parser.add_argument('--iters', type=int, default=1000,
                        help='Measurement iterations (default: 1000)')

    # Output
    parser.add_argument('--export-dir', type=str, default='sengine/exports',
                        help='Directory for ONNX and .sengine files')

    args = parser.parse_args()

    if not args.model and not args.config:
        parser.error("Provide --model or --config")

    from tengine.utils import get_dataset_config
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']
    tag = args.model or os.path.splitext(os.path.basename(args.config))[0]

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.export_dir, exist_ok=True)

    # Step 1: Export plugin ONNX (once, shared across batch sizes)
    print(f"\n{'='*60}")
    print(f"  Phase 1: Export plugin-mode ONNX")
    print(f"{'='*60}")
    plugin_onnx = export_plugin_onnx(args, ds_cfg, img_size, args.export_dir)

    # Step 2: Build/load + benchmark for each batch size
    print(f"\n{'='*60}")
    print(f"  Phase 2: Build + benchmark sengine (C++ executor)")
    print(f"{'='*60}")

    import sengine

    results = {}
    for B in batch_sizes:
        sengine_path = os.path.join(args.export_dir,
                                     f"{tag}_{args.dataset}_B{B}.sengine")
        try:
            if os.path.exists(sengine_path):
                print(f"\n  [B={B}] Loading cached: {sengine_path}")
                t0 = time.time()
                engine = sengine.load(sengine_path)
                print(f"          Loaded in {time.time()-t0:.1f}s")
            else:
                print(f"\n  [B={B}] Building sengine from ONNX...")
                t0 = time.time()
                engine = sengine.build(plugin_onnx, T=args.T, batch_size=B)
                build_s = time.time() - t0
                print(f"          Built in {build_s:.1f}s")
                engine.save(sengine_path)
                size_mb = os.path.getsize(sengine_path) / 1e6
                print(f"          Saved: {sengine_path} ({size_mb:.1f} MB)")

            # Verify C++ executor is active
            if engine._use_python_runtime:
                print(f"  WARNING: B={B} using Python runtime (C++ executor unavailable)")

            # Benchmark
            ms = engine.benchmark(warmup=args.warmup, iters=args.iters)
            fps = 1000.0 / ms * B if ms > 0 else 0
            runtime = "C++" if not engine._use_python_runtime else "Python"
            results[B] = {'mean_ms': ms, 'throughput': fps, 'runtime': runtime}
            print(f"  [B={B}] Latency: {ms:.3f} ms ({fps:.0f} img/s) [{runtime}]")

            engine.destroy()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [B={B}] FAILED: {e}")

    # Results table
    print(f"\n{'='*70}")
    print(f"  sengine (C++ CUDA Graph) | {tag} | {args.dataset} | T={args.T}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    print(f"  {'Batch':<8} {'Latency (ms)':>14} {'Throughput':>14} {'Runtime':>10}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10}")
    for B in batch_sizes:
        r = results.get(B)
        if r:
            print(f"  B={B:<5} {r['mean_ms']:>12.3f}ms "
                  f"{r['throughput']:>12.0f}/s {r['runtime']:>10}")
        else:
            print(f"  B={B:<5} {'FAIL':>14}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
