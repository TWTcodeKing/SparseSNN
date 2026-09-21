#!/usr/bin/env python3
"""Benchmark sengine vs TensorRT.

Usage:
    # Build from ONNX, benchmark, and save .sengine
    python -m sengine.scripts.bench --onnx model_plugin.onnx --T 4 --batch 1

    # Benchmark a pre-built .sengine file
    python -m sengine.scripts.bench --sengine model.sengine

    # Compare against TensorRT engine
    python -m sengine.scripts.bench --sengine model.sengine --trt model.engine --trt-input-shape 4,3,224,224

    # Build + compare in one shot
    python -m sengine.scripts.bench --onnx model_plugin.onnx --T 4 --batch 1 \
        --trt model.engine --trt-input-shape 64,3,224,224

    # Export execution schedule
    python -m sengine.scripts.bench --sengine model.sengine --export-schedule schedule.md
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')


def bench_sengine(args) -> float:
    """Build or load sengine, benchmark, return latency in ms."""
    import sengine

    if args.sengine:
        print(f"\n{'='*60}")
        print(f"  Loading sengine: {args.sengine}")
        print(f"{'='*60}")
        t0 = time.time()
        engine = sengine.load(args.sengine)
        print(f"  Loaded in {time.time()-t0:.1f}s")
    elif args.onnx:
        print(f"\n{'='*60}")
        print(f"  Building sengine from ONNX: {args.onnx}")
        print(f"  T={args.T}, batch_size={args.batch}")
        print(f"{'='*60}")
        t0 = time.time()
        engine = sengine.build(args.onnx, T=args.T, batch_size=args.batch,
                               autotune=args.autotune,
                               fusion=args.fusion,
                               fusion_rec=getattr(args, 'fusion_rec', None),
                               precision=getattr(args, 'precision', 'fp16'))
        build_time = time.time() - t0
        cpp_mode = "C++ CUDA Graph" if not engine._use_python_runtime else "Python"
        print(f"  Built in {build_time:.1f}s ({cpp_mode})")

        # Save if requested
        if args.save:
            engine.save(args.save)
            size_mb = os.path.getsize(args.save) / 1e6
            print(f"  Saved: {args.save} ({size_mb:.1f} MB)")
    else:
        print("Error: provide --onnx or --sengine")
        return 0.0

    # Benchmark
    ms = engine.benchmark(warmup=args.warmup, iters=args.iters)
    fps = 1000.0 / ms * args.batch if ms > 0 else 0
    print(f"\n  sengine: {ms:.3f} ms  ({fps:.0f} img/s)")
    return ms


def bench_trt(args) -> float:
    """Benchmark TensorRT engine, return latency in ms."""
    if not args.trt:
        return 0.0

    print(f"\n{'='*60}")
    print(f"  Benchmarking TRT: {args.trt}")
    print(f"{'='*60}")

    try:
        from iengine.backends.tensorrt.runtime import TRTRunner
    except ImportError:
        print("  TRT runtime not available (missing tensorrt package)")
        return 0.0

    # Parse input shape
    if args.trt_input_shape:
        shape = tuple(int(x) for x in args.trt_input_shape.split(','))
    else:
        # Default: (TB, 3, 224, 224)
        TB = args.T * args.batch
        shape = (TB, 3, 224, 224)

    try:
        with TRTRunner(args.trt, device=0) as runner:
            result = runner.benchmark_latency(
                input_shape=shape,
                n_warmup=args.warmup,
                n_measure=args.iters,
            )
            ms = result['mean_ms']
            fps = 1000.0 / ms * args.batch if ms > 0 else 0
            print(f"  TRT input: {shape}")
            print(f"  TRT: {ms:.3f} ms  ({fps:.0f} img/s)")
            return ms
    except Exception as e:
        print(f"  TRT failed: {e}")
        return 0.0


def compare(sengine_ms: float, trt_ms: float, batch: int):
    """Print comparison table."""
    if sengine_ms <= 0:
        return

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  {'Engine':<25} {'Latency':>10} {'Throughput':>12} {'vs TRT':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
    sfps = 1000.0 / sengine_ms * batch
    print(f"  {'sengine':<25} {sengine_ms:>8.3f}ms {sfps:>10.0f}/s {'':>10}")

    if trt_ms > 0:
        tfps = 1000.0 / trt_ms * batch
        ratio = trt_ms / sengine_ms
        if ratio >= 1:
            verdict = f"{ratio:.1f}x faster"
        else:
            verdict = f"{1/ratio:.1f}x slower"
        print(f"  {'TensorRT':<25} {trt_ms:>8.3f}ms {tfps:>10.0f}/s {'baseline':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
        print(f"  sengine vs TRT: {verdict}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sengine vs TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources
    src = parser.add_argument_group("Input (provide one)")
    src.add_argument('--onnx', type=str, help='Plugin-mode ONNX file (build from scratch)')
    src.add_argument('--sengine', type=str, help='Pre-built .sengine file (fast load)')

    # Build options
    build = parser.add_argument_group("Build options (with --onnx)")
    build.add_argument('--T', type=int, default=4, help='Temporal steps (default: 4)')
    build.add_argument('--batch', type=int, default=1, help='Batch size (default: 1)')
    build.add_argument('--autotune', action='store_true', help='Run autotuning sweep')
    build.add_argument('--fusion', type=str, default='none', choices=['none', 'slicer'],
                       help='Fusion strategy (default: none)')
    build.add_argument('--fusion-rec', type=str, default=None,
                       help='Path to fusion recommendation JSON from validator pre-pass')
    build.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'],
                       help='Precision (default: fp16)')
    build.add_argument('--save', type=str, help='Save built engine to .sengine file')

    # TRT comparison
    trt = parser.add_argument_group("TensorRT comparison")
    trt.add_argument('--trt', type=str, help='TensorRT .engine file for comparison')
    trt.add_argument('--trt-input-shape', type=str,
                     help='TRT input shape as comma-separated ints (e.g., 64,3,224,224)')

    # Benchmark options
    bench = parser.add_argument_group("Benchmark options")
    bench.add_argument('--warmup', type=int, default=200, help='Warmup iterations (default: 200)')
    bench.add_argument('--iters', type=int, default=1000, help='Measurement iterations (default: 1000)')

    # Export
    export = parser.add_argument_group("Export")
    export.add_argument('--export-schedule', type=str,
                        help='Export BA-MTTS execution schedule to .md file')

    args = parser.parse_args()

    if not args.onnx and not args.sengine:
        parser.error("Provide --onnx or --sengine")

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sengine_ms = bench_sengine(args)

    # If loading from .sengine and schedule export requested
    if args.sengine and args.export_schedule:
        from sengine.build.sengine_io import load_sengine
        from sengine.build.schedule_builder import export_schedule_md
        ir, schedule, T, B = load_sengine(args.sengine)
        model_name = os.path.splitext(os.path.basename(args.sengine))[0]
        export_schedule_md(ir, schedule, args.export_schedule, model_name=model_name)
        print(f"  Schedule: {args.export_schedule}")

    trt_ms = bench_trt(args)
    compare(sengine_ms, trt_ms, args.batch)


if __name__ == '__main__':
    main()
