#!/usr/bin/env python3
"""Benchmark sengine inference latency for SNN models.

Three-phase pipeline:
  Phase 1: Export plugin-mode ONNX (TDL transforms)
  Phase 2: Fusion validation pre-pass (profile fused vs decomposed per shape)
  Phase 3: Build + benchmark (with validated fusion + roofline tuning)

Usage:
    # Full pipeline with validation + autotuning
    python scripts/bench_sengine_latency.py \
        --config configs/maxformer/maxformer_10_512.yaml \
        --dataset imagenet --T 4 --batch-sizes 4 \
        --fusion slicer --autotune --gpu-ids 2

    # Compare decomposed vs validated fusion
    python scripts/bench_sengine_latency.py \
        --model sew_resnet18 --dataset imagenet \
        --fusion none,slicer --autotune --batch-sizes 4

    # Quick (no autotuning, no validation)
    python scripts/bench_sengine_latency.py \
        --model sew_resnet18 --dataset imagenet --batch-sizes 4
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Auto-detect CUDA home
for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch


def export_plugin_onnx(args, ds_cfg, img_size, output_dir):
    """Export plugin-mode ONNX for sengine."""
    from sengine.scripts.export_onnx import export_model
    model_name = args.model or os.path.splitext(os.path.basename(args.config))[0]
    plugin_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_plugin.onnx")
    if os.path.exists(plugin_path):
        print(f"  Plugin ONNX exists: {plugin_path}")
        return plugin_path
    print(f"  Exporting plugin ONNX: {model_name}...")
    result = export_model(
        model_name=args.model or model_name, output_dir=output_dir,
        T=args.T, dataset=args.dataset, img_size=img_size,
        config=args.config, checkpoint=args.checkpoint)
    if result and os.path.exists(result):
        print(f"  Saved: {result}")
    else:
        raise RuntimeError(f"ONNX export failed for {model_name}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark sengine inference latency")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fusion', type=str, default='none',
                        help='none, slicer, or none,slicer for comparison')
    parser.add_argument('--autotune', action='store_true',
                        help='Enable roofline-guided autotuning')
    parser.add_argument('--batch-sizes', type=str, default='1,4')
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--export-dir', type=str, default='sengine/exports')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp16', 'fp32'], help='Global precision (fp16 or fp32)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    args = parser.parse_args()

    if not args.model and not args.config:
        parser.error("Provide --model or --config")

    device_id = int(args.gpu_ids.split(',')[0])
    torch.cuda.set_device(device_id)

    from tengine.utils import get_dataset_config
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    fusion_modes = [s.strip() for s in args.fusion.split(',')]
    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']
    tag = args.model or os.path.splitext(os.path.basename(args.config))[0]

    props = torch.cuda.get_device_properties(device_id)
    gpu_name = props.name
    gpu_arch = f"sm_{props.major}{props.minor}"
    gpu_sms = props.multi_processor_count
    gpu_mem = props.total_memory / 1e9
    print(f"GPU {device_id}: {gpu_name} ({gpu_arch}, {gpu_sms} SMs, {gpu_mem:.1f} GB)")
    print(f"Model: {tag} | Dataset: {args.dataset} | T={args.T}")
    print(f"Fusion: {fusion_modes} | Autotune: {args.autotune} | Precision: {args.precision}")

    os.makedirs(args.export_dir, exist_ok=True)

    # ── Phase 1: Export plugin ONNX ──
    print(f"\n{'='*70}")
    print(f"  Phase 1: Export plugin-mode ONNX")
    print(f"{'='*70}")
    plugin_onnx = export_plugin_onnx(args, ds_cfg, img_size, args.export_dir)

    if args.export_only:
        print(f"\n  Export complete: {plugin_onnx}")
        return

    # ── Phase 2: Fusion validation (per batch size) ──
    # For each (fusion=slicer, batch_size), run the validator pre-pass
    # to determine which shapes benefit from fusion.
    rec_files = {}  # (fusion, B) → rec_path or None
    if 'slicer' in fusion_modes and args.autotune:
        print(f"\n{'='*70}")
        print(f"  Phase 2: Fusion validation pre-pass")
        print(f"{'='*70}")

        from sengine.build.fusion_validator import generate_recommendations

        for B in batch_sizes:
            rec_path = os.path.join('.cache', f'fusion_rec_{tag}_{args.dataset}_T{args.T}_B{B}.json')
            if os.path.exists(rec_path):
                print(f"\n  [B={B}] Recommendations exist: {rec_path}")
                rec_files[('slicer', B)] = rec_path
                continue
            print(f"\n  [B={B}] Validating fused shapes...")
            t0 = time.time()
            generate_recommendations(plugin_onnx, args.T, B, rec_path, precision=args.precision)
            print(f"  Done in {time.time()-t0:.0f}s")
            rec_files[('slicer', B)] = rec_path
    else:
        print(f"\n  Phase 2: Skipped (no slicer+autotune)")

    # ── Phase 3: Build + benchmark ──
    print(f"\n{'='*70}")
    print(f"  Phase 3: Build + benchmark")
    print(f"{'='*70}")

    all_results = {}

    for fusion in fusion_modes:
        for B in batch_sizes:
            label = f"fusion={fusion}, B={B}"
            rec_path = rec_files.get((fusion, B))

            # Run each build in a SUBPROCESS to guarantee clean CUDA state.
            # The validator pre-pass and previous builds may leave stale
            # CUDA errors that corrupt graph capture in the same process.
            import subprocess, json as json_mod
            prec_suffix = f'_{args.precision}' if args.precision != 'fp16' else ''
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      '.cache', f'sengine_B{B}{prec_suffix}')
            result_file = f'/tmp/sengine_bench_{tag}_{fusion}_B{B}_{args.precision}.json'

            sub_cmd = f"""
import os, sys, time, json; sys.path.insert(0, '{os.path.dirname(os.path.dirname(__file__))}')
os.environ['CUDA_HOME']='{os.environ.get("CUDA_HOME","")}'
os.environ['PATH']='{os.environ.get("PATH","")}'
import shutil
cache = '{cache_dir}'
if os.path.exists(cache): shutil.rmtree(cache)
import sengine
from sengine.ir import KernelVariant
try:
    t0 = time.time()
    e = sengine.build('{plugin_onnx}', T={args.T}, batch_size={B},
                       fusion='{fusion}', autotune={args.autotune},
                       fusion_rec={repr(rec_path)},
                       precision='{args.precision}')
    build_s = time.time() - t0
    mode = 'C++' if not e._use_python_runtime else 'Python'
    n_fused = sum(1 for n in e._ir.nodes.values()
                  if n.assigned_kernel in (KernelVariant.TileLangFusedConv1x1BNIF,
                                            KernelVariant.TileLangFusedConvBNIF))
    ms = e.benchmark(warmup={args.warmup}, iters={args.iters})
    fps = 1000.0 / ms * {B} if ms > 0 else 0
    json.dump({{'ms': ms, 'fps': fps, 'mode': mode, 'fused': n_fused,
                'build_s': build_s, 'validated': {rec_path is not None},
                'precision': '{args.precision}'}},
              open('{result_file}', 'w'))
except Exception as ex:
    json.dump({{'error': str(ex)}}, open('{result_file}', 'w'))
"""
            print(f"\n  [{label}] Building (subprocess)...")
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(device_id)
            proc = subprocess.run(
                [sys.executable, '-c', sub_cmd],
                env=env, timeout=1800,
                capture_output=True, text=True)

            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json_mod.load(f)
                if 'error' in r:
                    print(f"  [{label}] FAILED: {r['error']}")
                    all_results[(fusion, B)] = None
                else:
                    all_results[(fusion, B)] = r
                    v = 'yes' if r.get('validated') else 'no'
                    print(f"  [{label}] {r['ms']:.3f} ms | {r['fps']:.0f} img/s | "
                          f"fused={r['fused']} | {r['mode']} | build={r['build_s']:.0f}s"
                          + (" | validated" if r.get('validated') else ""))
                os.remove(result_file)
            else:
                print(f"  [{label}] FAILED: subprocess crashed")
                if proc.stderr:
                    # Show last few lines of error
                    err_lines = proc.stderr.strip().split('\n')
                    for line in err_lines[-3:]:
                        if 'Error' in line or 'FAILED' in line:
                            print(f"    {line}")
                all_results[(fusion, B)] = None

    # ── Results table ──
    print(f"\n{'='*80}")
    print(f"  {tag} | {args.dataset} | T={args.T}")
    print(f"  GPU {device_id}: {gpu_name} ({gpu_arch}, {gpu_sms} SMs)")
    print(f"  Autotune: {args.autotune} | Precision: {args.precision}")
    print(f"{'='*80}")
    print(f"  {'Fusion':<10} {'Batch':<6} {'Latency':>10} {'Throughput':>12} "
          f"{'Fused':>6} {'Runtime':>8} {'Build':>8} {'Valid':>6}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for fusion in fusion_modes:
        for B in batch_sizes:
            r = all_results.get((fusion, B))
            if r:
                v = 'yes' if r.get('validated') else 'no'
                print(f"  {fusion:<10} B={B:<4} {r['ms']:>8.3f}ms "
                      f"{r['fps']:>10.0f}/s {r['fused']:>6} "
                      f"{r['mode']:>8} {r['build_s']:>6.0f}s {v:>6}")
            else:
                print(f"  {fusion:<10} B={B:<4} {'FAIL':>10}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
