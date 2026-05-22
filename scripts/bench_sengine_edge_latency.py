#!/usr/bin/env python3
"""Benchmark sengine_edge inference latency for SNN models on Jetson AGX Orin.

Three-phase pipeline:
  Phase 1: Export plugin-mode ONNX (TDL transforms)
  Phase 2: Fusion validation pre-pass (profile fused vs decomposed per shape)
  Phase 3: Build + benchmark (with validated fusion + roofline tuning)

Usage:
    # Full pipeline with validation + autotuning
    python scripts/bench_sengine_edge_latency.py \
        --config configs/maxformer/maxformer_10_512.yaml \
        --dataset imagenet --T 4 --batch-sizes 4 \
        --fusion slicer --autotune --gpu-ids 0

    # Compare decomposed vs validated fusion
    python scripts/bench_sengine_edge_latency.py \
        --model sew_resnet18 --dataset imagenet \
        --fusion none,slicer --autotune --batch-sizes 4

    # Quick (no autotuning, no validation)
    python scripts/bench_sengine_edge_latency.py \
        --model sew_resnet18 --dataset imagenet --batch-sizes 1

    # Export only (export on x86, build on Orin)
    python scripts/bench_sengine_edge_latency.py \
        --model sew_resnet18 --dataset imagenet --T 4 --export-only
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Auto-detect CUDA home (JetPack default, then common paths)
for cuda_path in ['/usr/local/cuda', '/usr/local/cuda-12.8', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch


def export_plugin_onnx(args, ds_cfg, img_size, output_dir):
    """Export plugin-mode ONNX for sengine_edge."""
    from sengine_edge.scripts.export_onnx import export_model
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


def print_export_summary(onnx_path, model_tag, args):
    """Print TDL export summary with sengine_edge build specs for Orin deploy."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    file_mb = os.path.getsize(onnx_path) / 1e6

    # Input / output shapes
    inputs = []
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        inputs.append((inp.name, shape))
    outputs = []
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        outputs.append((out.name, shape))

    # Count ops by domain (standard vs custom SNN ops)
    op_counts = {}
    snn_ops = {}
    for node in model.graph.node:
        domain = node.domain or "onnx"
        op = node.op_type
        if domain != "onnx":
            snn_ops[op] = snn_ops.get(op, 0) + 1
        else:
            op_counts[op] = op_counts.get(op, 0) + 1

    n_total = len(model.graph.node)
    n_weights = len(model.graph.initializer)

    # Derive specs from input shape
    in_shape = inputs[0][1] if inputs else []
    in_channels = in_shape[1] if len(in_shape) == 4 else '?'
    img_h = in_shape[2] if len(in_shape) == 4 else '?'
    img_w = in_shape[3] if len(in_shape) == 4 else '?'

    print(f"\n{'='*70}")
    print(f"  TDL Plugin ONNX Export Summary")
    print(f"{'='*70}")
    print(f"  File       : {onnx_path}")
    print(f"  Size       : {file_mb:.1f} MB")
    print(f"  Model      : {model_tag}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Input      : ({in_channels}, {img_h}, {img_w})  [C, H, W per image]")
    print(f"  T          : {args.T}")
    print(f"  Nodes      : {n_total}  ({n_weights} initializers)")

    if snn_ops:
        snn_str = ", ".join(f"{k}={v}" for k, v in sorted(snn_ops.items()))
        print(f"  SNN ops    : {snn_str}")

    top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:8]
    ops_str = ", ".join(f"{k}={v}" for k, v in top_ops)
    print(f"  ONNX ops   : {ops_str}")

    # Build command for Orin
    prec = args.precision
    print(f"\n{'='*70}")
    print(f"  SEngine-Edge Build Command (run on Jetson AGX Orin)")
    print(f"{'='*70}")
    print(f"  python -c \"")
    print(f"  import sengine_edge")
    print(f"  e = sengine_edge.build('{os.path.basename(onnx_path)}',")
    print(f"                         T={args.T}, batch_size=1,")
    print(f"                         fusion='slicer', autotune=True,")
    print(f"                         precision='{prec}')")
    print(f"  e.save('{model_tag}.sengine')")
    print(f"  ms = e.benchmark()")
    print(f"  print(f'Latency: {{ms:.3f}} ms')\"")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sengine_edge inference latency (Jetson AGX Orin)")
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
    parser.add_argument('--batch-sizes', type=str, default='1',
                        help='Comma-separated batch sizes (default: 1 for edge)')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup iterations (default: 100, lower for Orin)')
    parser.add_argument('--iters', type=int, default=500,
                        help='Benchmark iterations (default: 500, lower for Orin)')
    parser.add_argument('--export-dir', type=str, default='sengine_edge/exports')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp16', 'fp32'], help='Global precision (fp16 or fp32)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--power-mode', type=str, default=None,
                        help='Orin power mode hint (e.g., MAXN, 30W, 15W) — logged only')
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
    if args.power_mode:
        print(f"Power mode: {args.power_mode}")

    os.makedirs(args.export_dir, exist_ok=True)

    # ── Phase 1: Export plugin ONNX ──
    print(f"\n{'='*70}")
    print(f"  Phase 1: Export plugin-mode ONNX")
    print(f"{'='*70}")
    plugin_onnx = export_plugin_onnx(args, ds_cfg, img_size, args.export_dir)

    if args.export_only:
        print_export_summary(plugin_onnx, tag, args)
        return

    # ── Phase 2: Fusion validation (per batch size) ──
    # For each (fusion=slicer, batch_size), run the validator pre-pass
    # to determine which shapes benefit from fusion.
    rec_files = {}  # (fusion, B) → rec_path or None
    if 'slicer' in fusion_modes and args.autotune:
        print(f"\n{'='*70}")
        print(f"  Phase 2: Fusion validation pre-pass")
        print(f"{'='*70}")

        from sengine_edge.build.fusion_validator import generate_recommendations, load_tuning_configs

        for B in batch_sizes:
            rec_path = os.path.join('.cache', f'fusion_rec_{tag}_{args.dataset}_T{args.T}_B{B}.json')
            if os.path.exists(rec_path):
                # Check if rec.json has embedded tuning configs (new format)
                cfgs, _, _, _, _ = load_tuning_configs(rec_path)
                if cfgs:
                    print(f"\n  [B={B}] Recommendations exist ({len(cfgs)} cached configs): {rec_path}")
                    rec_files[('slicer', B)] = rec_path
                    continue
                else:
                    # Old format without configs — regenerate
                    print(f"\n  [B={B}] Regenerating (old format without cached configs)...")
            else:
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
                                      '.cache', f'sengine_edge_B{B}{prec_suffix}')
            result_file = f'/tmp/sengine_edge_bench_{tag}_{fusion}_B{B}_{args.precision}.json'

            sub_cmd = f"""
import os, sys, time, json; sys.path.insert(0, '{os.path.dirname(os.path.dirname(__file__))}')
os.environ['CUDA_HOME']='{os.environ.get("CUDA_HOME","")}'
os.environ['PATH']='{os.environ.get("PATH","")}'
import sengine_edge
from sengine_edge.ir import KernelVariant
try:
    # Check how many configs are pre-loaded from rec.json
    n_preloaded = 0
    rec_path = {repr(rec_path)}
    if rec_path:
        from sengine_edge.build.fusion_validator import load_tuning_configs
        cfgs, _, _, _, _ = load_tuning_configs(rec_path)
        n_preloaded = len(cfgs)
    t0 = time.time()
    e = sengine_edge.build('{plugin_onnx}', T={args.T}, batch_size={B},
                            fusion='{fusion}', autotune={args.autotune},
                            fusion_rec=rec_path,
                            precision='{args.precision}')
    build_s = time.time() - t0
    mode = 'C++' if not e._use_python_runtime else 'Python'
    n_fused = sum(1 for n in e._ir.nodes.values()
                  if n.assigned_kernel in (KernelVariant.TileLangFusedConv1x1BNIF,
                                            KernelVariant.TileLangFusedConvBNIF))
    ms = e.benchmark(warmup={args.warmup}, iters={args.iters})
    fps = 1000.0 / ms * {B} if ms > 0 else 0
    json.dump({{'ms': ms, 'fps': fps, 'mode': mode, 'fused': n_fused,
                'build_s': build_s, 'validated': rec_path is not None,
                'preloaded': n_preloaded,
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
                env=env, timeout=3600,
                capture_output=True, text=True)

            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json_mod.load(f)
                if 'error' in r:
                    print(f"  [{label}] FAILED: {r['error']}")
                    all_results[(fusion, B)] = None
                else:
                    all_results[(fusion, B)] = r
                    extra = ""
                    if r.get('validated'):
                        extra += " | validated"
                    if r.get('preloaded', 0) > 0:
                        extra += f" | {r['preloaded']} cached configs"
                    print(f"  [{label}] {r['ms']:.3f} ms | {r['fps']:.0f} img/s | "
                          f"fused={r['fused']} | {r['mode']} | build={r['build_s']:.0f}s"
                          + extra)
                os.remove(result_file)
            else:
                print(f"  [{label}] FAILED: subprocess crashed")
                if proc.stderr:
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
    if args.power_mode:
        print(f"  Power mode: {args.power_mode}")
    print(f"{'='*80}")
    print(f"  {'Fusion':<10} {'Batch':<6} {'Latency':>10} {'Throughput':>12} "
          f"{'Fused':>6} {'Runtime':>8} {'Build':>8} {'Cached':>7}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")
    for fusion in fusion_modes:
        for B in batch_sizes:
            r = all_results.get((fusion, B))
            if r:
                cached = str(r.get('preloaded', 0)) if r.get('preloaded', 0) > 0 else '-'
                print(f"  {fusion:<10} B={B:<4} {r['ms']:>8.3f}ms "
                      f"{r['fps']:>10.0f}/s {r['fused']:>6} "
                      f"{r['mode']:>8} {r['build_s']:>6.0f}s {cached:>7}")
            else:
                print(f"  {fusion:<10} B={B:<4} {'FAIL':>10}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
