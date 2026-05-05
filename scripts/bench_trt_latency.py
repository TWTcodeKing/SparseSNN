#!/usr/bin/env python3
"""Benchmark TensorRT inference latency for SNN models.

Exports standard ONNX via torch.onnx.export (no TDL, no custom ops),
builds TRT engines, and measures latency across batch sizes.

Usage:
    # ResNet (full pipeline: export + build + benchmark)
    python scripts/bench_trt_latency.py --model sew_resnet18 --dataset cifar100

    # Transformer
    python scripts/bench_trt_latency.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar100

    # With sparse checkpoint + custom batch sizes
    python scripts/bench_trt_latency.py --model sew_resnet34 --dataset cifar100 \
        --checkpoint obc_pt/sew_resnet34_cifar100_sbc_2_4_global.pth --sparse \
        --batch-sizes 1,4,8,16,32

    # Export ONNX only (for cross-platform: export on x86, benchmark on Jetson)
    python scripts/bench_trt_latency.py --model sew_resnet34 --dataset imagenet \
        --export-only --batch-sizes 1,2,4,8

    # Jetson/ARM: skip export, use pre-exported ONNX
    python scripts/bench_trt_latency.py --onnx trt_engines/sew_resnet34_b1.onnx \
        --input-shape 1,3,224,224 --batch-sizes 1,2,4,8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from iengine.backends.tensorrt.builder import build_engine
from iengine.backends.tensorrt.runtime import TRTRunner


def build_snn_model(args, ds_cfg, img_size):
    """Build SNN model from --model or --config."""
    from tengine.utils import (
        build_model, build_model_from_config, load_model_config,
    )
    device = torch.device('cuda:0')
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        config['img_size'] = img_size
        model = build_model_from_config(config)
        tag = os.path.splitext(os.path.basename(args.config))[0]
    elif args.model:
        model = build_model(args.model, T=args.T,
                            num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'])
        tag = args.model
    else:
        raise ValueError("Provide --model or --config")

    model = model.to(device).eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    return model, tag


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT inference latency for SNN models")

    # Model (for ONNX export — not needed if --onnx is provided)
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')
    parser.add_argument('--config', type=str, default=None,
                        help='Transformer YAML config path')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (cifar100, imagenet, etc.)')
    parser.add_argument('--T', type=int, default=4,
                        help='Temporal steps (default: 4)')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (auto-detected from dataset if omitted)')

    # Pre-exported ONNX (skip export — for Jetson/ARM deployment)
    parser.add_argument('--onnx', type=str, default=None,
                        help='Pre-exported ONNX file (skips model build + export). '
                             'Use on Jetson/ARM where torch.onnx.export crashes.')
    parser.add_argument('--input-shape', type=str, default=None,
                        help='Input shape as comma-separated ints (e.g. 1,3,224,224). '
                             'Required with --onnx.')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (random weights if omitted)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable TRT SPARSE_WEIGHTS flag (2:4 sparse)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip onnxsim (for ARM/Jetson compatibility)')
    parser.add_argument('--export-only', action='store_true',
                        help='Export ONNX only, skip engine build and benchmark')
    parser.add_argument('--single-file', action='store_true',
                        help='Consolidate ONNX external data into a single .onnx file')

    # Benchmark
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16',
                        help='Comma-separated batch sizes (default: 1,4,8,16)')
    parser.add_argument('--warmup', type=int, default=200,
                        help='Warmup iterations (default: 200)')
    parser.add_argument('--iters', type=int, default=1000,
                        help='Measurement iterations (default: 1000)')

    # Output
    parser.add_argument('--engine-dir', type=str, default='trt_engines',
                        help='Directory for ONNX and engine files')

    args = parser.parse_args()

    # Validate args
    if args.onnx:
        # ONNX-only mode: skip export, just build + benchmark
        if not os.path.exists(args.onnx):
            parser.error(f"ONNX file not found: {args.onnx}")
        if not args.input_shape:
            parser.error("--input-shape required with --onnx "
                         "(e.g. --input-shape 1,3,224,224)")
    else:
        # Full pipeline mode: need model + dataset for export
        if not args.model and not args.config:
            parser.error("Provide --model/--config (for export) or --onnx (pre-exported)")
        if not args.dataset:
            parser.error("--dataset required for ONNX export")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.engine_dir, exist_ok=True)

    # ─── Mode A: Pre-exported ONNX (Jetson/ARM) ───
    if args.onnx:
        base_shape = tuple(int(x) for x in args.input_shape.split(','))
        in_channels = base_shape[1]
        img_size = base_shape[2]
        tag = os.path.splitext(os.path.basename(args.onnx))[0]
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

        results = {}
        for B in batch_sizes:
            input_shape = (B, in_channels, img_size, img_size)
            engine_path = os.path.join(args.engine_dir,
                                       f"{tag}_b{B}.engine")

            # Build TRT engine (re-export ONNX per batch not needed for
            # dynamic batch engines, but we build fixed-batch for best perf)
            if not os.path.exists(engine_path):
                print(f"\n  [B={B}] Building TRT engine (dense FP16)...")
                try:
                    build_engine(args.onnx, engine_path,
                                 sparse=args.sparse, fp16=True,
                                 min_batch=B, opt_batch=B, max_batch=B,
                                 workspace_gb=4.0, verbose=False)
                    print(f"          Saved: {engine_path}")
                except Exception as e:
                    print(f"          Build FAILED: {e}")
                    continue
            else:
                print(f"\n  [B={B}] Engine exists: {engine_path}")

            # Benchmark
            try:
                with TRTRunner(engine_path, device=0) as runner:
                    result = runner.benchmark_latency(
                        input_shape=input_shape,
                        n_warmup=args.warmup,
                        n_measure=args.iters,
                    )
                    results[B] = result
                    print(f"  [B={B}] Latency: {result['mean_ms']:.3f} ms "
                          f"({result['throughput_img_s']:.0f} img/s)")
            except Exception as e:
                print(f"  [B={B}] Benchmark FAILED: {e}")

        # Results table
        mode = "Sparse 2:4" if args.sparse else "Dense"
        print(f"\n{'='*70}")
        print(f"  TensorRT {mode} FP16 | {tag}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*70}")
        print(f"  {'Batch':<8} {'Latency (ms)':>14} {'Std (ms)':>10} {'Throughput':>14}")
        print(f"  {'-'*8} {'-'*14} {'-'*10} {'-'*14}")
        for B in batch_sizes:
            r = results.get(B)
            if r:
                print(f"  B={B:<5} {r['mean_ms']:>12.3f}ms {r['std_ms']:>8.3f}ms "
                      f"{r['throughput_img_s']:>12.0f}/s")
            else:
                print(f"  B={B:<5} {'FAIL':>14}")
        print(f"{'='*70}")
        return

    # ─── Mode B: Full pipeline (x86 — export + build + benchmark) ───
    from models.neurons import reset_net
    from tengine.utils import get_dataset_config
    from iengine.backends.tensorrt.export import export_onnx

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    model, tag = build_snn_model(args, ds_cfg, img_size)

    sparse_tag = '_sparse' if args.sparse else ''
    results = {}

    for B in batch_sizes:
        onnx_path = os.path.join(args.engine_dir, f"{tag}_b{B}{sparse_tag}.onnx")
        engine_path = os.path.join(args.engine_dir, f"{tag}_b{B}{sparse_tag}.engine")
        input_shape = (B, in_channels, img_size, img_size)

        # Step 1: Export standard ONNX (no TDL, no custom ops)
        if not os.path.exists(onnx_path):
            print(f"\n  [B={B}] Exporting ONNX (standard torch.onnx.export)...")
            reset_net(model)
            export_onnx(model, onnx_path, input_shape=input_shape,
                        dynamic_batch=False, simplify=not args.no_simplify,
                        verbose=False)
            # Consolidate external data into single file
            if args.single_file:
                import onnx
                from onnx.external_data_helper import convert_model_to_external_data
                onnx_model = onnx.load(onnx_path)
                onnx.save(onnx_model, onnx_path)
                # Remove leftover external data files
                onnx_dir = os.path.dirname(onnx_path) or '.'
                for f in os.listdir(onnx_dir):
                    fpath = os.path.join(onnx_dir, f)
                    if (os.path.isfile(fpath) and not f.endswith(('.onnx', '.engine'))
                            and not f.startswith('.')):
                        # Only remove files that look like ONNX external data
                        # (no extension, or named like weight tensors)
                        _, ext = os.path.splitext(f)
                        if not ext:
                            os.remove(fpath)
                print(f"          Saved (single file): {onnx_path}")
            else:
                print(f"          Saved: {onnx_path}")
        else:
            print(f"\n  [B={B}] ONNX exists: {onnx_path}")

        if args.export_only:
            continue

        # Step 2: Build TRT engine
        if not os.path.exists(engine_path):
            print(f"  [B={B}] Building TRT engine "
                  f"({'sparse 2:4' if args.sparse else 'dense'} FP16)...")
            try:
                build_engine(onnx_path, engine_path,
                             sparse=args.sparse, fp16=True,
                             min_batch=B, opt_batch=B, max_batch=B,
                             workspace_gb=4.0, verbose=False)
                print(f"          Saved: {engine_path}")
            except Exception as e:
                print(f"          Build FAILED: {e}")
                continue
        else:
            print(f"  [B={B}] Engine exists: {engine_path}")

        # Step 3: Benchmark latency
        try:
            with TRTRunner(engine_path, device=0) as runner:
                result = runner.benchmark_latency(
                    input_shape=input_shape,
                    n_warmup=args.warmup,
                    n_measure=args.iters,
                )
                results[B] = result
                print(f"  [B={B}] Latency: {result['mean_ms']:.3f} ms "
                      f"({result['throughput_img_s']:.0f} img/s)")
        except Exception as e:
            print(f"  [B={B}] Benchmark FAILED: {e}")

    del model
    torch.cuda.empty_cache()

    if args.export_only:
        print(f"\n  Export complete. ONNX files in: {args.engine_dir}/")
        return

    # Results table
    mode = "Sparse 2:4" if args.sparse else "Dense"
    print(f"\n{'='*70}")
    print(f"  TensorRT {mode} FP16 | {tag} | {args.dataset} | T={args.T}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    print(f"  {'Batch':<8} {'Latency (ms)':>14} {'Std (ms)':>10} {'Throughput':>14}")
    print(f"  {'-'*8} {'-'*14} {'-'*10} {'-'*14}")
    for B in batch_sizes:
        r = results.get(B)
        if r:
            print(f"  B={B:<5} {r['mean_ms']:>12.3f}ms {r['std_ms']:>8.3f}ms "
                  f"{r['throughput_img_s']:>12.0f}/s")
        else:
            print(f"  B={B:<5} {'FAIL':>14}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
