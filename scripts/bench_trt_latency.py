#!/usr/bin/env python3
"""Benchmark TensorRT inference latency for SNN models.

Exports ONNX (standard for ResNets, TDL-native for Transformers),
builds TRT engines, and measures latency across batch sizes.

Usage:
    # ResNet
    python scripts/bench_trt_latency.py --model sew_resnet18 --dataset cifar100

    # Transformer
    python scripts/bench_trt_latency.py --config configs/spikformer/spikformer_8_384.yaml \
        --dataset cifar100

    # With sparse checkpoint + custom batch sizes
    python scripts/bench_trt_latency.py --model sew_resnet34 --dataset cifar100 \
        --checkpoint obc_pt/sew_resnet34_cifar100_sbc_2_4_global.pth --sparse \
        --batch-sizes 1,4,8,16,32
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from models.neurons import reset_net
from tengine.utils import (
    build_model, build_model_from_config, load_model_config,
    get_dataset_config,
)
from iengine.backends.tensorrt.export import export_onnx
from iengine.backends.tensorrt.builder import build_engine
from iengine.backends.tensorrt.runtime import TRTRunner


def build_snn_model(args, ds_cfg, img_size):
    """Build SNN model from --model or --config."""
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
    parser.add_argument('--sparse', action='store_true',
                        help='Enable TRT SPARSE_WEIGHTS flag (2:4 sparse)')

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

    if not args.model and not args.config:
        parser.error("Provide --model or --config")

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']
    is_transformer = args.config is not None

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build model
    model, tag = build_snn_model(args, ds_cfg, img_size)
    os.makedirs(args.engine_dir, exist_ok=True)

    sparse_tag = '_sparse' if args.sparse else ''
    results = {}

    for B in batch_sizes:
        onnx_path = os.path.join(args.engine_dir, f"{tag}_b{B}{sparse_tag}.onnx")
        engine_path = os.path.join(args.engine_dir, f"{tag}_b{B}{sparse_tag}.engine")
        input_shape = (B, in_channels, img_size, img_size)

        # Step 1: Export ONNX
        if not os.path.exists(onnx_path):
            print(f"\n  [B={B}] Exporting ONNX "
                  f"({'TDL native' if is_transformer else 'direct'})...")
            reset_net(model)
            if is_transformer:
                from sengine.tdl.transforms import export_with_fused_neurons
                export_with_fused_neurons(
                    model, onnx_path, input_shape=input_shape,
                    opset=17, dynamic_batch=False, verbose=False)
            else:
                export_onnx(model, onnx_path, input_shape=input_shape,
                            dynamic_batch=False, simplify=True, verbose=False)
            print(f"          Saved: {onnx_path}")
        else:
            print(f"\n  [B={B}] ONNX exists: {onnx_path}")

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

    # Cleanup model
    del model
    torch.cuda.empty_cache()

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
