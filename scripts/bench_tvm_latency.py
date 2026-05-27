#!/usr/bin/env python3
"""Benchmark TVM-compiled ANN equivalents of SNN models.

Since TVM (via torch.compile backend='tvm') does not support SNN-specific
ops (spiking neurons, temporal dimension), we benchmark equivalent ANN models
with ReLU activations and B*T batch size to match total compute.

The ANN model has identical architecture topology (same Conv/Linear/BN/Attention)
but replaces spiking neurons with ReLU and removes the temporal dimension.
To match the SNN's total compute, the ANN batch size is B*T.

Usage:
    # ResNet (--model)
    python scripts/bench_tvm_latency.py --model sew_resnet18 --dataset cifar100 \
        --T 4 --batch-sizes 1,4 --gpu-ids 6

    # Transformer (--config)
    python scripts/bench_tvm_latency.py --config configs/spikformer/spikformer_4_384.yaml \
        --dataset cifar100 --T 4 --batch-sizes 1,4 --gpu-ids 6

    # NLP (SpikeBERT)
    python scripts/bench_tvm_latency.py --config configs/spike_bert/spike_bert_small.yaml \
        --dataset sst2 --T 4 --batch-sizes 1 --gpu-ids 6

    # Compare TVM vs eager ANN
    python scripts/bench_tvm_latency.py --model sew_resnet_cifar56 --dataset cifar100 \
        --T 4 --batch-sizes 1,4,16 --compare-eager --gpu-ids 6
"""

import argparse
import os
import sys
import time

import numpy as np
np.float_ = np.float64
np.int_ = np.int64
np.complex_ = np.complex128
np.object_ = np.object_
np.bool_ = np.bool_
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tengine.utils import get_dataset_config
from models_ann import build_ann_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TVM-compiled ANN equivalents of SNN models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="ResNet/VGG model name")
    group.add_argument("--config", type=str, help="Transformer/NLP YAML config")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--T", type=int, default=4,
                        help="Original SNN temporal steps (ANN batch = B*T)")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-sizes", type=str, default="1,4",
                        help="SNN-equivalent batch sizes (ANN uses B*T)")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--gpu-ids", type=int, default=0)
    parser.add_argument("--mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--fp16", action="store_true",
                        help="Run in FP16 (default: FP32)")
    parser.add_argument("--compare-eager", action="store_true",
                        help="Also benchmark eager PyTorch for comparison")
    return parser.parse_args()


def benchmark_latency(model, input_tensor, warmup, iters):
    """Measure inference latency with CUDA synchronization."""
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.array(times)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_ids}")
    torch.cuda.set_device(device)

    ds_cfg = get_dataset_config(args.dataset)
    is_nlp = ds_cfg.get('task') == 'nlp'
    img_size = args.img_size or ds_cfg['img_size']
    if isinstance(img_size, (list, tuple)):
        img_h, img_w = img_size
    else:
        img_h = img_w = img_size

    # Build ANN model
    build_kwargs = dict(num_classes=ds_cfg['num_classes'],
                        in_channels=ds_cfg['in_channels'], T=args.T)
    if not is_nlp:
        build_kwargs['img_size'] = img_h

    if args.config:
        model = build_ann_model(config=args.config, **build_kwargs)
        model_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        model = build_ann_model(args.model, **build_kwargs)
        model_name = args.model

    model = model.to(device).eval()
    if args.fp16:
        model.half()

    # NLP: read seq_len from config
    seq_len = ds_cfg.get('seq_len', 128)
    if is_nlp and args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        seq_len = cfg.get('max_seq_len', seq_len)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    gpu_name = torch.cuda.get_device_name(device)

    print(f"GPU {args.gpu_ids}: {gpu_name}")
    print(f"Model: {model_name} (ANN equivalent, {n_params:.1f}M params)")
    print(f"Dataset: {args.dataset} | T={args.T} (ANN batch = B×T)")
    print(f"Compile: torch.compile(backend='tvm', mode='{args.mode}')")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")

    # Compile with TVM backend
    print(f"\nCompiling...")
    t_compile = time.time()
    try:
        compiled_model = torch.compile(model, mode=args.mode, backend='tvm')
        compile_ok = True
    except Exception as e:
        print(f"  torch.compile(backend='tvm') failed: {e}")
        print(f"  Falling back to eager PyTorch")
        compiled_model = model
        compile_ok = False
    compile_s = time.time() - t_compile

    # Header
    print()
    print("=" * 85)
    tag = "TVM" if compile_ok else "Eager (TVM failed)"
    print(f"  {tag} | {model_name} | {args.dataset} | T={args.T}")
    print(f"  GPU: {gpu_name} | dtype: {'FP16' if args.fp16 else 'FP32'}")
    print(f"  ANN batch = SNN_B × T = same total compute as SNN")
    print("=" * 85)
    print(f"  {'SNN_B':<7} {'ANN_B':<7} {'Latency (ms)':>14} {'Std (ms)':>10} "
          f"{'Throughput':>14}")
    print(f"  {'-'*7} {'-'*7} {'-'*14} {'-'*10} {'-'*14}")

    results = {}
    for snn_B in batch_sizes:
        ann_B = snn_B * args.T  # ANN batch = B × T

        try:
            if is_nlp:
                input_tensor = torch.randint(0, 1000, (ann_B, seq_len),
                                             device=device)
            else:
                dtype = torch.float16 if args.fp16 else torch.float32
                input_tensor = torch.randn(ann_B, ds_cfg['in_channels'],
                                           img_h, img_w, dtype=dtype,
                                           device=device)

            # Trigger compilation on first input
            with torch.no_grad():
                _ = compiled_model(input_tensor)
            torch.cuda.synchronize()

            times = benchmark_latency(compiled_model, input_tensor,
                                      args.warmup, args.iters)
            mean_ms = times.mean()
            std_ms = times.std()
            throughput = ann_B / (mean_ms / 1000)

            results[snn_B] = {
                'ann_B': ann_B, 'mean_ms': mean_ms, 'std_ms': std_ms,
                'throughput': throughput,
            }
            print(f"  B={snn_B:<5} B={ann_B:<5} {mean_ms:>12.3f}ms {std_ms:>8.3f}ms "
                  f"{throughput:>12.0f}/s")

            del input_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  B={snn_B:<5} B={ann_B:<5} {'FAILED':>14}: {e}")
            torch.cuda.empty_cache()

    # Eager baseline comparison
    if args.compare_eager:
        print()
        print(f"  --- Eager PyTorch (no compile) ---")
        print(f"  {'SNN_B':<7} {'ANN_B':<7} {'Eager (ms)':>14} {'Std (ms)':>10} "
              f"{'Throughput':>14} {'TVM speedup':>12}")
        print(f"  {'-'*7} {'-'*7} {'-'*14} {'-'*10} {'-'*14} {'-'*12}")

        for snn_B in batch_sizes:
            ann_B = snn_B * args.T
            try:
                if is_nlp:
                    input_tensor = torch.randint(0, 1000, (ann_B, seq_len),
                                                 device=device)
                else:
                    dtype = torch.float16 if args.fp16 else torch.float32
                    input_tensor = torch.randn(ann_B, ds_cfg['in_channels'],
                                               img_h, img_w, dtype=dtype,
                                               device=device)

                times = benchmark_latency(model, input_tensor,
                                          args.warmup, args.iters)
                eager_ms = times.mean()
                eager_std = times.std()
                eager_tp = ann_B / (eager_ms / 1000)

                tvm_r = results.get(snn_B)
                if tvm_r:
                    speedup = eager_ms / tvm_r['mean_ms']
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "-"

                print(f"  B={snn_B:<5} B={ann_B:<5} {eager_ms:>12.3f}ms "
                      f"{eager_std:>8.3f}ms {eager_tp:>12.0f}/s {speedup_str:>12}")

                del input_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  B={snn_B:<5} FAILED: {e}")

    print("=" * 85)
    if compile_ok:
        print(f"  Compile time: {compile_s:.1f}s")
    print()


if __name__ == "__main__":
    main()
