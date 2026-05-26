"""Benchmark SNN models using torch.compile (Inductor backend).

Measures inference latency of PyTorch models compiled with torch.compile()
using the Inductor backend (triton code generation). This provides a baseline
for comparing against sengine and TensorRT.

Usage:
    python scripts/bench_inductor_latency.py \
        --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4,8,16

    python scripts/bench_inductor_latency.py \
        --config configs/maxformer/maxformer_10_512.yaml --dataset imagenet \
        --T 4 --batch-sizes 1,4
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tengine.utils import (
    build_model, build_model_from_config, load_model_config, get_dataset_config,
)
from models.neurons import reset_net


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SNN with torch.compile (Inductor)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="ResNet model name")
    group.add_argument("--config", type=str, help="Transformer YAML config path")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--T", type=int, default=4, help="Temporal steps")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--gpu-ids", type=int, default=0)
    parser.add_argument("--mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--backend", type=str, default="tvm",
                        help="torch.compile backend (default: inductor)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip compilation, benchmark eager PyTorch")
    parser.add_argument("--fp16", action="store_true",
                        help="Run in FP16 precision (default: FP32)")
    return parser.parse_args()


def build_snn_model(args, ds_cfg, device):
    """Build SNN model from --model or --config."""
    img_size = args.img_size or ds_cfg['img_size']

    if args.model:
        model = build_model(
            args.model,
            T=args.T,
            num_classes=ds_cfg['num_classes'],
            in_channels=ds_cfg['in_channels'],
        )
    else:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        config['img_size'] = img_size
        model = build_model_from_config(config)

    model = model.to(device).eval()

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict, strict=False)

    return model, img_size


def benchmark_latency(model, input_tensor, warmup, iters):
    """Measure inference latency with CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        reset_net(model)
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(iters):
        reset_net(model)
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
    model, img_size = build_snn_model(args, ds_cfg, device)
    model_name = args.model or os.path.splitext(os.path.basename(args.config))[0]

    # NLP: read seq_len from config YAML if available, else from dataset config
    if is_nlp:
        seq_len = ds_cfg.get('seq_len', 128)
        if args.config:
            import yaml
            with open(args.config) as f:
                cfg_yaml = yaml.safe_load(f)
            seq_len = cfg_yaml.get('max_seq_len', seq_len)

    # Normalize img_size to (H, W)
    if isinstance(img_size, (list, tuple)):
        img_h, img_w = img_size
    else:
        img_h = img_w = img_size

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    if args.fp16:
        model.half()

    # Inductor/Triton crashes on aarch64 (Jetson) — fall back to safe settings
    import platform
    if platform.machine() == 'aarch64' and not args.no_compile:
        if args.backend == 'inductor':
            print("[inductor] WARNING: aarch64 detected (Jetson). "
                  "Switching backend to 'cudagraphs' (Triton codegen unsupported on ARM)")
            args.backend = 'cudagraphs'
            args.mode = 'default'

    # Compile model
    if not args.no_compile:
        print(f"[inductor] Compiling with mode={args.mode}, backend={args.backend}...")
        compiled_model = torch.compile(model, mode=args.mode, backend=args.backend)
    else:
        compiled_model = model

    gpu_name = torch.cuda.get_device_name(device)
    mode_str = "eager PyTorch" if args.no_compile else f"torch.compile ({args.mode})"

    # Header
    print()
    print("=" * 80)
    print(f"  {mode_str} | {model_name} | {args.dataset} | T={args.T}")
    print(f"  GPU: {gpu_name} | dtype: {'FP16' if args.fp16 else 'FP32'}")
    print("=" * 80)
    print(f"  {'Batch':<10} {'Latency (ms)':>14} {'Std (ms)':>10} {'Throughput':>12}")
    print(f"  {'--------':<10} {'--------------':>14} {'----------':>10} {'--------':>12}")

    results = {}
    for bs in batch_sizes:
        try:
            if is_nlp:
                input_tensor = torch.randint(
                    0, 1000, (bs, seq_len), device=device,
                )
            else:
                input_tensor = torch.randn(
                    bs, ds_cfg['in_channels'], img_h, img_w,
                    dtype=torch.float16 if args.fp16 else torch.float32,
                    device=device,
                )

            # Trigger compilation on first batch size (graph capture)
            if not args.no_compile and bs == batch_sizes[0]:
                reset_net(compiled_model)
                with torch.no_grad():
                    _ = compiled_model(input_tensor)
                torch.cuda.synchronize()

            times = benchmark_latency(compiled_model, input_tensor, args.warmup, args.iters)
            mean_ms = times.mean()
            std_ms = times.std()
            throughput = bs / (mean_ms / 1000)

            results[bs] = {'mean_ms': mean_ms, 'std_ms': std_ms, 'throughput': throughput}
            print(f"  B={bs:<7} {mean_ms:>11.3f}ms {std_ms:>9.3f}ms {throughput:>9.0f}/s")

            del input_tensor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  B={bs:<7} FAILED: {e}")
            torch.cuda.empty_cache()

    print("=" * 80)
    print()

    return results


if __name__ == "__main__":
    main()
