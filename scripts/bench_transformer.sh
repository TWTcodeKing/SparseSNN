#!/bin/bash
# Benchmark sengine vs TensorRT for Spiking-Transformer models.
#
# Transformer models use --config (YAML) instead of --model.
#
# Usage:
#   # SpikFormer-8-512
#   bash scripts/bench_transformer.sh configs/spikformer/spikformer_8_512.yaml [gpu_id]
#
#   # MaxFormer-10-512
#   bash scripts/bench_transformer.sh configs/maxformer/maxformer_10_512.yaml [gpu_id]
#
#   # SpikingResFormer-M
#   bash scripts/bench_transformer.sh configs/spikingresformer/spikingresformer_m.yaml [gpu_id]

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/bench_transformer.sh <config.yaml> [gpu_id]"
    echo ""
    echo "Available configs:"
    ls configs/spikformer/*.yaml configs/maxformer/*.yaml configs/spikingresformer/*.yaml 2>/dev/null | grep -v recipe
    exit 1
fi

CONFIG=$1
GPU_ID=${2:-0}
SENGINE_ONLY=${SENGINE_ONLY:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"

python3 - "$CONFIG" "$SENGINE_ONLY" << 'PYEOF'
import sys, os, time, yaml
import numpy as np

CONFIG = sys.argv[1]
SENGINE_ONLY = sys.argv[2] == '1' if len(sys.argv) > 2 else False
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

T = 4
WARMUP = 200
ITERS = 1000

EXPORT_DIR = "sengine/exports"
TRT_DIR = "trt_engines"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(TRT_DIR, exist_ok=True)

# ── Derive model name from config ──
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

arch = cfg.get('arch', 'transformer')
config_stem = os.path.splitext(os.path.basename(CONFIG))[0]  # e.g. spikformer_8_512
MODEL_TAG = config_stem

# Auto-detect dataset/img_size from config variant
variant = cfg.get('variant', '')
if variant == 'cifar' or 'cifar' in config_stem:
    IMG_SIZE = 32
    DATASET = "cifar100"
    BATCH_SIZES = [16, 32, 64, 128]
    num_classes = 100
    in_channels = 3
elif variant == 'dvs' or 'dvs' in config_stem:
    IMG_SIZE = 128
    DATASET = "cifar10dvs"
    BATCH_SIZES = [4, 8, 16, 32]
    num_classes = 10
    in_channels = 2
else:
    IMG_SIZE = 224
    DATASET = "imagenet"
    BATCH_SIZES = [4, 8, 16, 32]
    num_classes = 1000
    in_channels = 3

import torch
GPU_NAME = torch.cuda.get_device_name(0)

print(f"{'='*70}")
print(f"  {GPU_NAME}")
print(f"  Model: {MODEL_TAG} (config: {CONFIG})")
print(f"  T={T}, img={IMG_SIZE}x{IMG_SIZE}, batches={BATCH_SIZES}")
print(f"{'='*70}")

# ──────────────────────────────────────────────
# Phase 1: Export plugin ONNX (for sengine)
# ──────────────────────────────────────────────
PLUGIN_ONNX = os.path.join(EXPORT_DIR, f"{MODEL_TAG}_{DATASET}_plugin.onnx")

if not os.path.exists(PLUGIN_ONNX):
    print(f"\n[1/3] Exporting plugin ONNX: {MODEL_TAG}", flush=True)
    from sengine.scripts.export_onnx import export_model
    export_model(model_name=MODEL_TAG, output_dir=EXPORT_DIR, T=T,
                 dataset=DATASET, img_size=IMG_SIZE, config=CONFIG)
else:
    print(f"\n[1/3] Plugin ONNX exists: {PLUGIN_ONNX}", flush=True)

# ──────────────────────────────────────────────
# Phase 2: Export native ONNX + build TRT engines
# ──────────────────────────────────────────────
trt_results = {}
if SENGINE_ONLY:
    print(f"\n[2/3] Skipping TRT (sengine-only mode)", flush=True)
else:
    print(f"\n[2/3] Building TRT engines", flush=True)

    from tengine.utils import build_model_from_config
    from models.neurons import reset_net
    from sengine.tdl.transforms import export_with_fused_neurons
    from iengine.backends.tensorrt.builder import build_engine
    from iengine.backends.tensorrt.runtime import TRTRunner

    # Export ONNX ONCE at B=1 with dynamic batch — avoids OOM on
    # large models (e.g. SpikFormer ImageNet 224×224 where B=8 trace
    # creates TB=32 frames at full resolution).
    native_onnx = os.path.join(TRT_DIR, f"{MODEL_TAG}_dynamic.onnx")
    if not os.path.exists(native_onnx):
        model_cfg = dict(cfg)
        model_cfg['num_classes'] = num_classes
        model_cfg['in_channels'] = in_channels
        model_cfg['T'] = T
        model_cfg['img_size'] = IMG_SIZE
        model = build_model_from_config(model_cfg)
        model = model.cuda().eval()
        reset_net(model)
        try:
            export_with_fused_neurons(
                model, native_onnx, input_shape=(1, in_channels, IMG_SIZE, IMG_SIZE),
                dynamic_batch=True, force_native_onnx=True, verbose=False,
                opset=17)
            print(f"  Exported TDL ONNX (dynamic batch)", flush=True)
        except Exception as e:
            print(f"  TDL ONNX export FAILED: {e}", flush=True)
            native_onnx = None
        del model
        torch.cuda.empty_cache()
    else:
        print(f"  TDL ONNX exists: {native_onnx}", flush=True)

    if native_onnx and os.path.exists(native_onnx):
        max_B = max(BATCH_SIZES)
        for B in BATCH_SIZES:
            engine_path = os.path.join(TRT_DIR, f"{MODEL_TAG}_b{B}_fp16.engine")

            if not os.path.exists(engine_path):
                try:
                    build_engine(native_onnx, engine_path, sparse=False, fp16=True,
                                 min_batch=1, opt_batch=B, max_batch=max_B,
                                 workspace_gb=16.0)
                    print(f"  Built TRT engine B={B}", flush=True)
                except Exception as e:
                    print(f"  TRT build FAILED B={B}: {e}", flush=True)
                    continue

            try:
                with TRTRunner(engine_path, device=0) as runner:
                    result = runner.benchmark_latency(
                        input_shape=(B, in_channels, IMG_SIZE, IMG_SIZE),
                        n_warmup=WARMUP, n_measure=ITERS)
                    trt_results[B] = result['mean_ms']
                    print(f"  TRT B={B}: {result['mean_ms']:.3f} ms", flush=True)
            except Exception as e:
                print(f"  TRT bench FAILED B={B}: {e}", flush=True)

        torch.cuda.empty_cache()

# ──────────────────────────────────────────────
# Phase 3: Build + benchmark sengine (cached via .sengine files)
# ──────────────────────────────────────────────
print(f"\n[3/3] Building + benchmarking sengine", flush=True)
import sengine

SENGINE_DIR = "sengine/exports"
sengine_results = {}
for B in BATCH_SIZES:
    sengine_path = os.path.join(SENGINE_DIR, f"{MODEL_TAG}_B{B}.sengine")
    try:
        if os.path.exists(sengine_path):
            print(f"  Loading cached sengine B={B}...", flush=True)
            engine = sengine.load(sengine_path)
        else:
            print(f"  Building sengine B={B}...", flush=True)
            engine = sengine.build(PLUGIN_ONNX, T=T, batch_size=B)
            engine.save(sengine_path)
        ms = engine.benchmark(warmup=WARMUP, iters=ITERS)
        sengine_results[B] = ms
        print(f"  sengine B={B}: {ms:.3f} ms", flush=True)
        engine.destroy()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  sengine FAILED B={B}: {e}", flush=True)

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  {GPU_NAME} | {MODEL_TAG} | {DATASET} T={T}")
print(f"{'='*70}")
print(f"  {'Batch':<8} {'sengine (ms)':>14} {'TRT FP16 (ms)':>14} {'Speedup':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10}")

for B in BATCH_SIZES:
    se = sengine_results.get(B)
    tr = trt_results.get(B)
    se_str = f"{se:.3f}" if se else "FAIL"
    tr_str = f"{tr:.3f}" if tr else "FAIL"
    if se and tr:
        sp_str = f"{tr/se:.2f}x"
    else:
        sp_str = "-"
    print(f"  B={B:<5} {se_str:>14} {tr_str:>14} {sp_str:>10}")

print(f"{'='*70}")
PYEOF
