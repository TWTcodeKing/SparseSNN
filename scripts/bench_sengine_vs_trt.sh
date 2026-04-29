#!/bin/bash
# Benchmark sengine vs TensorRT for a given model + batch sizes.
#
# Usage:
#   bash scripts/bench_sengine_vs_trt.sh [gpu_id]
#
# Outputs a markdown table to stdout.

set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"

MODEL="sew_resnet101"
DATASET="imagenet"
T=4
IMG_SIZE=224
BATCH_SIZES=(4 8 16 32)
WARMUP=200
ITERS=1000

EXPORT_DIR="sengine/exports"
TRT_DIR="trt_engines"
mkdir -p "$EXPORT_DIR" "$TRT_DIR"

PLUGIN_ONNX="${EXPORT_DIR}/${MODEL}_${DATASET}_plugin.onnx"

python3 - "$MODEL" "$DATASET" "$T" "$IMG_SIZE" "$PLUGIN_ONNX" "$TRT_DIR" "$WARMUP" "$ITERS" "${BATCH_SIZES[@]}" << 'PYEOF'
import sys, os, time, json
import numpy as np

args = sys.argv[1:]
MODEL = args[0]
DATASET = args[1]
T = int(args[2])
IMG_SIZE = int(args[3])
PLUGIN_ONNX = args[4]
TRT_DIR = args[5]
WARMUP = int(args[6])
ITERS = int(args[7])
BATCH_SIZES = [int(x) for x in args[8:]]

os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

import torch
GPU_NAME = torch.cuda.get_device_name(0)

# ──────────────────────────────────────────────
# Phase 1: Export plugin ONNX (once)
# ──────────────────────────────────────────────
if not os.path.exists(PLUGIN_ONNX):
    print(f"[1/3] Exporting plugin ONNX: {MODEL} → {PLUGIN_ONNX}", flush=True)
    from sengine.scripts.export_onnx import export_model
    export_model(MODEL, os.path.dirname(PLUGIN_ONNX), T=T,
                 dataset=DATASET, img_size=IMG_SIZE)
else:
    print(f"[1/3] Plugin ONNX exists: {PLUGIN_ONNX}", flush=True)

# ──────────────────────────────────────────────
# Phase 2: Export native ONNX + build TRT engines
# ──────────────────────────────────────────────
print(f"[2/3] Building TRT engines for B={BATCH_SIZES}", flush=True)

from tengine.utils import build_model
from models.neurons import reset_net
from iengine.backends.tensorrt.export import export_onnx
from iengine.backends.tensorrt.builder import build_engine
from iengine.backends.tensorrt.runtime import TRTRunner

num_classes = 1000 if 'imagenet' in DATASET else 100
model = build_model(MODEL, T=T, num_classes=num_classes, in_channels=3)
model = model.cuda().eval()

trt_results = {}
for B in BATCH_SIZES:
    native_onnx = os.path.join(TRT_DIR, f"{MODEL}_b{B}_native.onnx")
    engine_path = os.path.join(TRT_DIR, f"{MODEL}_b{B}_fp16.engine")

    # Export native ONNX
    if not os.path.exists(native_onnx):
        reset_net(model)
        export_onnx(model, native_onnx, input_shape=(B, 3, IMG_SIZE, IMG_SIZE),
                    dynamic_batch=False, simplify=True, verbose=False)
        print(f"  Exported native ONNX B={B}", flush=True)

    # Build TRT engine
    if not os.path.exists(engine_path):
        try:
            build_engine(native_onnx, engine_path, sparse=False, fp16=True,
                         min_batch=B, opt_batch=B, max_batch=B, workspace_gb=4.0)
            print(f"  Built TRT engine B={B}", flush=True)
        except Exception as e:
            print(f"  TRT build FAILED B={B}: {e}", flush=True)
            continue

    # Benchmark TRT
    try:
        with TRTRunner(engine_path, device=0) as runner:
            result = runner.benchmark_latency(
                input_shape=(B, 3, IMG_SIZE, IMG_SIZE),
                n_warmup=WARMUP, n_measure=ITERS)
            trt_results[B] = result['mean_ms']
            print(f"  TRT B={B}: {result['mean_ms']:.3f} ms", flush=True)
    except Exception as e:
        print(f"  TRT bench FAILED B={B}: {e}", flush=True)

del model
torch.cuda.empty_cache()

# ──────────────────────────────────────────────
# Phase 3: Build + benchmark sengine
# ──────────────────────────────────────────────
print(f"[3/3] Building + benchmarking sengine for B={BATCH_SIZES}", flush=True)
import sengine

sengine_results = {}
for B in BATCH_SIZES:
    try:
        engine = sengine.build(PLUGIN_ONNX, T=T, batch_size=B,
                               build_dir=f"/tmp/sengine_{MODEL}_B{B}")
        ms = engine.benchmark(warmup=WARMUP, iters=ITERS)
        sengine_results[B] = ms
        print(f"  sengine B={B}: {ms:.3f} ms", flush=True)
        engine.destroy()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  sengine FAILED B={B}: {e}", flush=True)

# ──────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  {GPU_NAME} | {MODEL} | ImageNet T={T}")
print(f"{'='*70}")
print(f"  {'Batch':<8} {'sengine (ms)':>14} {'TRT FP16 (ms)':>14} {'Speedup':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10}")

for B in BATCH_SIZES:
    se = sengine_results.get(B)
    tr = trt_results.get(B)
    se_str = f"{se:.3f}" if se else "FAIL"
    tr_str = f"{tr:.3f}" if tr else "FAIL"
    if se and tr:
        ratio = tr / se
        sp_str = f"{ratio:.2f}x"
    else:
        sp_str = "-"
    print(f"  B={B:<5} {se_str:>14} {tr_str:>14} {sp_str:>10}")

print(f"{'='*70}")
PYEOF
