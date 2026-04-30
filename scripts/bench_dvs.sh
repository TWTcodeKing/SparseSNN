#!/bin/bash
# Benchmark sengine vs TensorRT for DVS models on CIFAR10-DVS (T=16).
#
# Models benchmarked:
#   VGG-SNN-16      (--model snn_vgg16)
#   MaxFormer-DVS   (--config configs/maxformer/maxformer_dvs.yaml)
#   MS-QKFormer-DVS (--config configs/maxformer/ms_qkformer_dvs.yaml)
#   SpikingResFormer-S (--config configs/spikingresformer/spikingresformer_s.yaml)
#
# Usage:
#   bash scripts/bench_dvs.sh [gpu_id]
#
# Environment:
#   SENGINE_ONLY=1 bash scripts/bench_dvs.sh   # skip TRT, sengine only

set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"

SENGINE_ONLY=${SENGINE_ONLY:-0}

python3 - "$SENGINE_ONLY" << 'PYEOF'
import sys, os, time, yaml
import numpy as np

SENGINE_ONLY = sys.argv[1] == '1' if len(sys.argv) > 1 else False
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

# ── Benchmark settings ──
T = 16
IMG_SIZE = 128
DATASET = "cifar10dvs"
BATCH_SIZES = [4, 8, 16, 32]
WARMUP = 200
ITERS = 1000
NUM_CLASSES = 10
IN_CHANNELS = 2

EXPORT_DIR = "sengine/exports"
TRT_DIR = "trt_engines"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(TRT_DIR, exist_ok=True)

# ── Models to benchmark ──
# Each entry: (tag, model_name_or_None, config_path_or_None)
MODELS = [
    ("snn_vgg16",          "snn_vgg16",  None),
    ("maxformer_dvs",      None,         "configs/maxformer/maxformer_dvs.yaml"),
    ("ms_qkformer_dvs",    None,         "configs/maxformer/ms_qkformer_dvs.yaml"),
    ("spikingresformer_s", None,         "configs/spikingresformer/spikingresformer_s.yaml"),
]

import torch
GPU_NAME = torch.cuda.get_device_name(0)

print(f"{'='*80}")
print(f"  {GPU_NAME} | CIFAR10-DVS | T={T} | img={IMG_SIZE}x{IMG_SIZE}")
print(f"{'='*80}")


def export_plugin_onnx(tag, model_name, config_path):
    """Export plugin-mode ONNX for sengine."""
    onnx_path = os.path.join(EXPORT_DIR, f"{tag}_{DATASET}_plugin.onnx")
    if os.path.exists(onnx_path):
        print(f"  Plugin ONNX exists: {onnx_path}", flush=True)
        return onnx_path

    print(f"  Exporting plugin ONNX: {tag}", flush=True)
    from sengine.scripts.export_onnx import export_model
    export_model(
        model_name=model_name or tag,
        output_dir=EXPORT_DIR, T=T,
        dataset=DATASET, img_size=IMG_SIZE,
        config=config_path,
    )
    return onnx_path


def bench_trt_model(tag, model_name, config_path):
    """Build TRT engines and benchmark for all batch sizes."""
    from models.neurons import reset_net
    from iengine.backends.tensorrt.builder import build_engine
    from iengine.backends.tensorrt.runtime import TRTRunner

    results = {}

    # Export native ONNX (dynamic batch, single export)
    native_onnx = os.path.join(TRT_DIR, f"{tag}_dynamic.onnx")
    if not os.path.exists(native_onnx):
        if config_path:
            from tengine.utils import build_model_from_config
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg['num_classes'] = NUM_CLASSES
            cfg['in_channels'] = IN_CHANNELS
            cfg['T'] = T
            cfg['img_size'] = IMG_SIZE
            model = build_model_from_config(cfg)
        else:
            from tengine.utils import build_model
            model = build_model(model_name, T=T, num_classes=NUM_CLASSES,
                                in_channels=IN_CHANNELS)
        model = model.cuda().eval()
        reset_net(model)

        try:
            from sengine.tdl.transforms import export_with_fused_neurons
            export_with_fused_neurons(
                model, native_onnx,
                input_shape=(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE),
                dynamic_batch=True, force_native_onnx=True,
                verbose=False, opset=17)
            print(f"  Exported TDL ONNX (dynamic batch)", flush=True)
        except Exception as e:
            print(f"  TDL ONNX export FAILED: {e}", flush=True)
            native_onnx = None

        del model
        torch.cuda.empty_cache()
    else:
        print(f"  TDL ONNX exists: {native_onnx}", flush=True)

    if not native_onnx or not os.path.exists(native_onnx):
        return results

    max_B = max(BATCH_SIZES)
    for B in BATCH_SIZES:
        engine_path = os.path.join(TRT_DIR, f"{tag}_b{B}_fp16.engine")

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
                    input_shape=(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE),
                    n_warmup=WARMUP, n_measure=ITERS)
                results[B] = result['mean_ms']
                print(f"  TRT B={B}: {result['mean_ms']:.3f} ms", flush=True)
        except Exception as e:
            print(f"  TRT bench FAILED B={B}: {e}", flush=True)

    torch.cuda.empty_cache()
    return results


def bench_sengine_model(tag, plugin_onnx):
    """Build/load sengine and benchmark for all batch sizes."""
    import sengine

    results = {}
    for B in BATCH_SIZES:
        sengine_path = os.path.join(EXPORT_DIR, f"{tag}_T{T}_B{B}.sengine")
        try:
            if os.path.exists(sengine_path):
                print(f"  Loading cached sengine B={B}...", flush=True)
                engine = sengine.load(sengine_path,
                                      build_dir=f"/tmp/sengine_{tag}_B{B}")
            else:
                print(f"  Building sengine B={B}...", flush=True)
                engine = sengine.build(plugin_onnx, T=T, batch_size=B,
                                       build_dir=f"/tmp/sengine_{tag}_B{B}")
                engine.save(sengine_path)
            ms = engine.benchmark(warmup=WARMUP, iters=ITERS)
            results[B] = ms
            print(f"  sengine B={B}: {ms:.3f} ms", flush=True)
            engine.destroy()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  sengine FAILED B={B}: {e}", flush=True)
    return results


# ── Run benchmarks ──
all_sengine = {}   # tag -> {B: ms}
all_trt = {}       # tag -> {B: ms}

for tag, model_name, config_path in MODELS:
    print(f"\n{'─'*60}")
    print(f"  Model: {tag}")
    print(f"{'─'*60}")

    # Phase 1: Export plugin ONNX
    print(f"\n  [1/3] ONNX export", flush=True)
    try:
        plugin_onnx = export_plugin_onnx(tag, model_name, config_path)
    except Exception as e:
        print(f"  ONNX export FAILED: {e}", flush=True)
        continue

    # Phase 2: TRT
    if SENGINE_ONLY:
        print(f"\n  [2/3] Skipping TRT (sengine-only mode)", flush=True)
        all_trt[tag] = {}
    else:
        print(f"\n  [2/3] TensorRT benchmark", flush=True)
        all_trt[tag] = bench_trt_model(tag, model_name, config_path)

    # Phase 3: sengine
    print(f"\n  [3/3] sengine benchmark", flush=True)
    all_sengine[tag] = bench_sengine_model(tag, plugin_onnx)

# ── Final results table ──
TAGS = [t for t, _, _ in MODELS]
LABELS = ["VGG-SNN-16", "MaxFormer-DVS", "MS-QKFormer-DVS", "SpikingResFormer-S"]

col_w = 20
header = f"{'':>8}"
for label in LABELS:
    header += f"  {label:>{col_w}}"

print(f"\n\n{'='*100}")
print(f"  {GPU_NAME} | CIFAR10-DVS | T={T}")
print(f"{'='*100}")

# sengine results
print(f"\n  sengine latency (ms)")
print(f"  {header}")
print(f"  {'─'*8}" + f"  {'─'*col_w}" * len(LABELS))
for B in BATCH_SIZES:
    row = f"  B={B:<5}"
    for tag in TAGS:
        ms = all_sengine.get(tag, {}).get(B)
        row += f"  {ms:>{col_w}.3f}" if ms else f"  {'FAIL':>{col_w}}"
    print(row)

# TRT results
if not SENGINE_ONLY:
    print(f"\n  TRT FP16 latency (ms)")
    print(f"  {header}")
    print(f"  {'─'*8}" + f"  {'─'*col_w}" * len(LABELS))
    for B in BATCH_SIZES:
        row = f"  B={B:<5}"
        for tag in TAGS:
            ms = all_trt.get(tag, {}).get(B)
            row += f"  {ms:>{col_w}.3f}" if ms else f"  {'FAIL':>{col_w}}"
        print(row)

    # Speedup
    print(f"\n  Speedup (TRT / sengine)")
    print(f"  {header}")
    print(f"  {'─'*8}" + f"  {'─'*col_w}" * len(LABELS))
    for B in BATCH_SIZES:
        row = f"  B={B:<5}"
        for tag in TAGS:
            se = all_sengine.get(tag, {}).get(B)
            tr = all_trt.get(tag, {}).get(B)
            if se and tr:
                row += f"  {tr/se:>{col_w}.2f}x"
            else:
                row += f"  {'-':>{col_w}}"
        print(row)

print(f"{'='*100}")
PYEOF
