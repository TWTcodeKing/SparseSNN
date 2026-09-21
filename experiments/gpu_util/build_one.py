#!/usr/bin/env python3
"""Build a single sengine or TRT engine for one model + batch size.

Usage:
    # Build sengine for maxformer B=16
    python experiments/gpu_util/build_one.py --model maxformer_10_512 --batch 16 --backend sengine --gpu-id 3

    # Build TRT for sew_resnet101 B=32
    python experiments/gpu_util/build_one.py --model sew_resnet101 --batch 32 --backend trt --gpu-id 3

    # Build both
    python experiments/gpu_util/build_one.py --model maxformer_10_512 --batch 16 --backend both --gpu-id 3
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch

from experiments.gpu_util.config import (
    MODELS, T, PRECISION, DATASET, IMG_SIZE, IN_CHANNELS, NUM_CLASSES,
    PROJECT_ROOT, plugin_onnx_path, sengine_path,
    trt_engine_path, trt_onnx_path, SENGINE_DIR, TRT_ENGINES_DIR,
    select_gpu,
)


def build_sengine(model_key, batch, device_id):
    """Build sengine in subprocess."""
    onnx = plugin_onnx_path(model_key)
    out = sengine_path(model_key, batch)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if os.path.exists(out):
        print(f"[sengine] {model_key} B={batch}: already exists → {out}")
        return True

    if not os.path.exists(onnx):
        print(f"[sengine] ERROR: plugin ONNX not found: {onnx}")
        return False

    result_file = f'/tmp/gputil_build_{model_key}_B{batch}_sengine.json'
    sub_cmd = f"""
import os, sys, time, json
sys.path.insert(0, '{PROJECT_ROOT}')
os.environ['CUDA_HOME'] = '{os.environ.get("CUDA_HOME", "")}'
os.environ['PATH'] = '{os.environ.get("PATH", "")}'
import sengine
try:
    t0 = time.time()
    e = sengine.build('{onnx}', T={T}, batch_size={batch},
                       fusion='slicer', autotune=True, precision='{PRECISION}')
    build_s = time.time() - t0
    e.save('{out}')
    ms = e.benchmark(warmup=50, iters=200)
    json.dump({{'ok': True, 'build_s': build_s, 'ms': ms}}, open('{result_file}', 'w'))
except Exception as ex:
    import traceback
    json.dump({{'ok': False, 'error': str(ex), 'tb': traceback.format_exc()}},
              open('{result_file}', 'w'))
"""
    print(f"[sengine] Building {model_key} B={batch} (subprocess, autotune=True)...")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, '-c', sub_cmd],
        env=env, timeout=3600, capture_output=True, text=True)

    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        os.remove(result_file)
        if r.get('ok'):
            print(f"[sengine] Done in {r['build_s']:.0f}s, latency={r['ms']:.3f}ms → {out}")
            return True
        else:
            print(f"[sengine] FAILED: {r['error']}")
            if r.get('tb'):
                for line in r['tb'].strip().split('\n')[-5:]:
                    print(f"  {line}")
            return False
    else:
        print(f"[sengine] Subprocess crashed after {time.time()-t0:.0f}s")
        if proc.stderr:
            for line in proc.stderr.strip().split('\n')[-8:]:
                print(f"  {line}")
        return False


def build_trt(model_key, batch, device_id):
    """Build TRT engine: export standard ONNX + build .engine."""
    engine_out = trt_engine_path(model_key, batch)
    onnx_out = trt_onnx_path(model_key, batch)
    os.makedirs(os.path.dirname(engine_out), exist_ok=True)

    if os.path.exists(engine_out):
        print(f"[TRT] {model_key} B={batch}: engine exists → {engine_out}")
        return True

    from tengine.utils import build_model, build_model_from_config, load_model_config, get_dataset_config
    from models.neurons import reset_net
    from iengine.backends.tensorrt.export import export_onnx
    from iengine.backends.tensorrt.builder import build_engine

    ds_cfg = get_dataset_config(DATASET)
    info = MODELS[model_key]

    # Export standard ONNX if needed
    if not os.path.exists(onnx_out):
        print(f"[TRT] Exporting standard ONNX for {model_key} B={batch}...")
        device = torch.device(f'cuda:{device_id}')
        if info["type"] == "config":
            config = load_model_config(os.path.join(PROJECT_ROOT, info["spec"]))
            config.update(ds_cfg)
            config['T'] = T
            config['img_size'] = IMG_SIZE
            model = build_model_from_config(config)
        else:
            model = build_model(info["spec"], T=T,
                                num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
        model = model.to(device).eval()
        reset_net(model)
        export_onnx(model, onnx_out,
                    input_shape=(batch, IN_CHANNELS, IMG_SIZE, IMG_SIZE),
                    dynamic_batch=False, simplify=True, verbose=False)
        print(f"  Saved: {onnx_out}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"[TRT] ONNX exists: {onnx_out}")

    # Build TRT engine
    print(f"[TRT] Building engine {model_key} B={batch} (FP16)...")
    gpu_mem = torch.cuda.get_device_properties(device_id).total_memory
    gpu_free = gpu_mem - torch.cuda.memory_reserved(device_id)
    workspace_gb = max(1.0, (gpu_free / (1 << 30)) - 2.0)

    t0 = time.time()
    try:
        build_engine(onnx_out, engine_out,
                     sparse=False, fp16=True,
                     min_batch=batch, opt_batch=batch, max_batch=batch,
                     workspace_gb=workspace_gb, verbose=False)
        print(f"[TRT] Done in {time.time()-t0:.0f}s → {engine_out}")
        return True
    except Exception as e:
        print(f"[TRT] FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build one engine")
    parser.add_argument('--model', type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--backend', type=str, required=True, choices=['sengine', 'trt', 'both'])
    parser.add_argument('--gpu-id', type=int, default=None)
    args = parser.parse_args()

    device_id = args.gpu_id if args.gpu_id is not None else select_gpu()
    torch.cuda.set_device(device_id)
    print(f"GPU {device_id}: {torch.cuda.get_device_name(device_id)}")

    if args.backend in ('sengine', 'both'):
        build_sengine(args.model, args.batch, device_id)
    if args.backend in ('trt', 'both'):
        build_trt(args.model, args.batch, device_id)


if __name__ == '__main__':
    main()
