#!/usr/bin/env python3
"""Phase 1: Build all sengine + TRT engines for GPU utilization profiling.

Usage:
    python GPUtil/build_engines.py [--gpu-id 3] [--skip-sengine] [--skip-trt]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Auto-detect CUDA
for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch

from GPUtil.config import (
    MODELS, BATCH_SIZES, T, PRECISION, DATASET, IMG_SIZE, IN_CHANNELS, NUM_CLASSES,
    PROJECT_ROOT, select_gpu, plugin_onnx_path, sengine_path,
    trt_engine_path, trt_onnx_path, SENGINE_DIR, TRT_ENGINES_DIR,
)


def build_sengine_subprocess(model_key, batch, device_id):
    """Build one sengine engine in a subprocess for clean CUDA state."""
    onnx = plugin_onnx_path(model_key)
    out_path = sengine_path(model_key, batch)
    if os.path.exists(out_path):
        print(f"  [sengine] {model_key} B={batch}: exists, skipping")
        return True

    result_file = f'/tmp/gputil_sengine_{model_key}_B{batch}.json'
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
    e.save('{out_path}')
    ms = e.benchmark(warmup=50, iters=200)
    json.dump({{'ok': True, 'build_s': build_s, 'ms': ms}}, open('{result_file}', 'w'))
except Exception as ex:
    import traceback
    json.dump({{'ok': False, 'error': str(ex), 'tb': traceback.format_exc()}},
              open('{result_file}', 'w'))
"""
    print(f"  [sengine] {model_key} B={batch}: building (subprocess)...")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device_id)
    proc = subprocess.run(
        [sys.executable, '-c', sub_cmd],
        env=env, timeout=3600, capture_output=True, text=True)

    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        os.remove(result_file)
        if r.get('ok'):
            print(f"  [sengine] {model_key} B={batch}: done in {r['build_s']:.0f}s, "
                  f"latency={r['ms']:.3f}ms")
            return True
        else:
            print(f"  [sengine] {model_key} B={batch}: FAILED: {r['error']}")
            if proc.stderr:
                for line in proc.stderr.strip().split('\n')[-5:]:
                    print(f"    {line}")
            return False
    else:
        print(f"  [sengine] {model_key} B={batch}: subprocess crashed")
        if proc.stderr:
            for line in proc.stderr.strip().split('\n')[-5:]:
                print(f"    {line}")
        return False


def build_all_trt(device_id):
    """Build all TRT engines: export standard ONNX, then build .engine files."""
    from tengine.utils import (
        build_model, build_model_from_config, load_model_config, get_dataset_config,
    )
    from models.neurons import reset_net
    from iengine.backends.tensorrt.export import export_onnx
    from iengine.backends.tensorrt.builder import build_engine

    ds_cfg = get_dataset_config(DATASET)
    device = torch.device(f'cuda:{device_id}')

    # Phase 1: Export all standard ONNX (model must be on GPU)
    for model_key, info in MODELS.items():
        # Check if all batch ONNX already exist
        all_exist = all(os.path.exists(trt_onnx_path(model_key, B)) for B in BATCH_SIZES)
        if all_exist:
            print(f"  [TRT export] {model_key}: all ONNX exist, skipping")
            continue

        # Build the PyTorch model
        print(f"  [TRT export] {model_key}: loading model...")
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

        for B in BATCH_SIZES:
            onnx_out = trt_onnx_path(model_key, B)
            if os.path.exists(onnx_out):
                print(f"  [TRT export] {model_key} B={B}: ONNX exists")
                continue
            print(f"  [TRT export] {model_key} B={B}: exporting standard ONNX...")
            reset_net(model)
            export_onnx(model, onnx_out,
                        input_shape=(B, IN_CHANNELS, IMG_SIZE, IMG_SIZE),
                        dynamic_batch=False, simplify=True, verbose=False)
            print(f"    Saved: {onnx_out}")

        # Free model before next one
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 2: Build TRT engines (model freed, max GPU memory available)
    gpu_mem = torch.cuda.get_device_properties(device_id).total_memory
    gpu_free = gpu_mem - torch.cuda.memory_reserved(device_id)
    workspace_gb = max(1.0, (gpu_free / (1 << 30)) - 2.0)

    for model_key in MODELS:
        for B in BATCH_SIZES:
            engine_out = trt_engine_path(model_key, B)
            if os.path.exists(engine_out):
                print(f"  [TRT build] {model_key} B={B}: engine exists, skipping")
                continue
            onnx_in = trt_onnx_path(model_key, B)
            if not os.path.exists(onnx_in):
                print(f"  [TRT build] {model_key} B={B}: ONNX missing, skipping")
                continue
            print(f"  [TRT build] {model_key} B={B}: building (FP16, workspace={workspace_gb:.1f}GB)...")
            t0 = time.time()
            try:
                build_engine(onnx_in, engine_out,
                             sparse=False, fp16=True,
                             min_batch=B, opt_batch=B, max_batch=B,
                             workspace_gb=workspace_gb, verbose=False)
                print(f"    Done in {time.time()-t0:.0f}s: {engine_out}")
            except Exception as e:
                print(f"    FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build all engines for GPU utilization profiling")
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='GPU device (auto-select if omitted)')
    parser.add_argument('--skip-sengine', action='store_true')
    parser.add_argument('--skip-trt', action='store_true')
    args = parser.parse_args()

    device_id = args.gpu_id if args.gpu_id is not None else select_gpu()
    torch.cuda.set_device(device_id)
    print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Precision: {PRECISION}")

    os.makedirs(SENGINE_DIR, exist_ok=True)
    os.makedirs(TRT_ENGINES_DIR, exist_ok=True)

    # Build sengine engines
    if not args.skip_sengine:
        print(f"\n{'='*60}")
        print(f"  Building sengine engines")
        print(f"{'='*60}")
        for model_key in MODELS:
            for B in BATCH_SIZES:
                build_sengine_subprocess(model_key, B, device_id)

    # Build TRT engines
    if not args.skip_trt:
        print(f"\n{'='*60}")
        print(f"  Building TRT engines")
        print(f"{'='*60}")
        build_all_trt(device_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Build Summary")
    print(f"{'='*60}")
    for model_key in MODELS:
        for B in BATCH_SIZES:
            se = "OK" if os.path.exists(sengine_path(model_key, B)) else "MISSING"
            te = "OK" if os.path.exists(trt_engine_path(model_key, B)) else "MISSING"
            print(f"  {model_key:<25} B={B:<4}  sengine={se:<8} TRT={te}")


if __name__ == '__main__':
    main()
