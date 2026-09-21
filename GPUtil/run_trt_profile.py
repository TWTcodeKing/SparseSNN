#!/usr/bin/env python3
"""Minimal TRT engine runner for ncu profiling.

Uses cudaProfilerStart/Stop to mark the measured inference pass.
Combined with ncu --profile-from-start off, only inference kernels are profiled.

Usage (standalone test):
    python GPUtil/run_trt_profile.py --model sew_resnet101 --batch 16 --warmup 10 --iters 1

Under ncu:
    sudo CUDA_VISIBLE_DEVICES=3 /opt/.../ncu --profile-from-start off \
        --target-processes all --metrics ... \
        .venv/bin/python GPUtil/run_trt_profile.py --model sew_resnet101 --batch 16
"""

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from GPUtil.config import MODELS, T, trt_engine_path

_trt_lib = os.path.join(os.path.dirname(__file__), '..', '.venv', 'lib', 'python3.12',
                        'site-packages', 'tensorrt_libs', 'libnvinfer.so.10')
if os.path.exists(_trt_lib):
    ctypes.CDLL(_trt_lib, mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description="TRT runner for ncu profiling")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=1)
    args = parser.parse_args()

    engine_path = trt_engine_path(args.model, args.batch)
    if not os.path.exists(engine_path):
        print(f"ERROR: TRT engine not found: {engine_path}")
        print(f"Build it first:  python GPUtil/build_one.py --model {args.model} "
              f"--batch {args.batch} --backend trt --gpu-id <N>")
        sys.exit(1)

    print(f"=== TRT ncu profile runner ===")
    print(f"Model: {args.model}, B={args.batch}")
    print(f"Engine: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    inp_shape = tuple(max(1, s) for s in engine.get_tensor_shape(inp_name))
    ctx.set_input_shape(inp_name, inp_shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    print(f"Input:  {inp_name} {inp_shape}")
    print(f"Output: {out_name} {out_shape}")

    x = torch.randn(*inp_shape, dtype=torch.float32, device='cuda:0')
    y = torch.zeros(*out_shape, dtype=torch.float32, device='cuda:0')
    ctx.set_tensor_address(inp_name, x.data_ptr())
    ctx.set_tensor_address(out_name, y.data_ptr())
    stream = torch.cuda.Stream(device=0)

    # Warmup (NOT profiled)
    print(f"Warmup: {args.warmup} iters...")
    for _ in range(args.warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # cudaProfilerStart — ncu begins capturing here
    torch.cuda.cudart().cudaProfilerStart()

    print(f"Profiling: {args.iters} iters...")
    t0 = time.time()
    for _ in range(args.iters):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    elapsed = time.time() - t0

    # cudaProfilerStop
    torch.cuda.cudart().cudaProfilerStop()

    ms = elapsed / args.iters * 1000
    print(f"Done: {ms:.3f} ms/iter")


if __name__ == '__main__':
    main()
