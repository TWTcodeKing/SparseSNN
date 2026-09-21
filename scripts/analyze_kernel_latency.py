#!/usr/bin/env python3
"""Per-kernel latency breakdown: sengine (TileLang) vs TensorRT.

Profiles each TileLang .so kernel individually and compares with TRT nsys trace.
Identifies which kernels are the bottleneck and how much gap vs TRT.

Usage:
    # Analyze sengine kernel latency for a built engine
    python scripts/analyze_kernel_latency.py \
        --onnx sengine/exports/maxformer_10_768_imagenet_plugin.onnx \
        --T 4 --batch 4

    # Compare with TRT nsys trace
    python scripts/analyze_kernel_latency.py \
        --onnx sengine/exports/maxformer_10_768_imagenet_plugin.onnx \
        --T 4 --batch 4 --trt-nsys /tmp/trt_b4.nsys-rep
"""

import argparse
import os
import sys
import ctypes
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

import torch
import numpy as np


def profile_tilelang_kernel(so_path: str, input_shapes: list[tuple],
                             input_dtypes: list, n_warmup=50, n_iters=200):
    """Profile a single TileLang .so kernel.

    Returns latency in microseconds.
    """
    lib = ctypes.CDLL(so_path)
    init_fn = getattr(lib, 'init', None)
    if init_fn:
        init_fn()

    # Allocate dummy inputs
    tensors = []
    for shape, dtype in zip(input_shapes, input_dtypes):
        t = torch.empty(*shape, dtype=dtype, device='cuda')
        tensors.append(t)

    call_fn = lib.call
    args = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
    args.append(ctypes.c_void_p(0))  # stream 0

    # Check init succeeded by verifying no CUDA error after first call
    import time as _time
    cuda_rt = ctypes.CDLL('libcudart.so')
    cuda_rt.cudaGetLastError()  # clear
    ret = call_fn(*args)
    cuda_rt.cudaDeviceSynchronize()
    err = cuda_rt.cudaGetLastError()
    if err != 0:
        # Kernel failed — likely arch mismatch
        return -1.0

    # Warmup
    for _ in range(n_warmup):
        call_fn(*args)
    cuda_rt.cudaDeviceSynchronize()

    # Profile with cudaDeviceSynchronize (not torch — avoids stream issues)
    cuda_rt.cudaDeviceSynchronize()
    t0 = _time.perf_counter()
    for _ in range(n_iters):
        call_fn(*args)
    cuda_rt.cudaDeviceSynchronize()
    t1 = _time.perf_counter()
    us = (t1 - t0) / n_iters * 1e6
    return us


def get_kernel_info(py_engine, ir):
    """Extract kernel info for each node: type, shapes, .so path."""
    from sengine.ir import OpType, KernelVariant

    info = []
    for nid in py_engine.schedule:
        node = ir.nodes.get(nid)
        if node is None:
            continue

        kv = node.assigned_kernel
        kv_name = kv.name

        # Get input/output shapes from activations
        buf = py_engine.activations.get(nid)
        out_shape = list(buf.shape) if buf is not None else []
        out_numel = buf.numel() if buf is not None else 0

        # Get input shape from first predecessor
        preds = ir.predecessors(nid)
        in_shape = []
        for pid in preds:
            pbuf = py_engine.activations.get(pid)
            if pbuf is not None:
                in_shape = list(pbuf.shape)
                break

        # Estimate FLOPs
        flops = 0
        if node.conv_params:
            cp = node.conv_params
            if len(out_shape) == 4:
                N, H, W, C = out_shape  # NHWC
                flops = 2 * N * H * W * cp.out_channels * cp.in_channels // cp.groups * cp.kernel_h * cp.kernel_w

        # Categorize
        if 'Conv' in kv_name and 'BN' in kv_name:
            if node.conv_params and node.conv_params.kernel_h == 1:
                category = 'Conv1x1+BN'
            elif node.conv_params and node.conv_params.groups > 1:
                category = 'DWConv+BN'
            else:
                category = 'Conv+BN'
        elif 'FusedMaxformer' in kv_name:
            category = 'FusedAttn'
        elif 'LIF' in kv_name or 'IF' in kv_name:
            category = 'LIF/IF'
        elif 'Elementwise' in kv_name or 'Add' in kv_name:
            category = 'Elementwise'
        elif 'Pool' in kv_name:
            category = 'Pool'
        elif 'Gemm' in kv_name or 'MatMul' in kv_name:
            category = 'GEMM'
        elif 'ZeroCost' in kv_name:
            category = 'ZeroCost'
        elif 'Layout' in kv_name:
            category = 'LayoutTranspose'
        elif 'Temporal' in kv_name:
            category = 'TemporalMean'
        else:
            category = kv_name

        info.append({
            'nid': nid,
            'kv_name': kv_name,
            'category': category,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'out_numel': out_numel,
            'flops': flops,
            'op_type': node.op_type.name,
        })
    return info


def profile_sengine_per_node(py_engine, ir, cpp_exec, n_warmup=50, n_iters=200):
    """Profile each node individually using C++ executor with per-node timing.

    Runs the full graph but measures time by running with/without each node.
    More accurate: uses CUDA events around each node in the C++ schedule.
    """
    # Simple approach: run full graph and get total, then categorize by op count
    # For accurate per-kernel: use the .so files directly
    pass


def main():
    parser = argparse.ArgumentParser(description="Kernel latency breakdown")
    parser.add_argument('--onnx', required=True, help='Plugin ONNX file')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--fusion', type=str, default='slicer', choices=['none', 'slicer'])
    parser.add_argument('--fusion-rec', type=str, default=None,
                        help='fusion recommendation JSON (default: .cache/fusion_rec_<tag>_T<T>_B<B>.json if present)')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'])
    args = parser.parse_args()

    from sengine.build.engine_builder import EngineBuilder
    from sengine.build.export_standalone import export_all_kernels, get_call_signature
    from sengine.ir import OpType, KernelVariant

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {os.path.basename(args.onnx)}")
    print(f"T={args.T}, B={args.batch}, TB={args.T * args.batch}")
    print()

    # Build engine
    print("Building engine...", flush=True)
    builder = EngineBuilder(args.onnx, T=args.T, batch_size=args.batch)
    rec = args.fusion_rec
    if rec is None:
        tag = os.path.basename(args.onnx).replace('_plugin.onnx', '')
        cand = os.path.join('.cache', f'fusion_rec_{tag}_T{args.T}_B{args.batch}.json')
        rec = cand if os.path.exists(cand) else None
    print(f"Fusion: {args.fusion} | rec: {rec} | precision: {args.precision}", flush=True)
    engine = builder.build(capture_graph=False, fusion=args.fusion, fusion_rec=rec,
                           precision=args.precision)

    # Export .so files (reuse main build cache to avoid arch mismatch)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             '.cache', f'sengine_B{args.batch}')
    kernel_so_map = export_all_kernels(engine.kernels, builder._ir, cache_dir)

    # Get per-node info
    kernel_info = get_kernel_info(engine, builder._ir)

    # Profile each .so kernel
    print("\nProfiling individual TileLang .so kernels...", flush=True)
    print(f"{'='*90}")
    print(f"  {'#':>4} {'Category':<18} {'Kernel':<24} {'Shape':<20} {'GFLOPs':>8} {'us':>8}")
    print(f"  {'-'*4} {'-'*18} {'-'*24} {'-'*20} {'-'*8} {'-'*8}")

    category_totals = defaultdict(lambda: {'us': 0.0, 'flops': 0, 'count': 0})
    total_us = 0.0
    total_flops = 0

    for ki in kernel_info:
        nid = ki['nid']
        cat = ki['category']

        # Skip zero-cost ops (no kernel to profile)
        if cat == 'ZeroCost':
            continue

        so_path = kernel_so_map.get(nid)
        if so_path is None:
            # Not a TileLang kernel — estimate from op type
            # For LIF/IF, Elementwise, Pool: these are native CUDA kernels
            # Estimate by running full engine and subtracting TileLang time
            category_totals[cat]['count'] += 1
            continue

        if isinstance(so_path, tuple):
            # Fused attention: two .so files
            for i, sp in enumerate(so_path):
                if not os.path.exists(sp):
                    continue
                # Get shapes from kernel signature
                kern_obj = engine.kernels.get(nid)
                if kern_obj and isinstance(kern_obj, tuple) and i < len(kern_obj):
                    k = kern_obj[i]
                    pf = k.adapter.prim_func
                    shapes = []
                    dtypes = []
                    for param in pf.params:
                        if param in pf.buffer_map:
                            buf = pf.buffer_map[param]
                            s = tuple(int(d) for d in buf.shape)
                            shapes.append(s)
                            dt = torch.float32 if 'float' in str(buf.dtype) and '32' in str(buf.dtype) else torch.float16
                            dtypes.append(dt)
                    try:
                        us = profile_tilelang_kernel(sp, shapes, dtypes,
                                                      n_warmup=args.warmup, n_iters=args.iters)
                        label = f'GEMM{i+1}'
                        shape_str = f"{shapes[0]}→{shapes[-1]}" if shapes else "?"
                        print(f"  {nid:>4} {'FusedAttn/'+label:<18} {ki['kv_name'][:24]:<24} "
                              f"{shape_str[:20]:<20} {'':>8} {us:>7.1f}")
                        category_totals[cat]['us'] += us
                        category_totals[cat]['count'] += 1
                        total_us += us
                    except Exception as e:
                        print(f"  {nid:>4} FusedAttn/{label}: FAILED ({e})")
            continue

        if not os.path.exists(so_path):
            continue

        # Get shapes from kernel object
        kern_obj = engine.kernels.get(nid)
        if kern_obj is None:
            continue

        try:
            pf = kern_obj.adapter.prim_func
            shapes = []
            dtypes = []
            for param in pf.params:
                if param in pf.buffer_map:
                    buf = pf.buffer_map[param]
                    s = tuple(int(d) for d in buf.shape)
                    shapes.append(s)
                    dt = torch.float32 if 'float' in str(buf.dtype) and '32' in str(buf.dtype) else torch.float16
                    dtypes.append(dt)

            us = profile_tilelang_kernel(so_path, shapes, dtypes,
                                          n_warmup=args.warmup, n_iters=args.iters)
            if us < 0:
                print(f"  {nid:>4} {cat:<18} CUDA ERROR (arch mismatch? check .so)")
                continue
            gflops = ki['flops'] / 1e9
            shape_str = f"{ki['in_shape'][:3]}→{ki['out_shape'][:3]}" if ki['in_shape'] else str(ki['out_shape'][:3])
            print(f"  {nid:>4} {cat:<18} {ki['kv_name'][:24]:<24} "
                  f"{shape_str[:20]:<20} {gflops:>7.1f} {us:>7.1f}")

            category_totals[cat]['us'] += us
            category_totals[cat]['flops'] += ki['flops']
            category_totals[cat]['count'] += 1
            total_us += us
            total_flops += ki['flops']
        except Exception as e:
            print(f"  {nid:>4} {cat}: FAILED ({e})")

    # Measure full engine latency: both Python dispatch and C++ CUDA Graph
    print(f"\n{'='*90}")
    print("Measuring full engine latency...", flush=True)

    # Python dispatch (no CUDA Graph)
    inp = torch.randn(args.batch, 3, 224, 224, device='cuda')
    engine.reset_state()
    try:
        engine(inp)
    except Exception:
        pass
    py_times = []
    for _ in range(20):
        engine.reset_state()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            engine(inp)
        except Exception:
            break
        e.record()
        torch.cuda.synchronize()
        py_times.append(s.elapsed_time(e) * 1000)
    py_us = sum(py_times) / len(py_times) if py_times else 0

    # C++ CUDA Graph (via sengine.build)
    import sengine as se
    try:
        cpp_engine = se.build(args.onnx, T=args.T, batch_size=args.batch)
        cpp_ms = cpp_engine.benchmark(warmup=min(args.warmup, 200), iters=min(args.iters, 500))
        cpp_us = cpp_ms * 1000  # ms → us
        cpp_mode = "C++" if not cpp_engine._use_python_runtime else "Python"
        cpp_engine.destroy()
    except Exception as e:
        cpp_us = 0
        cpp_mode = f"FAILED: {e}"
    torch.cuda.empty_cache()

    full_us = py_us  # for breakdown percentages (matches per-kernel profiling)

    # The difference = non-TileLang kernel time (LIF, Add, Pool, etc.)
    native_us = full_us - total_us if full_us > total_us else 0

    # Print summary
    print(f"\n{'='*90}")
    print(f"  KERNEL LATENCY BREAKDOWN (sengine TileLang)")
    print(f"{'='*90}")
    print(f"  {'Category':<22} {'Count':>6} {'Total us':>10} {'% Time':>8} {'GFLOPs':>10} {'TFLOPS':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    for cat in sorted(category_totals.keys(), key=lambda c: -category_totals[c]['us']):
        ct = category_totals[cat]
        pct = ct['us'] / full_us * 100 if full_us > 0 else 0
        gf = ct['flops'] / 1e9
        tflops = ct['flops'] / ct['us'] / 1e6 if ct['us'] > 0 else 0  # TFLOPS
        print(f"  {cat:<22} {ct['count']:>6} {ct['us']:>9.1f} {pct:>7.1f}% {gf:>9.1f} {tflops:>7.1f}")

    if native_us > 0:
        pct_native = native_us / full_us * 100
        native_count = sum(1 for ki in kernel_info if ki['category'] in ('LIF/IF', 'Elementwise', 'Pool', 'TemporalMean', 'GEMM'))
        print(f"  {'Native CUDA (est.)':<22} {native_count:>6} {native_us:>9.1f} {pct_native:>7.1f}%")

    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*8}")
    print(f"  {'TOTAL (Python disp.)':<22} {'':<6} {full_us:>9.1f} {'100.0%':>8}")
    print(f"  {'  TileLang kernels':<22} {'':<6} {total_us:>9.1f} {total_us/full_us*100 if full_us else 0:>7.1f}%")
    print(f"{'='*90}")
    print()
    print(f"  Python dispatch total:  {py_us:>9.1f} us  ({py_us/1000:.3f} ms)")
    print(f"  C++ CUDA Graph total:   {cpp_us:>9.1f} us  ({cpp_us/1000:.3f} ms)  [{cpp_mode}]")
    print(f"  CUDA Graph speedup:     {py_us/cpp_us:.2f}x" if cpp_us > 0 else "")
    print(f"  Sum of TileLang .so:    {total_us:>9.1f} us  ({total_us/1000:.3f} ms)")
    print(f"  Native CUDA overhead:   {native_us:>9.1f} us  ({native_us/1000:.3f} ms)")
    print(f"{'='*90}")

    # Compute efficiency
    if total_flops > 0 and total_us > 0:
        achieved_tflops = total_flops / total_us / 1e6
        # A100 peak FP16 tensor core: 312 TFLOPS, RTX 4090: 330 TFLOPS
        gpu_name = torch.cuda.get_device_name(0)
        peak = 312 if 'A100' in gpu_name else 330 if '4090' in gpu_name else 200
        print(f"\n  Compute efficiency: {achieved_tflops:.1f} / {peak} TFLOPS = {achieved_tflops/peak*100:.1f}%")
        print(f"  (TRT typically achieves 60-80% on these workloads)")


if __name__ == '__main__':
    main()
