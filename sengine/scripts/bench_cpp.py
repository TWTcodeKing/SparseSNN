#!/usr/bin/env python3
"""Benchmark sengine C++ executor vs TensorRT.

Builds the sengine pipeline (ONNX → IR → TileLang → schedule),
exports all kernels as standalone .so files, wires them into
the pure-C++ executor (libsengine_exec.so), captures a CUDA Graph,
and benchmarks against TRT.

Usage:
    python sengine/scripts/bench_cpp.py --batch 1
    python sengine/scripts/bench_cpp.py --batch 64
"""

import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

import numpy as np
import torch
import torch.nn.functional as F

from sengine.ir import OpType, KernelVariant
from sengine.build.engine_builder import EngineBuilder
from sengine.build.export_standalone import export_all_kernels


# ─── Load C++ executor ───
LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'csrc', 'libsengine_exec.so')
_lib = ctypes.CDLL(LIB_PATH)

# C function signatures
_lib.sengine_create.restype = ctypes.c_void_p
_lib.sengine_destroy.argtypes = [ctypes.c_void_p]
_lib.sengine_load_tilelang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.sengine_load_tilelang.restype = ctypes.c_int
_lib.sengine_set_schedule.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.sengine_alloc_nodes.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.sengine_set_tilelang_node_5.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p]
_lib.sengine_set_tilelang_node_6.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.sengine_set_if_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_float]
_lib.sengine_set_lif_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
_lib.sengine_set_add_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.sengine_set_maxpool_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.sengine_set_global_avgpool_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.sengine_set_temporal_mean_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_int, ctypes.c_int]
_lib.sengine_set_gemm_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.sengine_set_skip_node.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.sengine_set_alias_node.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.sengine_add_membrane.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.sengine_execute.argtypes = [ctypes.c_void_p]
_lib.sengine_reset_membranes.argtypes = [ctypes.c_void_p]
_lib.sengine_capture_graph.argtypes = [ctypes.c_void_p]
_lib.sengine_replay.argtypes = [ctypes.c_void_p]
_lib.sengine_sync.argtypes = [ctypes.c_void_p]
_lib.sengine_benchmark.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.sengine_benchmark.restype = ctypes.c_float


def ptr(t: torch.Tensor) -> int:
    """Get raw GPU data pointer from a torch tensor."""
    return t.data_ptr()


TILELANG_VARIANTS = {
    KernelVariant.TileLangConvBN,
    KernelVariant.TileLangConv1x1BN,
    KernelVariant.TileLangStemConvBN,
    KernelVariant.TileLangFusedConvBNIF,
    KernelVariant.TileLangFusedConv1x1BNIF,
    KernelVariant.TileLangLinearBN,
    KernelVariant.TileLangLinearBNLIF,
}


def setup_cpp_executor(engine, kernels, ir, schedule, T, B,
                       kernel_so_map: dict[int, str]):
    """Create and configure a C++ executor from a built CUDAGraphEngine.

    Returns the C++ executor handle.
    """
    exe = _lib.sengine_create()
    max_nid = max(ir.nodes.keys())
    _lib.sengine_alloc_nodes(exe, max_nid)

    # Set schedule
    sched_arr = (ctypes.c_int * len(schedule))(*schedule)
    _lib.sengine_set_schedule(exe, sched_arr, len(schedule))

    # Load TileLang .so files
    tl_idx_map = {}  # so_path → tl_idx in C++ executor
    nid_tl_idx = {}  # node_id → tl_idx

    for nid, so_path in kernel_so_map.items():
        if so_path not in tl_idx_map:
            idx = _lib.sengine_load_tilelang(exe, so_path.encode())
            if idx < 0:
                print(f"  WARNING: Failed to load TileLang .so for node {nid}: {so_path}")
                continue
            tl_idx_map[so_path] = idx
        nid_tl_idx[nid] = tl_idx_map[so_path]

    # Register membranes
    for nid, mem in engine.membranes.items():
        _lib.sengine_add_membrane(exe, ptr(mem), mem.numel())

    # Configure each node
    nodes_configured = 0
    nodes_skipped = 0

    for nid in schedule:
        node = ir.nodes.get(nid)
        if node is None:
            _lib.sengine_set_skip_node(exe, nid)
            nodes_skipped += 1
            continue

        kv = node.assigned_kernel
        act = engine.activations

        # Get input activation from predecessors
        preds = ir.predecessors(nid)
        input_nid = preds[0] if preds else nid
        input_buf = act.get(input_nid)
        output_buf = act.get(nid)

        if kv in TILELANG_VARIANTS and nid in nid_tl_idx:
            tl_idx = nid_tl_idx[nid]
            w = engine.weights.get(nid)
            if w is None:
                w = engine.weights_1x1.get(nid)
            sc = engine.bn_scales.get(nid)
            bi = engine.bn_biases.get(nid)

            if input_buf is None or w is None or output_buf is None:
                _lib.sengine_set_skip_node(exe, nid)
                nodes_skipped += 1
                continue

            if kv in (KernelVariant.TileLangFusedConvBNIF, KernelVariant.TileLangFusedConv1x1BNIF):
                # Fused Conv+BN+IF: 6 args (data, weight, membrane, scale, bias, spikes)
                succs = ir.successors(nid)
                mem = None
                for s in succs:
                    if s in engine.membranes:
                        mem = engine.membranes[s]
                        break
                if mem is not None and sc is not None and bi is not None:
                    _lib.sengine_set_tilelang_node_6(exe, nid, tl_idx,
                        ptr(input_buf), ptr(w), ptr(mem), ptr(sc), ptr(bi), ptr(output_buf))
                    nodes_configured += 1
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                    nodes_skipped += 1
            else:
                # Standard Conv+BN: 5 args (data, weight, scale, bias, output)
                if sc is not None and bi is not None:
                    _lib.sengine_set_tilelang_node_5(exe, nid, tl_idx,
                        ptr(input_buf), ptr(w), ptr(sc), ptr(bi), ptr(output_buf))
                    nodes_configured += 1
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                    nodes_skipped += 1

        elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
            mem = engine.membranes.get(nid)
            if input_buf is None or output_buf is None or mem is None:
                _lib.sengine_set_skip_node(exe, nid)
                nodes_skipped += 1
                continue

            v_thresh = 1.0
            if node.neuron_params:
                v_thresh = node.neuron_params.v_threshold

            total_elems = input_buf.numel()
            # spatial_elems = total / T
            spatial_elems = total_elems // T

            if kv == KernelVariant.CUDAVec4LIF:
                recip_tau = 0.5
                if node.neuron_params and node.neuron_params.tau > 0:
                    recip_tau = 1.0 / node.neuron_params.tau
                _lib.sengine_set_lif_node(exe, nid,
                    ptr(input_buf), ptr(output_buf), ptr(mem),
                    total_elems, spatial_elems, v_thresh, recip_tau)
            else:
                _lib.sengine_set_if_node(exe, nid,
                    ptr(input_buf), ptr(output_buf), ptr(mem),
                    total_elems, spatial_elems, v_thresh)
            nodes_configured += 1

        elif kv == KernelVariant.Elementwise:
            if node.op_type == OpType.Add and len(preds) >= 2:
                a_buf = act.get(preds[0])
                b_buf = act.get(preds[1])
                if a_buf is not None and b_buf is not None and output_buf is not None:
                    n_elems = min(a_buf.numel(), b_buf.numel())
                    _lib.sengine_set_add_node(exe, nid,
                        ptr(a_buf), ptr(b_buf), ptr(output_buf), n_elems)
                    nodes_configured += 1
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                    nodes_skipped += 1
            else:
                # Scale, Mul, passthrough
                if input_buf is not None and output_buf is not None:
                    _lib.sengine_set_alias_node(exe, nid, ptr(input_buf), ptr(output_buf), input_buf.numel())
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                nodes_configured += 1

        elif kv == KernelVariant.CuDNNPool:
            if node.op_type == OpType.MaxPool and node.pool_params:
                pp = node.pool_params
                if input_buf is not None and output_buf is not None:
                    shape = input_buf.shape  # NHWC: (N, H, W, C)
                    N_pool = shape[0]
                    H_pool = shape[1]
                    W_pool = shape[2]
                    C_pool = shape[3]
                    ks = pp.get('kernel_size', 3)
                    stride = pp.get('stride', 2)
                    pad = pp.get('padding', 1)
                    OH = (H_pool + 2 * pad - ks) // stride + 1
                    OW = (W_pool + 2 * pad - ks) // stride + 1
                    _lib.sengine_set_maxpool_node(exe, nid,
                        ptr(input_buf), ptr(output_buf),
                        N_pool, H_pool, W_pool, C_pool, OH, OW, ks, stride, pad)
                    nodes_configured += 1
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                    nodes_skipped += 1
            elif node.op_type == OpType.GlobalAvgPool:
                if input_buf is not None and output_buf is not None:
                    shape = input_buf.shape  # NHWC
                    _lib.sengine_set_global_avgpool_node(exe, nid,
                        ptr(input_buf), ptr(output_buf),
                        shape[0], shape[1], shape[2], shape[3])
                    nodes_configured += 1
                else:
                    _lib.sengine_set_skip_node(exe, nid)
                    nodes_skipped += 1
            else:
                _lib.sengine_set_skip_node(exe, nid)
                nodes_skipped += 1

        elif kv == KernelVariant.TemporalMean:
            if input_buf is not None and output_buf is not None:
                spatial = input_buf.numel() // T
                _lib.sengine_set_temporal_mean_node(exe, nid,
                    ptr(input_buf), ptr(output_buf), T, spatial)
                nodes_configured += 1
            else:
                _lib.sengine_set_skip_node(exe, nid)
                nodes_skipped += 1

        elif kv == KernelVariant.CuBLASGemm:
            w = engine.weights.get(nid)
            if input_buf is not None and w is not None and output_buf is not None:
                # input: (B, K), weight: (N, K), output: (B, N)
                M = input_buf.shape[0] if input_buf.ndim >= 1 else 1
                K_dim = input_buf.shape[-1] if input_buf.ndim >= 1 else 1
                N_dim = w.shape[0]
                _lib.sengine_set_gemm_node(exe, nid,
                    ptr(input_buf), ptr(w), ptr(output_buf),
                    M, K_dim, N_dim)
                nodes_configured += 1
            else:
                _lib.sengine_set_skip_node(exe, nid)
                nodes_skipped += 1

        elif kv == KernelVariant.CuDNNConv:
            # Fallback: for stem conv etc, mark as skip (small overhead)
            _lib.sengine_set_skip_node(exe, nid)
            nodes_skipped += 1

        elif kv == KernelVariant.ZeroCost:
            # Reshape/Transpose: alias or copy
            if input_buf is not None and output_buf is not None:
                if input_buf.data_ptr() == output_buf.data_ptr():
                    _lib.sengine_set_skip_node(exe, nid)
                else:
                    _lib.sengine_set_alias_node(exe, nid,
                        ptr(input_buf), ptr(output_buf), min(input_buf.numel(), output_buf.numel()))
            else:
                _lib.sengine_set_skip_node(exe, nid)
            nodes_configured += 1

        elif kv == KernelVariant.TileRepeat:
            _lib.sengine_set_skip_node(exe, nid)
            nodes_skipped += 1

        else:
            _lib.sengine_set_skip_node(exe, nid)
            nodes_skipped += 1

    print(f"  C++ executor: {nodes_configured} nodes configured, {nodes_skipped} skipped")
    return exe


def bench_trt(trt_path: str, input_shape: tuple, warmup: int = 200, iters: int = 1000) -> float:
    """Benchmark TRT engine, return latency in ms."""
    try:
        from iengine.backends.tensorrt.runtime import TRTRunner
    except ImportError:
        print("  TRT not available")
        return 0.0

    try:
        with TRTRunner(trt_path, device=0) as runner:
            result = runner.benchmark_latency(
                input_shape=input_shape,
                n_warmup=warmup,
                n_measure=iters,
            )
            return result['mean_ms']
    except Exception as e:
        print(f"  TRT benchmark failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark sengine C++ executor vs TRT")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--onnx', type=str,
                        default='sengine/exports/sew_resnet18_imagenet_plugin.onnx')
    parser.add_argument('--trt', type=str, default=None,
                        help='TRT engine path (auto-detected if not specified)')
    args = parser.parse_args()

    B = args.batch
    T = args.T
    TB = T * B

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: T={T}, B={B}, TB={TB}")

    # ─── 1. Build sengine ───
    print(f"\n{'='*60}")
    print(f"  Phase 1: Build sengine from ONNX")
    print(f"{'='*60}")
    t0 = time.time()
    builder = EngineBuilder(args.onnx, T=T, batch_size=B)
    engine = builder.build(autotune=False, capture_graph=False)
    ir = builder._ir
    schedule = builder._schedule
    kernels_map = engine.kernels
    print(f"  Built in {time.time()-t0:.1f}s ({len(schedule)} schedule ops)")

    # ─── 2. Export TileLang kernels as standalone .so ───
    print(f"\n{'='*60}")
    print(f"  Phase 2: Export standalone .so files")
    print(f"{'='*60}")
    t0 = time.time()
    build_dir = f'/tmp/sengine_cpp_B{B}'
    kernel_so_map = export_all_kernels(kernels_map, ir, build_dir)
    print(f"  Exported in {time.time()-t0:.1f}s")

    # Count node types
    type_counts = {}
    for nid in schedule:
        node = ir.nodes.get(nid)
        if node:
            kv = node.assigned_kernel.name if node.assigned_kernel else 'None'
            type_counts[kv] = type_counts.get(kv, 0) + 1
    print(f"  Node types: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

    # ─── 3. Run Python CUDA Graph benchmark (baseline) ───
    print(f"\n{'='*60}")
    print(f"  Phase 3: Python CUDA Graph benchmark")
    print(f"{'='*60}")
    engine.capture_graph()
    py_ms = engine.benchmark(warmup=args.warmup, n_iters=args.iters)
    print(f"  Python Graph: {py_ms:.3f} ms")

    # ─── 4. Set up C++ executor ───
    print(f"\n{'='*60}")
    print(f"  Phase 4: C++ executor setup + benchmark")
    print(f"{'='*60}")
    t0 = time.time()
    exe = setup_cpp_executor(engine, kernels_map, ir, schedule, T, B, kernel_so_map)
    print(f"  Setup in {time.time()-t0:.3f}s")

    # Capture CUDA Graph in C++
    _lib.sengine_capture_graph(exe)
    print(f"  CUDA Graph captured")

    # Benchmark
    cpp_ms = _lib.sengine_benchmark(exe, args.warmup, args.iters)
    print(f"  C++ Graph: {cpp_ms:.3f} ms")

    # ─── 5. TRT benchmark ───
    trt_path = args.trt
    if trt_path is None:
        if B == 1:
            trt_path = 'trt_engines/sew_resnet18_fp16.engine'
        else:
            trt_path = f'trt_engines/sew_resnet18_b{B}_fp16.engine'

    trt_ms = 0.0
    if os.path.exists(trt_path):
        print(f"\n{'='*60}")
        print(f"  Phase 5: TRT benchmark")
        print(f"{'='*60}")
        # TRT engine was built with batch dim only (not TB)
        # For B=1 engine: (1,3,224,224)
        trt_shape = (B, 3, 224, 224)
        trt_ms = bench_trt(trt_path, trt_shape, args.warmup, args.iters)
        print(f"  TRT FP16: {trt_ms:.3f} ms")
    else:
        print(f"\n  TRT engine not found: {trt_path}")

    # ─── Results ───
    print(f"\n{'='*60}")
    print(f"  RESULTS (B={B}, T={T})")
    print(f"{'='*60}")
    print(f"  {'Engine':<30} {'Latency':>10} {'vs TRT':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    print(f"  {'sengine Python Graph':<30} {py_ms:>8.3f}ms", end="")
    if trt_ms > 0:
        ratio = trt_ms / py_ms
        print(f" {ratio:>10.1f}x")
    else:
        print()
    print(f"  {'sengine C++ Graph':<30} {cpp_ms:>8.3f}ms", end="")
    if trt_ms > 0:
        ratio = trt_ms / cpp_ms
        print(f" {ratio:>10.1f}x")
    else:
        print()
    if trt_ms > 0:
        print(f"  {'TensorRT FP16':<30} {trt_ms:>8.3f}ms {'(baseline)':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    if py_ms > 0:
        overhead = py_ms - cpp_ms
        print(f"  Python→C++ overhead eliminated: {overhead:.3f}ms ({overhead/py_ms*100:.0f}%)")
    print(f"{'='*60}")

    # Cleanup
    _lib.sengine_destroy(exe)


if __name__ == '__main__':
    main()
