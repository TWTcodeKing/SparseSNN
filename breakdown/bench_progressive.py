#!/usr/bin/env python3
"""Progressive breakdown: quantify DRAM traffic and latency hiding contributions.

Supports Conv1x1, Conv3x3, and MatMul+LIF (transformers) fused layers.

Conv1x1 / MatMul+LIF variants:
  1. Decomposed  — separate GEMM+BN kernel + IF kernel, intermediate via DRAM
  2. Fused-PS    — single kernel, all-T GEMM → all-T IF via smem (no interleaving)
  3. Production  — interleaved per-T GEMM→IF, acc stays in registers

Conv3x3 variants (Fused-PS skipped — im2col + T*smem too large):
  1. Decomposed  — separate Conv+BN kernel + IF kernel, intermediate via DRAM
  2. Production  — interleaved per-T Conv→IF with im2col

Usage:
    python breakdown/bench_progressive.py --sengine GPUtil/engines/sew_resnet101_B32.sengine --gpu-id 2
    python breakdown/bench_progressive.py --sengine GPUtil/engines/spikformer_4_512_B4.sengine --gpu-id 2
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import tilelang
import tilelang.language as T
import torch


# ═══════════════════════════════════════════════════════════════════
# Standalone IF neuron (decomposed baseline for all layer types)
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def standalone_if(
    M_total, F, spatial,
    block_M, block_N, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    """IF neuron: reads BN output from DRAM, writes spikes to DRAM."""
    @T.prim_func
    def main(
        intermediate: T.Tensor((M_total, F), io_dtype),
        state:        T.Tensor((spatial, F), T.float32),
        output:       T.Tensor((M_total, F), io_dtype),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M_total, block_M),
                      threads=threads) as (bx, by):
            in_s = T.alloc_shared((block_M, block_N), io_dtype)
            os_  = T.alloc_shared((block_M, block_N), io_dtype)
            T.copy(intermediate[by * block_M, bx * block_N], in_s)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M_total and f < F:
                    s_idx = m % spatial
                    v = state[s_idx, f]
                    h = v + T.cast(in_s[i, j], T.float32)
                    sp = T.if_then_else(h >= T.float32(v_threshold),
                                        T.float32(1), T.float32(0))
                    state[s_idx, f] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                    os_[i, j] = T.cast(sp, io_dtype)
            T.copy(os_, output[by * block_M, bx * block_N])
    return main


# ═══════════════════════════════════════════════════════════════════
# Conv1x1: Fused Phase-Separated (smem y_buf, no interleaving)
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def fused_phase_separated(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, S=1, v_threshold=1.0, v_reset=0.0,
):
    TB = T_steps * B
    OH = (H + S - 1) // S; OW = (W + S - 1) // S
    M = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), io_dtype),
        weight:   T.Tensor((C_in, F), io_dtype),
        state:    T.Tensor((M, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        output:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        o = T.Tensor((TB * OH * OW, F), io_dtype, output.data)
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            ds  = T.alloc_shared((block_M, block_K), io_dtype)
            ws  = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            y_buf = T.alloc_shared((T_steps * block_M, block_N), io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                m_ = by * block_M + i; f = bx * block_N + j
                if m_ < M and f < F:
                    mem[i, j] = state[m_, f]
                else:
                    mem[i, j] = T.float32(0)

            for t_step in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K),
                                       num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m_ = by * block_M + i
                        n = t_step * B + m_ // (OH * OW); hw = m_ % (OH * OW)
                        oh = hw // OW; ow = hw % OW
                        ds[i, j] = T.if_then_else(
                            (m_ < M) and (c < C_in),
                            data[n, oh * S, ow * S, c], T.cast(0, io_dtype))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m_ = by * block_M + i; f = bx * block_N + j
                    if m_ < M and f < F:
                        y_buf[t_step * block_M + i, j] = T.cast(
                            acc[i, j] * bn_scale[f] + bn_bias[f], io_dtype)

            for t_step in range(T_steps):
                for i, j in T.Parallel(block_M, block_N):
                    m_ = by * block_M + i; f = bx * block_N + j
                    if m_ < M and f < F:
                        y = T.cast(y_buf[t_step * block_M + i, j], T.float32)
                        h = mem[i, j] + y
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        os_[i, j] = T.cast(sp, io_dtype)
                T.copy(os_, o[t_step * M + by * block_M, bx * block_N])

            for i, j in T.Parallel(block_M, block_N):
                m_ = by * block_M + i; f = bx * block_N + j
                if m_ < M and f < F:
                    state[m_, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Conv3x3: Fused Phase-Separated (smem y_buf, im2col, no interleaving)
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def fused_phase_separated_3x3(
    B, C_in, H, W, F, KH, S, D, P, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    KW = KH
    TB = T_steps * B
    OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1
    M = B * OH * OW
    K_red = KH * KW * C_in

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), io_dtype),
        weight:   T.Tensor((KH, KW, C_in, F), io_dtype),
        state:    T.Tensor((M, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        output:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        w_flat = T.Tensor((K_red, F), io_dtype, weight.data)
        o_flat = T.Tensor((TB * OH * OW, F), io_dtype, output.data)
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            ds  = T.alloc_shared((block_M, block_K), io_dtype)
            ws  = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            y_buf = T.alloc_shared((T_steps * block_M, block_N), io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                m_ = by * block_M + i; f = bx * block_N + j
                if m_ < M and f < F:
                    mem[i, j] = state[m_, f]
                else:
                    mem[i, j] = T.float32(0)

            # Phase 1: ALL GEMM (im2col) → smem y_buf
            for t_step in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(K_red, block_K),
                                       num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        k = ki * block_K + j; m_ = by * block_M + i
                        n_idx = t_step * B + m_ // (OH * OW)
                        hw = m_ % (OH * OW)
                        oh = hw // OW; ow = hw % OW
                        kh = k // (KW * C_in)
                        kw = (k // C_in) % KW
                        cin = k % C_in
                        ih = oh * S + kh * D - P
                        iw = ow * S + kw * D - P
                        ib = ((ih >= 0) and (iw >= 0) and (ih < H) and (iw < W)
                              and (m_ < M) and (k < K_red))
                        ds[i, j] = T.if_then_else(ib, data[n_idx, ih, iw, cin],
                                                   T.cast(0, io_dtype))
                    T.copy(w_flat[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m_ = by * block_M + i; f = bx * block_N + j
                    if m_ < M and f < F:
                        y_buf[t_step * block_M + i, j] = T.cast(
                            acc[i, j] * bn_scale[f] + bn_bias[f], io_dtype)

            # Phase 2: ALL IF from smem y_buf
            for t_step in range(T_steps):
                for i, j in T.Parallel(block_M, block_N):
                    m_ = by * block_M + i; f = bx * block_N + j
                    if m_ < M and f < F:
                        y = T.cast(y_buf[t_step * block_M + i, j], T.float32)
                        h = mem[i, j] + y
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        os_[i, j] = T.cast(sp, io_dtype)
                T.copy(os_, o_flat[t_step * M + by * block_M, bx * block_N])

            for i, j in T.Parallel(block_M, block_N):
                m_ = by * block_M + i; f = bx * block_N + j
                if m_ < M and f < F:
                    state[m_, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Fused-PS with automatic tile shrinking on smem overflow
# ═══════════════════════════════════════════════════════════════════

def _try_fused_ps(build_fn, run_fn_factory, bM, bN, bK, ns, thr, T_steps):
    """Try to build and run fused-PS, shrinking block_M on smem failure.

    y_buf smem = T * block_M * block_N * 2 bytes — shrink block_M to fit.
    Returns (kernel_callable, adjusted_bM) or raises if all attempts fail.
    """
    cur_bM = bM
    while cur_bM >= 8:
        try:
            kern = build_fn(cur_bM, bN, bK, ns, thr)
            fn = run_fn_factory(kern)
            fn()  # trigger runtime smem errors
            torch.cuda.synchronize()
            return kern, fn, cur_bM
        except Exception:
            cur_bM //= 2
            torch.cuda.empty_cache()
    raise RuntimeError(f"fused-PS: all tile sizes failed (smallest bM=8)")


# ═══════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════

def bench(fn, warmup=50, reps=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    lo, hi = int(len(times) * 0.05), int(len(times) * 0.95)
    t = times[lo:hi] if hi > lo else times
    m = sum(t) / len(t)
    return m


# ═══════════════════════════════════════════════════════════════════
# Layer extraction
# ═══════════════════════════════════════════════════════════════════

def extract_layers(sengine_path, top_n=10):
    """Extract top-N fused GEMM+neuron layers from .sengine."""
    from sengine.build.sengine_io import load_sengine
    from sengine.ir import KernelVariant

    ir, schedule, T_val, B_val = load_sengine(sengine_path)

    layers = []
    for nid in schedule:
        node = ir.nodes.get(nid)
        if not (node and node.tilelang_config):
            continue
        cfg = node.tilelang_config
        kv = node.assigned_kernel

        if kv == KernelVariant.TileLangFusedConv1x1BNIF:
            cp = node.conv_params
            if not cp or cp.kernel_h != 1:
                continue
            shape = node.input_shapes[0] if node.input_shapes else ()
            if len(shape) < 3:
                continue
            layers.append({
                'kind': 'conv1x1',
                'nid': nid,
                'C_in': cp.in_channels, 'C_out': cp.out_channels,
                'H': shape[1], 'W': shape[2],
                'stride': cp.stride_h, 'pad': cp.pad_h,
                'kernel_size': 1, 'dilation': 1,
                'block_M': cfg['block_M'], 'block_N': cfg['block_N'],
                'block_K': cfg['block_K'], 'num_stages': cfg['num_stages'],
                'threads': cfg['threads'],
                'est_us': node.est_latency_us,
            })

        elif kv == KernelVariant.TileLangFusedConvBNIF:
            cp = node.conv_params
            if not cp:
                continue
            shape = node.input_shapes[0] if node.input_shapes else ()
            if len(shape) < 3:
                continue
            # Skip stem convolutions (C_in < 4)
            if cp.in_channels < 4:
                continue
            layers.append({
                'kind': f'conv{cp.kernel_h}x{cp.kernel_w}',
                'nid': nid,
                'C_in': cp.in_channels, 'C_out': cp.out_channels,
                'H': shape[1], 'W': shape[2],
                'stride': cp.stride_h, 'pad': cp.pad_h,
                'kernel_size': cp.kernel_h, 'dilation': cp.dilation_h,
                'block_M': cfg['block_M'], 'block_N': cfg['block_N'],
                'block_K': cfg['block_K'], 'num_stages': cfg['num_stages'],
                'threads': cfg['threads'],
                'est_us': node.est_latency_us,
            })

        elif kv == KernelVariant.TileLangFusedMatMulLIF:
            if len(node.input_shapes) < 2:
                continue
            s0 = node.input_shapes[0]
            s1 = node.input_shapes[1]
            if len(s1) != 2:
                continue
            K = s1[0]; N = s1[1]
            M = 1
            for d in s0:
                M *= d
            M = M // K if K > 0 else M
            spatial = M // T_val
            # Map to conv1x1 with H=spatial/B, W=1
            spatial_per_sample = spatial // B_val if B_val > 0 else spatial
            layers.append({
                'kind': 'matmul',
                'nid': nid,
                'C_in': K, 'C_out': N,
                'H': spatial_per_sample, 'W': 1,
                'stride': 1, 'pad': 0,
                'kernel_size': 1, 'dilation': 1,
                'block_M': cfg['block_M'], 'block_N': cfg['block_N'],
                'block_K': cfg['block_K'], 'num_stages': cfg['num_stages'],
                'threads': cfg['threads'],
                'est_us': node.est_latency_us,
            })

        elif kv == KernelVariant.TileLangFusedGroupedConvBNLIF:
            cp = node.conv_params
            if not cp:
                continue
            shape = node.input_shapes[0] if node.input_shapes else ()
            if len(shape) < 3:
                continue
            layers.append({
                'kind': f'gconv{cp.kernel_h}x{cp.kernel_w}',
                'nid': nid,
                'C_in': cp.in_channels, 'C_out': cp.out_channels,
                'H': shape[1], 'W': shape[2],
                'stride': cp.stride_h, 'pad': cp.pad_h,
                'kernel_size': cp.kernel_h, 'dilation': cp.dilation_h,
                'groups': cp.groups,
                'block_M': cfg['block_M'], 'block_N': cfg['block_N'],
                'block_K': cfg['block_K'], 'num_stages': cfg['num_stages'],
                'threads': cfg['threads'],
                'est_us': node.est_latency_us,
            })

    layers.sort(key=lambda x: x['est_us'], reverse=True)
    return layers, T_val, B_val


# ═══════════════════════════════════════════════════════════════════
# Per-layer benchmarking
# ═══════════════════════════════════════════════════════════════════

def bench_layer(layer, T_val, B_val):
    """Benchmark decomposed vs fused-PS vs production for one layer."""
    from sengine.kernels.conv2d_bn_if_t4 import (
        conv1x1_bn_t4_kernel, conv2d_bn_t4_kernel)
    from sengine.kernels.interleaved_templates import conv1x1_bn_if
    from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_if_interleaved_kernel

    kind = layer['kind']
    C_in = layer['C_in']; F = layer['C_out']
    H = layer['H']; W = layer['W']; S = layer['stride']
    P = layer['pad']; K = layer['kernel_size']; D = layer['dilation']
    bM = layer['block_M']; bN = layer['block_N']; bK = layer['block_K']
    ns = layer['num_stages']; thr = layer['threads']
    TB = T_val * B_val

    dev = "cuda"
    results = {}

    if kind in ('conv1x1', 'matmul'):
        OH = (H + S - 1) // S; OW = (W + S - 1) // S
        M_spatial = B_val * OH * OW

        data     = torch.randn(TB, H, W, C_in, device=dev, dtype=torch.float16)
        weight   = torch.randn(C_in, F, device=dev, dtype=torch.float16)
        bn_scale = torch.ones(F, device=dev, dtype=torch.float32)
        bn_bias  = torch.zeros(F, device=dev, dtype=torch.float32)
        state    = torch.zeros(M_spatial, F, device=dev, dtype=torch.float32)

        # Decomposed: conv1x1_bn (GEMM+BN → DRAM) + standalone IF (DRAM → spikes)
        gk = conv1x1_bn_t4_kernel(TB, C_in, H, W, F, S, bM, bN, bK, ns, thr)
        lk = standalone_if(TB * OH * OW, F, M_spatial, bM, bN, thr)
        inter = gk(data, weight, bn_scale, bn_bias)
        inter_flat = inter.reshape(TB * OH * OW, F)
        results['gemm'] = bench(lambda: gk(data, weight, bn_scale, bn_bias))
        results['lif'] = bench(lambda: lk(inter_flat, state))
        def run_decomposed():
            o = gk(data, weight, bn_scale, bn_bias)
            lk(o.reshape(TB * OH * OW, F), state)
        results['decomposed'] = bench(run_decomposed)

        # Fused-PS with auto tile shrinking on smem overflow
        def _build_ps(bM_, bN_, bK_, ns_, thr_):
            return fused_phase_separated(B_val, C_in, H, W, F, T_val,
                                          bM_, bN_, bK_, ns_, thr_, S=S)
        def _run_factory(kern):
            def fn():
                state.zero_()
                kern(data, weight, state, bn_scale, bn_bias)
            return fn
        _, run_ps, _ = _try_fused_ps(_build_ps, _run_factory, bM, bN, bK, ns, thr, T_val)
        results['fused_ps'] = bench(run_ps)

        # Production (interleaved)
        pk = conv1x1_bn_if(B_val, C_in, H, W, F, T_val, bM, bN, bK, ns, thr, S=S)
        def run_prod():
            state.zero_()
            pk(data, weight, state, bn_scale, bn_bias)
        results['production'] = bench(run_prod)

        del data, weight, bn_scale, bn_bias, state
        torch.cuda.empty_cache()

    elif kind.startswith('conv'):
        # 3x3 or general NxN conv
        OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
        M_spatial = B_val * OH * OW

        data     = torch.randn(TB, H, W, C_in, device=dev, dtype=torch.float16)
        weight   = torch.randn(K, K, C_in, F, device=dev, dtype=torch.float16)
        bn_scale = torch.ones(F, device=dev, dtype=torch.float32)
        bn_bias  = torch.zeros(F, device=dev, dtype=torch.float32)
        state    = torch.zeros(M_spatial, F, device=dev, dtype=torch.float32)

        # Decomposed: conv_bn (GEMM+BN → DRAM) + standalone IF
        gk = conv2d_bn_t4_kernel(TB, C_in, H, W, F, K, S, D, P, bM, bN, bK, ns, thr)
        lk = standalone_if(TB * OH * OW, F, M_spatial, bM, bN, thr)
        inter = gk(data, weight, bn_scale, bn_bias)
        inter_flat = inter.reshape(TB * OH * OW, F)
        results['gemm'] = bench(lambda: gk(data, weight, bn_scale, bn_bias))
        results['lif'] = bench(lambda: lk(inter_flat, state))
        def run_decomposed():
            o = gk(data, weight, bn_scale, bn_bias)
            lk(o.reshape(TB * OH * OW, F), state)
        results['decomposed'] = bench(run_decomposed)

        # Fused-PS (im2col version) with auto tile shrinking
        def _build_ps_3x3(bM_, bN_, bK_, ns_, thr_):
            return fused_phase_separated_3x3(B_val, C_in, H, W, F, K, S, D, P, T_val,
                                              bM_, bN_, bK_, ns_, thr_)
        def _run_factory_3x3(kern):
            def fn():
                state.zero_()
                kern(data, weight, state, bn_scale, bn_bias)
            return fn
        _, run_ps, _ = _try_fused_ps(_build_ps_3x3, _run_factory_3x3, bM, bN, bK, ns, thr, T_val)
        results['fused_ps'] = bench(run_ps)

        # Production (interleaved with im2col)
        pk = conv2d_bn_if_interleaved_kernel(
            B_val, C_in, H, W, F, K, S, D, P, T_val, bM, bN, bK, ns, thr)
        def run_prod():
            state.zero_()
            pk(data, weight, state, bn_scale, bn_bias)
        results['production'] = bench(run_prod)

        del data, weight, bn_scale, bn_bias, state
        torch.cuda.empty_cache()

    else:
        # Unsupported kind — return zeros
        results = {'gemm': 0, 'lif': 0, 'decomposed': 0,
                   'fused_ps': float('nan'), 'production': 0}

    return results


def layer_label(layer):
    kind = layer['kind']
    if kind == 'matmul':
        return f"{layer['C_in']}→{layer['C_out']} M{layer['H']*layer['W']*1}"  # spatial
    else:
        return f"{layer['C_in']}→{layer['C_out']} {layer['H']}²"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Progressive kernel breakdown")
    parser.add_argument('--sengine', type=str, required=True,
                        help='Path to .sengine file')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    gpu_name = torch.cuda.get_device_name(args.gpu_id)
    print(f"GPU {args.gpu_id}: {gpu_name}")

    all_candidates, T_val, B_val = extract_layers(args.sengine, args.top)
    model_name = os.path.splitext(os.path.basename(args.sengine))[0]
    print(f"Model: {model_name}, T={T_val}, B={B_val}")
    print(f"Found {len(all_candidates)} candidate layers, profiling top {args.top} ...\n")

    # Profile layers, skipping any that crash (OOM, smem overflow, etc.)
    layers = []
    all_results = []
    for i, layer in enumerate(all_candidates):
        if len(all_results) >= args.top:
            break
        label = layer_label(layer)
        kind = layer['kind']
        print(f"  [{len(all_results)+1}/{args.top}] {kind:>7} {label} ...", end="", flush=True)
        try:
            r = bench_layer(layer, T_val, B_val)
            # Skip layers where fusion is worse than decomposed
            if r['production'] >= r['decomposed']:
                print(f" SKIP (fusion slower: prod {r['production']:.3f} >= decomp {r['decomposed']:.3f}ms)")
                torch.cuda.empty_cache()
                continue
            layers.append(layer)
            all_results.append(r)
            print(" OK")
        except Exception as e:
            print(f" SKIP ({e})")
            torch.cuda.empty_cache()
            continue

    # Print clean table
    print()
    print(f"{'#':<3} {'Kind':<8} {'Layer':<20} {'Tile':<16} "
          f"{'GEMM':>8} {'LIF':>8} {'G+L':>8} │ "
          f"{'Fused-PS':>9} {'Prod':>9} │ "
          f"{'ΔDRAM':>8} {'ΔIntlv':>8}")
    print("─" * 135)

    total = {'gemm': 0, 'lif': 0, 'decomposed': 0, 'fused_ps': 0, 'production': 0}

    for i, (layer, r) in enumerate(zip(layers, all_results)):
        label = layer_label(layer)
        tile  = f"{layer['block_M']}×{layer['block_N']}×{layer['block_K']}"
        kind  = layer['kind']

        total['gemm'] += r['gemm']
        total['lif'] += r['lif']
        total['decomposed'] += r['decomposed']
        total['fused_ps'] += r['fused_ps']
        total['production'] += r['production']

        d_dram = r['decomposed'] - r['fused_ps']
        d_intlv = r['fused_ps'] - r['production']

        print(f"{i+1:<3} {kind:<8} {label:<20} {tile:<16} "
              f"{r['gemm']:>7.3f}  {r['lif']:>7.3f}  {r['decomposed']:>7.3f} │ "
              f"{r['fused_ps']:>8.3f}  {r['production']:>8.3f} │ "
              f"{d_dram:>7.3f}  {d_intlv:>7.3f}")

    # Totals
    print("─" * 135)
    n = len(all_results)
    d_dram_t = total['decomposed'] - total['fused_ps']
    d_intlv_t = total['fused_ps'] - total['production']
    d_total = total['decomposed'] - total['production']

    print(f"{'Σ':<3} {'':8} {f'({n} layers)':<20} {'':16} "
          f"{total['gemm']:>7.3f}  {total['lif']:>7.3f}  {total['decomposed']:>7.3f} │ "
          f"{total['fused_ps']:>8.3f}  {total['production']:>8.3f} │ "
          f"{d_dram_t:>7.3f}  {d_intlv_t:>7.3f}")

    # Summary
    if d_total > 0:
        print(f"\n  Decomposed → Production: {d_total:.3f}ms saved, "
              f"{total['decomposed']/total['production']:.2f}× ({n} layers)")
        print(f"    ① DRAM traffic elimination (G+L → Fused-PS):      "
              f"{d_dram_t:>7.3f}ms  ({d_dram_t/d_total*100:>5.1f}%)")
        print(f"    ② Interleaved temporal fusion (Fused-PS → Prod):   "
              f"{d_intlv_t:>7.3f}ms  ({d_intlv_t/d_total*100:>5.1f}%)")

        lif_frac = total['lif'] / total['decomposed'] * 100 if total['decomposed'] > 0 else 0
        lif_hidden = ((total['gemm'] + total['lif'] - total['production'])
                      / total['lif'] * 100 if total['lif'] > 0 else 0)
        print(f"\n  LIF fraction of decomposed: {lif_frac:.1f}%")
        print(f"  LIF hidden ratio: {lif_hidden:.1f}% of standalone LIF cost absorbed into GEMM")


if __name__ == '__main__':
    main()
