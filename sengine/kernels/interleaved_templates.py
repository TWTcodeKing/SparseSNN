"""Interleaved kernel templates for compute+memory fusion.

Per-CTA T-loop kernels that fuse compute-bound ops (Conv, MatMul) with
memory-bound epilogue chains (BN, IF, LIF, Add, MaxPool).

Each template:
  - GEMM prologue: pipelined K-reduction (Conv1x1 or Conv3x3 im2col)
  - Epilogue: element-wise chain processed per-tile in registers
  - T-loop: each CTA owns its spatial tile across all T timesteps
  - Membrane: persists in fragment registers across T iterations

Template naming: conv1x1_bn_{epilogue_chain}_kernel
  Epilogue chain tokens: if, lif, add, maxpool, avgpool

Data layout: NHWC. Accumulation: FP32. Input/output: FP16. Membrane: FP32.
"""

import tilelang
import tilelang.language as T


# ═══════════════════════════════════════════════════════════════════
# Template 1: Conv1x1 + BN + IF
# Pattern: Conv → BN → IF(membrane) → spike
# Args: data, weight, membrane, bn_scale, bn_bias → output
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_if(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, S=1, v_threshold=1.0, v_reset=0.0,
):
    TB = T_steps * B
    OH = (H + S - 1) // S
    OW = (W + S - 1) // S
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
            ds = T.alloc_shared((block_M, block_K), io_dtype)
            ws = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: mem[i, j] = state[m, f]
                else: mem[i, j] = T.float32(0)
            for t in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m = by * block_M + i
                        n = t * B + m // (OH * OW); hw = m % (OH * OW)
                        oh = hw // OW; ow = hw % OW
                        ds[i, j] = T.if_then_else(
                            (m < M) and (c < C_in),
                            data[n, oh * S, ow * S, c], io_dtype(0))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i; f = bx * block_N + j
                    if m < M and f < F:
                        bn = acc[i, j] * bn_scale[f] + bn_bias[f]
                        h = mem[i, j] + bn
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        os_[i, j] = T.cast(sp, io_dtype)
                T.copy(os_, o[t * M + by * block_M, bx * block_N])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: state[m, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Template 1b: Conv1x1 + BN + LIF (with decay)
# Pattern: Conv → BN → LIF(membrane, tau) → spike
# Difference from IF: h = decay * mem + recip_tau * bn (leaky integrate)
# Args: data, weight, membrane, bn_scale, bn_bias → output
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_lif(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, S=1, v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
):
    TB = T_steps * B
    OH = (H + S - 1) // S
    OW = (W + S - 1) // S
    M = B * OH * OW
    decay = 1.0 - recip_tau

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
            ds = T.alloc_shared((block_M, block_K), io_dtype)
            ws = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: mem[i, j] = state[m, f]
                else: mem[i, j] = T.float32(0)
            for t in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m = by * block_M + i
                        n = t * B + m // (OH * OW); hw = m % (OH * OW)
                        oh = hw // OW; ow = hw % OW
                        ds[i, j] = T.if_then_else(
                            (m < M) and (c < C_in),
                            data[n, oh * S, ow * S, c], io_dtype(0))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i; f = bx * block_N + j
                    if m < M and f < F:
                        bn = acc[i, j] * bn_scale[f] + bn_bias[f]
                        # LIF: leaky integrate (decay * mem + recip_tau * input)
                        h = T.float32(decay) * mem[i, j] + T.float32(recip_tau) * bn
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        os_[i, j] = T.cast(sp, io_dtype)
                T.copy(os_, o[t * M + by * block_M, bx * block_N])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: state[m, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Template 2: Conv1x1 + BN + Add + LIF
# Pattern: Conv → BN → Add(bn_out, residual) → LIF(membrane) → spike
# Args: data, weight, membrane, bn_scale, bn_bias, residual → output
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_add_lif(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    TB = T_steps * B
    OH = H; OW = W
    M = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), io_dtype),
        weight:   T.Tensor((C_in, F), io_dtype),
        state:    T.Tensor((M, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        residual: T.Tensor((TB, OH, OW, F), io_dtype),
        output:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        o = T.Tensor((TB * OH * OW, F), io_dtype, output.data)
        r = T.Tensor((TB * OH * OW, F), io_dtype, residual.data)
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            ds = T.alloc_shared((block_M, block_K), io_dtype)
            ws = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: mem[i, j] = state[m, f]
                else: mem[i, j] = T.float32(0)
            for t in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m = by * block_M + i
                        n = t * B + m // (OH * OW); hw = m % (OH * OW)
                        ds[i, j] = T.if_then_else(
                            (m < M) and (c < C_in),
                            data[n, hw // OW, hw % OW, c], io_dtype(0))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i; f = bx * block_N + j
                    if m < M and f < F:
                        bn = acc[i, j] * bn_scale[f] + bn_bias[f]
                        rv = T.cast(r[t * M + m, f], T.float32)
                        h = mem[i, j] + bn + rv
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        os_[i, j] = T.cast(sp, io_dtype)
                T.copy(os_, o[t * M + by * block_M, bx * block_N])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: state[m, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Template 3: Conv1x1 + BN + IF + Add
# Pattern: Conv → BN → IF(membrane) → spike; output = spike + residual
# Args: data, weight, membrane, bn_scale, bn_bias, residual → output
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_if_add(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    TB = T_steps * B
    OH = H; OW = W
    M = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), io_dtype),
        weight:   T.Tensor((C_in, F), io_dtype),
        state:    T.Tensor((M, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        residual: T.Tensor((TB, OH, OW, F), io_dtype),
        output:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        o = T.Tensor((TB * OH * OW, F), io_dtype, output.data)
        r = T.Tensor((TB * OH * OW, F), io_dtype, residual.data)
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            ds = T.alloc_shared((block_M, block_K), io_dtype)
            ws = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            mem = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: mem[i, j] = state[m, f]
                else: mem[i, j] = T.float32(0)
            for t in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m = by * block_M + i
                        n = t * B + m // (OH * OW); hw = m % (OH * OW)
                        ds[i, j] = T.if_then_else(
                            (m < M) and (c < C_in),
                            data[n, hw // OW, hw % OW, c], io_dtype(0))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i; f = bx * block_N + j
                    if m < M and f < F:
                        bn = acc[i, j] * bn_scale[f] + bn_bias[f]
                        h = mem[i, j] + bn
                        sp = T.if_then_else(h >= T.float32(v_threshold),
                                            T.float32(1), T.float32(0))
                        mem[i, j] = (T.float32(1) - sp) * h + sp * T.float32(v_reset)
                        rv = T.cast(r[t * M + m, f], T.float32)
                        os_[i, j] = T.cast(sp + rv, io_dtype)
                T.copy(os_, o[t * M + by * block_M, bx * block_N])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F: state[m, f] = mem[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Template 4: Conv1x1 + BN + IF + Add + LIF
# Pattern: Conv → BN → IF(mem1) → spike1 + residual → LIF(mem2) → spike2
# Args: data, weight, mem1, mem2, bn_scale, bn_bias, residual → output
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_if_add_lif(
    B, C_in, H, W, F, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    TB = T_steps * B
    OH = H; OW = W
    M = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), io_dtype),
        weight:   T.Tensor((C_in, F), io_dtype),
        state1:   T.Tensor((M, F), T.float32),
        state2:   T.Tensor((M, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        residual: T.Tensor((TB, OH, OW, F), io_dtype),
        output:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        o = T.Tensor((TB * OH * OW, F), io_dtype, output.data)
        r = T.Tensor((TB * OH * OW, F), io_dtype, residual.data)
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            ds = T.alloc_shared((block_M, block_K), io_dtype)
            ws = T.alloc_shared((block_K, block_N), io_dtype)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            m1 = T.alloc_fragment((block_M, block_N), T.float32)
            m2 = T.alloc_fragment((block_M, block_N), T.float32)
            os_ = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F:
                    m1[i, j] = state1[m, f]; m2[i, j] = state2[m, f]
                else:
                    m1[i, j] = T.float32(0); m2[i, j] = T.float32(0)
            for t in range(T_steps):
                T.clear(acc)
                for ki in T.Pipelined(T.ceildiv(C_in, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        c = ki * block_K + j; m = by * block_M + i
                        n = t * B + m // (OH * OW); hw = m % (OH * OW)
                        ds[i, j] = T.if_then_else(
                            (m < M) and (c < C_in),
                            data[n, hw // OW, hw % OW, c], io_dtype(0))
                    T.copy(weight[ki * block_K, bx * block_N], ws)
                    T.gemm(ds, ws, acc)
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i; f = bx * block_N + j
                    if m < M and f < F:
                        bn = acc[i, j] * bn_scale[f] + bn_bias[f]
                        h1 = m1[i, j] + bn
                        sp1 = T.if_then_else(h1 >= T.float32(v_threshold),
                                             T.float32(1), T.float32(0))
                        m1[i, j] = (T.float32(1) - sp1) * h1 + sp1 * T.float32(v_reset)
                        rv = T.cast(r[t * M + m, f], T.float32)
                        h2 = m2[i, j] + sp1 + rv
                        sp2 = T.if_then_else(h2 >= T.float32(v_threshold),
                                             T.float32(1), T.float32(0))
                        m2[i, j] = (T.float32(1) - sp2) * h2 + sp2 * T.float32(v_reset)
                        os_[i, j] = T.cast(sp2, io_dtype)
                T.copy(os_, o[t * M + by * block_M, bx * block_N])
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i; f = bx * block_N + j
                if m < M and f < F:
                    state1[m, f] = m1[i, j]; state2[m, f] = m2[i, j]
    return main


# ═══════════════════════════════════════════════════════════════════
# Template library: pattern → kernel function mapping
# ═══════════════════════════════════════════════════════════════════

KERNEL_TEMPLATES = {
    # Conv2d anchored (Conv1x1 GEMM)
    # IF (integrate-and-fire, no decay) vs LIF (leaky IF, with decay)
    'Conv2d+IF':          conv1x1_bn_if,
    'Conv2d+LIF':         conv1x1_bn_lif,       # LIF has decay factor
    'Conv2d+Add+LIF':     conv1x1_bn_add_lif,   # TODO: add LIF decay to Add+LIF
    'Conv2d+Add+IF':      conv1x1_bn_add_lif,
    'Conv2d+IF+Add':      conv1x1_bn_if_add,
    'Conv2d+LIF+Add':     conv1x1_bn_if_add,    # TODO: add LIF decay
    'Conv2d+IF+Add+LIF':  conv1x1_bn_if_add_lif,
    'Conv2d+IF+Add+IF':   conv1x1_bn_if_add_lif,
    'Conv2d+LIF+Add+LIF': conv1x1_bn_if_add_lif,
    # MatMul/Linear anchored
    'MatMul+LIF':          conv1x1_bn_lif,       # LIF with decay
    'MatMul+IF':           conv1x1_bn_if,
    'MatMul+Add+LIF':      conv1x1_bn_add_lif,
    'MatMul+Add+IF':       conv1x1_bn_add_lif,
    'MatMul+LIF+Add':      conv1x1_bn_if_add,
    'MatMul+Add+LIF+Add':  conv1x1_bn_if_add,
    'MatMul+IF+Add':       conv1x1_bn_if_add,
    'MatMul+IF+Add+LIF':   conv1x1_bn_if_add_lif,
    'MatMul+LIF+Add+LIF':  conv1x1_bn_if_add_lif,
    # Linear (same as MatMul)
    'Linear+LIF':          conv1x1_bn_lif,
    'Linear+Add+LIF':      conv1x1_bn_add_lif,
    'Linear+Add+LIF+Add':  conv1x1_bn_if_add,
}



# ═══════════════════════════════════════════════════════════════════
# Template 5: MaxPool + LIF (memory-bound, no GEMM anchor)
# Pattern: MaxPool(input, k=3, s=2) → LIF(membrane) → spikes
# Fuses pool + neuron into single kernel. Per-CTA T-loop.
# Args: input, membrane, dummy1, dummy2, bn_dummy → output
# Uses 5-arg interface for set_tilelang_5 compatibility.
# ═══════════════════════════════════════════════════════════════════

@tilelang.jit(out_idx=[-1])
def maxpool_lif(
    TB, C, H, W, pool_k, pool_s, pool_p, T_steps,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16, v_threshold=1.0, v_reset=0.0,
):
    OH = (H + 2 * pool_p - pool_k) // pool_s + 1
    OW = (W + 2 * pool_p - pool_k) // pool_s + 1
    B = TB // T_steps
    spatial = B * OH * OW
    M_total = TB * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), io_dtype),
        membrane: T.Tensor((spatial, C), T.float32),
        dummy1:   T.Tensor((C,), T.float32),
        dummy2:   T.Tensor((C,), T.float32),
        dummy3:   T.Tensor((C,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, C), io_dtype),
    ):
        out_flat = T.Tensor((M_total, C), io_dtype, spikes.data)

        with T.Kernel(
            T.ceildiv(C, block_N), T.ceildiv(spatial, block_M),
            threads=threads,
        ) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                s = by * block_M + i
                f = bx * block_N + j
                if s < spatial and f < C:
                    for t in range(T_steps):
                        n_idx = t * B + s // (OH * OW)
                        hw = s % (OH * OW)
                        oh = hw // OW
                        ow = hw % OW
                        # MaxPool: use output slot as scratch for max
                        out_flat[t * spatial + s, f] = io_dtype(-65504.0)
                        for kh in range(pool_k):
                            for kw in range(pool_k):
                                ih = oh * pool_s + kh - pool_p
                                iw = ow * pool_s + kw - pool_p
                                ib = (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)
                                cur = T.if_then_else(ib, data[n_idx, ih, iw, f], io_dtype(-65504.0))
                                prev = out_flat[t * spatial + s, f]
                                out_flat[t * spatial + s, f] = T.if_then_else(cur > prev, cur, prev)
                        # LIF
                        pooled = T.cast(out_flat[t * spatial + s, f], T.float32)
                        h = membrane[s, f] + pooled
                        spike = T.if_then_else(h >= T.float32(v_threshold), T.float32(1), T.float32(0))
                        membrane[s, f] = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                        out_flat[t * spatial + s, f] = T.cast(spike, io_dtype)
    return main


# ═══════════════════════════════════════════════════════════════════
# Update template library with new patterns
# ═══════════════════════════════════════════════════════════════════

# Conv+MaxPool: spatial dims change, can't fuse into GEMM epilogue.
# The slicer identifies these but they stay as Conv (GEMM) + MaxPool (native).
# Conv+MaxPool+LIF: Conv stays separate, MaxPool+LIF uses fused template.
# Conv+Add+GlobalAvgPool+LIF: rare tail pattern, stays decomposed.

KERNEL_TEMPLATES.update({
    # MaxPool+LIF standalone (memory-bound fusion, no GEMM anchor)
    'MaxPool+LIF':          maxpool_lif,
    # Conv+MaxPool patterns: Conv stays as GEMM, MaxPool absorbed partially
    # The slicer splits: Conv (compute anchor) | MaxPool+LIF (memory chain)
})
