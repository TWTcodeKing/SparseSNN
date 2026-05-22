"""Depthwise Conv2d + BN kernels for TileLang.

Used by MaxFormer (DWC3/DWC5/DWC7) and MS-QKFormer depthwise conv layers.

Unlike standard Conv (implicit GEMM), depthwise conv is per-channel:
  output[n, oh, ow, c] = sum_kh,kw(input[n, ih, iw, c] * weight[c, kh, kw])

Memory-bound workload — the win is fusing Conv + BN (and optionally + IF)
to avoid the extra DRAM round-trip.

Data layout: NHWC.  Accumulation: FP32.  Input/output: FP16.
"""

import tilelang
import tilelang.language as T


# ─── DW Conv + BN (no neuron) ───

@tilelang.jit(out_idx=[-1])
def dwconv_bn_kernel(TB, C, H, W, K, S, P,
                     block_C=32, block_HW=128, threads=256,
                     io_dtype=T.float16):
    """Fused Depthwise Conv2d + BN.

    Args:
        TB: batch * T (temporal dim merged into batch)
        C: channels (= groups for depthwise)
        H, W: input spatial size
        K: kernel size (square)
        S: stride
        P: padding
    """
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1
    M = TB * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), io_dtype),
        weight:   T.Tensor((C, K, K), io_dtype),
        bn_scale: T.Tensor((C,), T.float32),
        bn_bias:  T.Tensor((C,), T.float32),
        output:   T.Tensor((TB, OH, OW, C), io_dtype),
    ):
        output_flat = T.Tensor((M, C), io_dtype, output.data)
        with T.Kernel(
            T.ceildiv(C, block_C), T.ceildiv(M, block_HW),
            threads=threads,
        ) as (bx, by):
            acc_frag = T.alloc_fragment((block_HW, block_C), T.float32)
            out_frag = T.alloc_fragment((block_HW, block_C), io_dtype)
            T.clear(acc_frag)

            for kh in range(K):
                for kw in range(K):
                    for i, j in T.Parallel(block_HW, block_C):
                        m = by * block_HW + i
                        c = bx * block_C + j
                        n_idx = m // (OH * OW)
                        hw = m % (OH * OW)
                        oh = hw // OW
                        ow = hw % OW
                        ih = oh * S + kh - P
                        iw = ow * S + kw - P
                        ib = (m < M) and (c < C) and \
                             (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)
                        acc_frag[i, j] += T.if_then_else(
                            ib,
                            T.cast(data[n_idx, ih, iw, c], T.float32) *
                            T.cast(weight[c, kh, kw], T.float32),
                            T.float32(0))

            for i, j in T.Parallel(block_HW, block_C):
                c = bx * block_C + j
                c_safe = T.min(c, C - 1)
                out_frag[i, j] = T.cast(
                    acc_frag[i, j] * bn_scale[c_safe] + bn_bias[c_safe],
                    io_dtype)
            T.copy(out_frag, output_flat[by * block_HW, bx * block_C])
    return main


# ─── DW Conv + BN + IF (fused with neuron) ───

@tilelang.jit(out_idx=[-1])
def dwconv_bn_if_kernel(TB, C, H, W, K, S, P,
                        block_C=32, block_HW=128, threads=256,
                        io_dtype=T.float16,
                        T_steps=4, v_threshold=1.0, v_reset=0.0):
    """Fused Depthwise Conv2d + BN + IF neuron.

    Processes T_steps=1 per launch (called T times from Python).
    TB should be B (not T*B) when using per-timestep invocation.
    """
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1
    M = TB * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), io_dtype),
        weight:   T.Tensor((C, K, K), io_dtype),
        membrane: T.Tensor((TB, OH, OW, C), T.float32),
        bn_scale: T.Tensor((C,), T.float32),
        bn_bias:  T.Tensor((C,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, C), io_dtype),
    ):
        output_flat = T.Tensor((M, C), io_dtype, spikes.data)
        mem_flat = T.Tensor((M, C), T.float32, membrane.data)
        with T.Kernel(
            T.ceildiv(C, block_C), T.ceildiv(M, block_HW),
            threads=threads,
        ) as (bx, by):
            acc_frag = T.alloc_fragment((block_HW, block_C), T.float32)
            out_frag = T.alloc_fragment((block_HW, block_C), io_dtype)
            T.clear(acc_frag)

            for kh in range(K):
                for kw in range(K):
                    for i, j in T.Parallel(block_HW, block_C):
                        m = by * block_HW + i
                        c = bx * block_C + j
                        n_idx = m // (OH * OW)
                        hw = m % (OH * OW)
                        oh = hw // OW
                        ow = hw % OW
                        ih = oh * S + kh - P
                        iw = ow * S + kw - P
                        ib = (m < M) and (c < C) and \
                             (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)
                        acc_frag[i, j] += T.if_then_else(
                            ib,
                            T.cast(data[n_idx, ih, iw, c], T.float32) *
                            T.cast(weight[c, kh, kw], T.float32),
                            T.float32(0))

            # BN + IF neuron epilogue
            for i, j in T.Parallel(block_HW, block_C):
                m = by * block_HW + i
                c = bx * block_C + j
                c_safe = T.min(c, C - 1)
                m_safe = T.min(m, M - 1)
                bn_out = acc_frag[i, j] * bn_scale[c_safe] + bn_bias[c_safe]
                v = mem_flat[m_safe, c_safe]
                h = v + bn_out
                spike = T.if_then_else(h >= T.float32(v_threshold),
                                       T.float32(1), T.float32(0))
                mem_flat[m_safe, c_safe] = (T.float32(1) - spike) * h + \
                                            spike * T.float32(v_reset)
                out_frag[i, j] = T.cast(spike, io_dtype)
            T.copy(out_frag, output_flat[by * block_HW, bx * block_C])
    return main
