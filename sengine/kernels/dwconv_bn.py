"""Depthwise Conv2d + BN kernels for TileLang.

Used by MaxFormer (DWC3/DWC5/DWC7) and MS-QKFormer depthwise conv layers.

Unlike standard Conv (implicit GEMM), depthwise conv is per-channel:
  output[n, oh, ow, c] = sum_kh,kw(input[n, ih, iw, c] * weight[c, kh, kw])

Bandwidth-bound workload — optimization focuses on:
  1. Weight preload into shared memory (reused across all spatial positions)
  2. Input tile loaded into shared memory per kernel position (coalesced)
  3. Parameterized tile sizes for autotuning
  4. Fused BN epilogue (and optionally IF neuron) to avoid DRAM round-trip

Data layout: NHWC.  Accumulation: FP32.  Input/output: FP16.
"""

import tilelang
import tilelang.language as T


# ─── DW Conv + BN (no neuron) ───

@tilelang.jit(out_idx=[-1])
def dwconv_bn_kernel(TB, C, H, W, K, S, P,
                     block_C=64, block_HW=256, threads=256):
    """Fused Depthwise Conv2d + BN with shared memory weight preload.

    Args:
        TB: batch * T (temporal dim merged into batch)
        C: channels (= groups for depthwise)
        H, W: input spatial size
        K: kernel size (square)
        S: stride
        P: padding
        block_C: channel tile size
        block_HW: spatial tile size (output positions per block)
        threads: thread block size
    """
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1
    M = TB * OH * OW
    KK = K * K

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), T.float16),
        weight:   T.Tensor((C, K, K), T.float16),
        bn_scale: T.Tensor((C,), T.float32),
        bn_bias:  T.Tensor((C,), T.float32),
        output:   T.Tensor((TB, OH, OW, C), T.float16),
    ):
        output_flat = T.Tensor((M, C), T.float16, output.data)
        with T.Kernel(
            T.ceildiv(C, block_C), T.ceildiv(M, block_HW),
            threads=threads,
        ) as (bx, by):
            # Preload weight for this channel block into shared memory
            # Shape: [block_C, KK] — reused across all spatial positions
            weight_shared = T.alloc_shared((block_C, KK), T.float16)
            for j, k in T.Parallel(block_C, KK):
                c = bx * block_C + j
                if c < C:
                    kh = k // K
                    kw = k % K
                    weight_shared[j, k] = weight[c, kh, kw]
                else:
                    weight_shared[j, k] = T.float16(0)

            acc_frag = T.alloc_fragment((block_HW, block_C), T.float32)
            T.clear(acc_frag)

            # Accumulate over kernel positions with shared memory input tile
            for k_pos in range(KK):
                input_shared = T.alloc_shared((block_HW, block_C), T.float16)
                kh = k_pos // K
                kw = k_pos % K

                # Load input tile for this kernel position (coalesced along C)
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
                    input_shared[i, j] = T.if_then_else(
                        ib, data[n_idx, ih, iw, c], T.float16(0))

                # Multiply-accumulate: input * weight (both from shared memory)
                for i, j in T.Parallel(block_HW, block_C):
                    acc_frag[i, j] += T.cast(input_shared[i, j], T.float32) * \
                                      T.cast(weight_shared[j, k_pos], T.float32)

            # BN epilogue + write output
            out_frag = T.alloc_fragment((block_HW, block_C), T.float16)
            for i, j in T.Parallel(block_HW, block_C):
                c = bx * block_C + j
                c_safe = T.min(c, C - 1)
                out_frag[i, j] = T.cast(
                    acc_frag[i, j] * bn_scale[c_safe] + bn_bias[c_safe],
                    T.float16)
            T.copy(out_frag, output_flat[by * block_HW, bx * block_C])
    return main


# ─── DW Conv + BN + IF (fused with neuron) ───

@tilelang.jit(out_idx=[-1])
def dwconv_bn_if_kernel(TB, C, H, W, K, S, P,
                        T_steps=1, v_threshold=1.0, v_reset=0.0,
                        block_C=64, block_HW=256, threads=256):
    """Fused Depthwise Conv2d + BN + IF neuron with shared memory optimization.

    Processes T_steps=1 per launch (called T times from Python) when used in
    per-timestep mode, or T_steps>1 for T-batched mode.
    TB should be B (not T*B) when using per-timestep invocation.
    """
    OH = (H + 2*P - K) // S + 1
    OW = (W + 2*P - K) // S + 1
    M = TB * OH * OW
    KK = K * K
    B = TB // T_steps if T_steps > 1 else TB
    spatial = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), T.float16),
        weight:   T.Tensor((C, K, K), T.float16),
        membrane: T.Tensor((B, OH, OW, C), T.float32),
        bn_scale: T.Tensor((C,), T.float32),
        bn_bias:  T.Tensor((C,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, C), T.float16),
    ):
        output_flat = T.Tensor((M, C), T.float16, spikes.data)
        mem_flat = T.Tensor((B * OH * OW, C), T.float32, membrane.data)
        with T.Kernel(
            T.ceildiv(C, block_C), T.ceildiv(M, block_HW),
            threads=threads,
        ) as (bx, by):
            # Preload weight for this channel block into shared memory
            weight_shared = T.alloc_shared((block_C, KK), T.float16)
            for j, k in T.Parallel(block_C, KK):
                c = bx * block_C + j
                if c < C:
                    kh = k // K
                    kw = k % K
                    weight_shared[j, k] = weight[c, kh, kw]
                else:
                    weight_shared[j, k] = T.float16(0)

            acc_frag = T.alloc_fragment((block_HW, block_C), T.float32)
            T.clear(acc_frag)

            # Accumulate over kernel positions
            for k_pos in range(KK):
                input_shared = T.alloc_shared((block_HW, block_C), T.float16)
                kh = k_pos // K
                kw = k_pos % K

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
                    input_shared[i, j] = T.if_then_else(
                        ib, data[n_idx, ih, iw, c], T.float16(0))

                for i, j in T.Parallel(block_HW, block_C):
                    acc_frag[i, j] += T.cast(input_shared[i, j], T.float32) * \
                                      T.cast(weight_shared[j, k_pos], T.float32)

            # BN + IF neuron epilogue
            out_frag = T.alloc_fragment((block_HW, block_C), T.float16)
            for i, j in T.Parallel(block_HW, block_C):
                m = by * block_HW + i
                c = bx * block_C + j
                c_safe = T.min(c, C - 1)
                m_safe = T.min(m, M - 1)
                bn_out = acc_frag[i, j] * bn_scale[c_safe] + bn_bias[c_safe]

                # Membrane index: shared across T frames
                s_idx = m_safe % spatial
                v = mem_flat[s_idx, c_safe]
                h = v + bn_out
                spike = T.if_then_else(h >= T.float32(v_threshold),
                                       T.float32(1), T.float32(0))
                mem_flat[s_idx, c_safe] = (T.float32(1) - spike) * h + \
                                           spike * T.float32(v_reset)
                out_frag[i, j] = T.cast(spike, T.float16)
            T.copy(out_frag, output_flat[by * block_HW, bx * block_C])
    return main
