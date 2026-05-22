"""Fused MaxPool+LIF kernel with per-CTA T-loop.

Pattern: MaxPool2d(input) → LIF(pooled, membrane) → spikes
Fused: single kernel does pooling + neuron dynamics per spatial position.

Uses 5-arg signature for set_tilelang_5 compatibility:
  (input, membrane, pool_params_dummy, dummy2, output)
  Pool params (kernel, stride, pad) are compile-time constants.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def maxpool_lif_fused_kernel(
    TB, C, H, W, pool_k, pool_s, pool_p,
    block_M=64, block_N=64, threads=128,
    io_dtype=T.float16,
    T_steps=4, v_threshold=1.0, v_reset=0.0, decay=1.0,
):
    """Fused MaxPool2d + LIF with per-CTA T-loop."""
    OH = (H + 2 * pool_p - pool_k) // pool_s + 1
    OW = (W + 2 * pool_p - pool_k) // pool_s + 1
    B = TB // T_steps
    spatial = B * OH * OW
    M_total = TB * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C), io_dtype),
        membrane: T.Tensor((B * OH * OW, C), T.float32),
        dummy1:   T.Tensor((C,), T.float32),
        dummy2:   T.Tensor((C,), T.float32),
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

                        # MaxPool: use out_flat as scratch for max accumulation.
                        # Write -inf, then update with max of each pool position.
                        out_flat[t * spatial + s, f] = io_dtype(-65504.0)
                        for kh in range(pool_k):
                            for kw in range(pool_k):
                                ih = oh * pool_s + kh - pool_p
                                iw = ow * pool_s + kw - pool_p
                                ib = (ih >= 0) and (ih < H) and \
                                     (iw >= 0) and (iw < W)
                                cur = T.if_then_else(
                                    ib, data[n_idx, ih, iw, f],
                                    io_dtype(-65504.0))
                                prev = out_flat[t * spatial + s, f]
                                out_flat[t * spatial + s, f] = T.if_then_else(
                                    cur > prev, cur, prev)

                        # LIF: read pooled value from scratch, integrate, fire
                        pooled = T.cast(out_flat[t * spatial + s, f], T.float32)
                        h = membrane[s, f] * T.float32(decay) + pooled
                        spike = T.if_then_else(
                            h >= T.float32(v_threshold),
                            T.float32(1), T.float32(0))
                        membrane[s, f] = (T.float32(1) - spike) * h + \
                                          spike * T.float32(v_reset)
                        out_flat[t * spatial + s, f] = T.cast(spike, io_dtype)

    return main
