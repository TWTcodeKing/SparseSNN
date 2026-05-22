"""Fused Add+LIF kernel with per-CTA T-loop.

Fuses element-wise Add (residual connection) + LIF neuron into a single
kernel launch, eliminating the DRAM round-trip for the intermediate sum.

Pattern: Add(input_a, input_b) → LIF(sum, membrane) → spikes
Fused:   spikes = LIF(membrane + input_a + input_b)

Per-CTA T-loop ensures correct temporal membrane updates without cross-CTA races.
Uses 5-arg signature to match set_tilelang_5 C++ executor interface:
  (input_a, input_b, membrane, dummy_scale, output)
The dummy_scale arg is unused but needed for C++ pointer registration.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def add_lif_fused_kernel(
    TB, OH, OW, F,
    block_M=64, block_N=64, threads=128,
    io_dtype=T.float16,
    T_steps=4, v_threshold=1.0, v_reset=0.0, decay=1.0,
):
    """Fused Add+LIF with per-CTA T-loop.

    Args:
        TB: T * B (total batch with temporal)
        OH, OW: spatial dimensions
        F: feature channels
        T_steps: number of timesteps
        decay: LIF decay factor (1.0 = IF neuron, <1.0 = LIF)
    """
    B = TB // T_steps
    spatial = B * OH * OW
    M_total = TB * OH * OW

    @T.prim_func
    def main(
        input_a:  T.Tensor((TB, OH, OW, F), io_dtype),
        input_b:  T.Tensor((TB, OH, OW, F), io_dtype),
        membrane: T.Tensor((B * OH * OW, F), T.float32),
        dummy:    T.Tensor((F,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, F), io_dtype),
    ):
        a_flat = T.Tensor((M_total, F), io_dtype, input_a.data)
        b_flat = T.Tensor((M_total, F), io_dtype, input_b.data)
        out_flat = T.Tensor((M_total, F), io_dtype, spikes.data)

        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(spatial, block_M),
            threads=threads,
        ) as (bx, by):
            # Each CTA owns a tile of spatial positions, loops over T
            for i, j in T.Parallel(block_M, block_N):
                s = by * block_M + i
                f = bx * block_N + j
                if s < spatial and f < F:
                    for t in range(T_steps):
                        m = t * spatial + s
                        a_val = T.cast(a_flat[m, f], T.float32)
                        b_val = T.cast(b_flat[m, f], T.float32)
                        # LIF/IF integrate: membrane * decay + input_sum
                        h = membrane[s, f] * T.float32(decay) + a_val + b_val
                        # Fire
                        spike = T.if_then_else(
                            h >= T.float32(v_threshold),
                            T.float32(1), T.float32(0))
                        # Reset
                        membrane[s, f] = (T.float32(1) - spike) * h + \
                                          spike * T.float32(v_reset)
                        out_flat[m, f] = T.cast(spike, io_dtype)

    return main
