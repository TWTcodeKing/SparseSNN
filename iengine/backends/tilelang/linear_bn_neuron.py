"""Dense Linear + BN + IF/LIF fused TileLang kernels.

Standard GEMM (no im2col) with pipelined K-reduction, followed by a
register-resident BN scale+bias epilogue and IF or LIF spiking neuron
dynamics.  Used for fully-connected layers in transformer attention blocks.

Calling convention matches conv2d_bn_neuron: ``state`` is modified
in-place, ``spikes`` is allocated and returned.

    spikes = kernel(input_2d, weight, state, bn_scale, bn_bias)

Input layout: (M, K) where M = batch * seq_len, K = in_features.
Weight layout: (K, N) where N = out_features.
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# Dense Linear + BN + IF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def linear_bn_if_kernel(
    M, K, N_out,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0,
):
    """Fused Linear + BN scale/bias + IF neuron.

    Parameters
    ----------
    M : int
        Batch dimension (batch * seq_len for transformers).
    K : int
        Input features.
    N_out : int
        Output features.
    """

    @T.prim_func
    def main(
        inp:      T.Tensor((M, K), T.float16),
        weight:   T.Tensor((K, N_out), T.float16),
        state:    T.Tensor((M, N_out), T.float32),
        bn_scale: T.Tensor((N_out,), T.float32),
        bn_bias:  T.Tensor((N_out,), T.float32),
        spikes:   T.Tensor((M, N_out), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N_out, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(acc)

            for k_iter in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages,
            ):
                T.copy(inp[by * block_M, k_iter * block_K], A_shared)
                T.copy(weight[k_iter * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)

            # ----- BN + IF neuron epilogue -----
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N_out:
                    bn_out = acc[i, j] * bn_scale[n] + bn_bias[n]
                    v = state[m, n]
                    h = v + bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state[m, n] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Dense Linear + BN + LIF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def linear_bn_lif_kernel(
    M, K, N_out,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
):
    """Fused Linear + BN scale/bias + LIF neuron."""
    one_sub_recip = 1.0 - recip_tau

    @T.prim_func
    def main(
        inp:      T.Tensor((M, K), T.float16),
        weight:   T.Tensor((K, N_out), T.float16),
        state:    T.Tensor((M, N_out), T.float32),
        bn_scale: T.Tensor((N_out,), T.float32),
        bn_bias:  T.Tensor((N_out,), T.float32),
        spikes:   T.Tensor((M, N_out), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N_out, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(acc)

            for k_iter in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages,
            ):
                T.copy(inp[by * block_M, k_iter * block_K], A_shared)
                T.copy(weight[k_iter * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)

            # ----- BN + LIF neuron epilogue -----
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N_out:
                    bn_out = acc[i, j] * bn_scale[n] + bn_bias[n]
                    v = state[m, n]
                    h = (T.float32(one_sub_recip) * v
                         + T.float32(recip_tau) * bn_out)
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state[m, n] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes[by * block_M, bx * block_N])

    return main
