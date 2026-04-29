"""TileLang kernels for SpikFormer inference.

Three kernel types:
  1. Linear+BN+LIF (T-batched): Q/K/V projections, output proj, MLP fc1/fc2
  2. Linear+BN (no neuron): for ops where BN output feeds directly to matmul
  3. Matmul (pure GEMM): QK^T and attn@V attention matmuls, with optional scaling

All kernels are T-batched (M = T*B*N_patches) and graph-capturable.
"""

import tilelang
import tilelang.language as T


# ─── Kernel 1: Linear + BN + LIF (T-batched, shared membrane) ───

@tilelang.jit(out_idx=[-1])
def linear_bn_lif_t4_kernel(
    M, K, N_out,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
    T_steps=4, spatial=1,
):
    """Fused Linear+BN+LIF with T-sequential membrane.

    M = T * B * seq_len (total rows across all timesteps).
    spatial = B * seq_len (rows per timestep).
    Membrane state is (spatial, N_out), shared across T.
    """
    @T.prim_func
    def main(
        inp:      T.Tensor((M, K), T.float16),
        weight:   T.Tensor((K, N_out), T.float16),
        state:    T.Tensor((spatial, N_out), T.float32),
        bn_scale: T.Tensor((N_out,), T.float32),
        bn_bias:  T.Tensor((N_out,), T.float32),
        spikes:   T.Tensor((M, N_out), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N_out, block_N), T.ceildiv(M, block_M),
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

            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N_out:
                    bn_out = acc[i, j] * bn_scale[n] + bn_bias[n]
                    s_idx = m % spatial
                    one_sub = T.float32(1) - T.float32(recip_tau)
                    v = state[s_idx, n]
                    h = one_sub * v + T.float32(recip_tau) * bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state[s_idx, n] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)
            T.copy(out_shared, spikes[by * block_M, bx * block_N])
    return main


# ─── Kernel 2: Linear + BN only (no neuron) ─────────────────────

@tilelang.jit(out_idx=[-1])
def linear_bn_kernel(
    M, K, N_out,
    block_M, block_N, block_K, num_stages, threads,
):
    """Linear + BN scale/bias. No neuron — writes BN output directly."""
    @T.prim_func
    def main(
        inp:      T.Tensor((M, K), T.float16),
        weight:   T.Tensor((K, N_out), T.float16),
        bn_scale: T.Tensor((N_out,), T.float32),
        bn_bias:  T.Tensor((N_out,), T.float32),
        output:   T.Tensor((M, N_out), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N_out, block_N), T.ceildiv(M, block_M),
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
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N_out:
                    val = acc[i, j] * bn_scale[n] + bn_bias[n]
                    out_shared[i, j] = T.cast(val, T.float16)
            T.copy(out_shared, output[by * block_M, bx * block_N])
    return main


# ─── Kernel 3: Pure matmul (for attention QK^T and attn@V) ──────

@tilelang.jit(out_idx=[-1])
def matmul_kernel(
    M, K, N,
    block_M, block_N, block_K, num_stages, threads,
    scale=1.0,
):
    """Pure GEMM: output = (A @ B) * scale.

    Used for:
      QK^T: A=(TB*H, N_patches, head_dim), B=(TB*H, head_dim, N_patches) → (TB*H, N, N)
      attn@V: A=(TB*H, N_patches, N_patches), B=(TB*H, N_patches, head_dim) → (TB*H, N, head_dim)

    Flattened to 2D: A=(M, K), B=(K, N), C=(M, N).
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)
            for k_iter in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages,
            ):
                T.copy(A[by * block_M, k_iter * block_K], A_shared)
                T.copy(B[k_iter * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), T.float16)
            T.copy(out_shared, C[by * block_M, bx * block_N])
    return main


# ─── Kernel 4: Standalone LIF neuron (memory-bound) ─────────────

@tilelang.jit(out_idx=[-1])
def lif_neuron_kernel(
    M, N,
    block_M, block_N, threads=128,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
    T_steps=4, spatial=1,
):
    """Standalone LIF neuron with T-sequential membrane."""
    @T.prim_func
    def main(
        inp:    T.Tensor((M, N), T.float16),
        state:  T.Tensor((spatial, N), T.float32),
        spikes: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    x_val = T.cast(inp[m, n], T.float32)
                    s_idx = m % spatial
                    one_sub = T.float32(1) - T.float32(recip_tau)
                    v = state[s_idx, n]
                    h = one_sub * v + T.float32(recip_tau) * x_val
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state[s_idx, n] = v_new
                    spikes[m, n] = T.cast(spike, T.float16)
    return main
