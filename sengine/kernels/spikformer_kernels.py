"""TileLang kernels for SpikFormer inference.

Five kernel types:
  1. Linear+BN+LIF (T-batched): Q/K/V projections, output proj, MLP fc1/fc2
  2. Linear+BN (no neuron): for ops where BN output feeds directly to matmul
  3. Matmul (pure GEMM): QK^T and attn@V attention matmuls, with optional scaling
  4. Matmul+LIF (fused): attn@V + LIF epilogue, per-timestep (T=1/launch)
  5. Standalone LIF neuron (memory-bound): decomposed path for BA-MTTS

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


# ─── Kernel 3b: Batched MatMul + Scale (for multi-head attention) ─

@tilelang.jit(out_idx=[-1])
def batched_matmul_kernel(
    batch, M, K, N,
    block_M, block_N, block_K, num_stages, threads,
    scale=1.0,
):
    """Batched GEMM: C[b] = (A[b] @ B[b]) * scale, for b in 0..batch-1.

    Used for multi-head attention Q@K^T and attn@V where batch = TB * num_heads.
    Each batch slice is an independent (M, K) @ (K, N) → (M, N) matmul.

    Tensors stored contiguously as (batch, M, K), (batch, K, N), (batch, M, N).
    """
    @T.prim_func
    def main(
        A: T.Tensor((batch * M, K), T.float16),
        B: T.Tensor((batch * K, N), T.float16),
        C: T.Tensor((batch * M, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            # Batch offset: each batch slice starts at b*M*K (A) and b*K*N (B)
            a_base = bz * M
            b_base = bz * K

            for k_iter in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages,
            ):
                T.copy(A[a_base + by * block_M, k_iter * block_K], A_shared)
                T.copy(B[b_base + k_iter * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)

            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            c_base = bz * M
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), T.float16)
            T.copy(out_shared, C[c_base + by * block_M, bx * block_N])
    return main


# ─── Kernel 3c: Batched MatMul with B transposed (for Q @ K^T) ─

@tilelang.jit(out_idx=[-1])
def batched_matmul_bt_kernel(
    batch, M, K, N,
    block_M, block_N, block_K, num_stages, threads,
    scale=1.0,
):
    """Batched GEMM with B transposed: C[b] = (A[b] @ B[b]^T) * scale.

    A: (batch*M, K) contiguous, B: (batch*N, K) contiguous (NOT transposed).
    C: (batch*M, N) contiguous.
    Computes: for each b, C[b](M,N) = A[b](M,K) @ B[b](N,K)^T * scale.

    Used for attention Q@K^T where Q=(batch,N_patches,hd), K=(batch,N_patches,hd).
    """
    @T.prim_func
    def main(
        A: T.Tensor((batch * M, K), T.float16),
        B: T.Tensor((batch * N, K), T.float16),
        C: T.Tensor((batch * M, N), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_N, block_K), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            a_base = bz * M
            b_base = bz * N

            for k_iter in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages,
            ):
                T.copy(A[a_base + by * block_M, k_iter * block_K], A_shared)
                T.copy(B[b_base + bx * block_N, k_iter * block_K], B_shared)
                # GEMM: A_shared(M,K) @ B_shared(N,K)^T = A_shared @ B_shared^T
                T.gemm(A_shared, B_shared, acc, transpose_B=True)

            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            c_base = bz * M
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), T.float16)
            T.copy(out_shared, C[c_base + by * block_M, bx * block_N])
    return main


# ─── Kernel 4: Fused MatMul + LIF (per-timestep, for attention) ─

@tilelang.jit(out_idx=[-1])
def matmul_lif_kernel(
    M, K, N,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
    T_steps=4, spatial=1,
):
    """Fused MatMul + LIF: spikes = fire((A @ B), membrane).

    Same GEMM body as matmul_kernel, with LIF epilogue from
    linear_bn_lif_t4_kernel.  Called per-timestep from the runtime
    (T_steps=1, M = B * heads * seq_len for one frame).

    M       = rows per timestep (B * heads * N_patches when T_steps=1).
    spatial = M (same since called per-timestep).
    state   = (spatial, N) FP32, shared across T invocations.
    """
    @T.prim_func
    def main(
        A:      T.Tensor((M, K), T.float16),
        B_mat:  T.Tensor((K, N), T.float16),
        state:  T.Tensor((spatial, N), T.float32),
        spikes: T.Tensor((M, N), T.float16),
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
                T.copy(B_mat[k_iter * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, acc)

            # LIF epilogue: integrate GEMM output into membrane, fire spikes
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                n = bx * block_N + j
                if m < M and n < N:
                    gemm_out = acc[i, j]
                    s_idx = m % spatial
                    one_sub = T.float32(1) - T.float32(recip_tau)
                    v = state[s_idx, n]
                    h = one_sub * v + T.float32(recip_tau) * gemm_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state[s_idx, n] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)
            T.copy(out_shared, spikes[by * block_M, bx * block_N])
    return main


# ─── Kernel 5: Standalone LIF neuron (memory-bound) ─────────────

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
