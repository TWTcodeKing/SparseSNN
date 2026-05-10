"""Fused multi-head attention TileLang kernels.

Each kernel takes NHWC (or token-format) inputs directly and computes the
full attention core in a SINGLE kernel — no separate permute/transpose
kernel calls. The multi-head reshape and NCHW↔NHWC conversion are fused
into the GEMM prologue/epilogue via index math.

This eliminates ALL explicit layout_transpose_kernel calls from the
fused attention C++ dispatch.
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# MaxFormer Linear Attention: K^T@V → Q@kv*scale → merge → LIF
# Two GEMM kernels that read/write NHWC directly.
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def maxformer_kTv_kernel(
    TB, heads, hd, N, H, W,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM1 for MaxFormer: kv[b,head] = K[b,head]^T @ V[b,head].

    Reads K, V from NHWC layout (TB, H, W, C) where C = heads * hd, N = H * W.
    K stored as (TB, N, heads, hd) in memory, need K^T = (hd, N).
    V stored as (TB, N, heads, hd) in memory, need V = (N, hd).

    Output: kv (TB*heads, hd, hd) contiguous.

    GEMM per (tb, head): C(hd, hd) = A(hd, N) @ B(N, hd)
    where A = K^T, B = V. Both read from NHWC with stride.
    """
    C = heads * hd
    batch = TB * heads

    @T.prim_func
    def main(
        K_nhwc: T.Tensor((TB * N, C), io_dtype),   # (TB, H, W, C) flattened to (TB*N, C)
        V_nhwc: T.Tensor((TB * N, C), io_dtype),
        kv_out: T.Tensor((batch * hd, hd), io_dtype),  # (batch, hd, hd) contiguous
    ):
        with T.Kernel(
            T.ceildiv(hd, block_N), T.ceildiv(hd, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(N, block_K), num_stages=num_stages):
                # Load A = K^T block: A[d, n] = K_nhwc[tb*N + n, head*hd + d]
                # A_shared (block_M=hd_block, block_K=N_block)
                for i, j in T.Parallel(block_M, block_K):
                    d = by * block_M + i
                    n = k_iter * block_K + j
                    if d < hd and n < N:
                        A_shared[i, j] = K_nhwc[tb * N + n, head * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # Load B = V block: B[n, d] = V_nhwc[tb*N + n, head*hd + d]
                for i, j in T.Parallel(block_K, block_N):
                    n = k_iter * block_K + i
                    d = bx * block_N + j
                    if n < N and d < hd:
                        B_shared[i, j] = V_nhwc[tb * N + n, head * hd + d]
                    else:
                        B_shared[i, j] = io_dtype(0)

                T.gemm(A_shared, B_shared, acc)

            # Write output: kv_out[bz * hd + d1, d2]
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                d1 = by * block_M + i
                d2 = bx * block_N + j
                if d1 < hd and d2 < hd:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)
            T.copy(out_shared, kv_out[bz * hd + by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def maxformer_qkv_kernel(
    TB, heads, hd, N, H, W,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
    scale=1.0,
):
    """GEMM2 for MaxFormer: out[b,head] = (Q[b,head] @ kv[b,head]) * scale.

    Reads Q from NHWC layout (TB, H, W, C).
    Reads kv from contiguous (TB*heads, hd, hd) — output of GEMM1.
    Writes output to NHWC layout (TB, H, W, C).

    LIF neuron is applied separately (native CUDA kernel) because LIF
    is stateful across timesteps and requires sequential T processing.

    GEMM per (tb, head): C(N, hd) = A(N, hd) @ B(hd, hd) * scale
    where A = Q (read from NHWC), B = kv (contiguous).
    """
    C = heads * hd
    batch = TB * heads

    @T.prim_func
    def main(
        Q_nhwc:  T.Tensor((TB * N, C), io_dtype),
        kv_in:   T.Tensor((batch * hd, hd), io_dtype),
        out_nhwc: T.Tensor((TB * N, C), io_dtype),
    ):
        with T.Kernel(
            T.ceildiv(hd, block_N), T.ceildiv(N, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(hd, block_K), num_stages=num_stages):
                # Load A = Q block: A[n, d] = Q_nhwc[tb*N + n, head*hd + d]
                for i, j in T.Parallel(block_M, block_K):
                    n = by * block_M + i
                    d = k_iter * block_K + j
                    if n < N and d < hd:
                        A_shared[i, j] = Q_nhwc[tb * N + n, head * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # Load B = kv block: contiguous
                T.copy(kv_in[bz * hd + k_iter * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, acc)

            # Epilogue: scale + write NHWC (no LIF — done separately)
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n = by * block_M + i
                d = bx * block_N + j
                if n < N and d < hd:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), io_dtype)

            # Write to NHWC: out[tb*N + n, head*hd + d]
            for i, j in T.Parallel(block_M, block_N):
                n = by * block_M + i
                d = bx * block_N + j
                if n < N and d < hd:
                    out_nhwc[tb * N + n, head * hd + d] = out_shared[i, j]

    return main


# ---------------------------------------------------------------------------
# SpikingResFormer DSSA: K^T@Q*scale1 → LIF → V@attn*scale2
# Two GEMM kernels that read NHWC y_kv and x_query directly.
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def dssa_kTq_kernel(
    TB, heads, hd, spatial_kv, spatial_q,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM1 for DSSA: attn[b,head] = K[b,head]^T @ Q[b,head].

    Reads K from y_kv NHWC (TB, h'*w', 2*heads*hd) — first C channels.
    Reads Q from x_query NHWC (TB, H*W, heads*hd).
    Output: attn (TB*heads, spatial_kv, spatial_q) contiguous.

    GEMM per (tb, head): C(spatial_kv, spatial_q) = K^T(spatial_kv, hd) @ Q(hd, spatial_q)
    Scale1 applied separately (it's a per-head tensor, not a scalar).
    """
    C = heads * hd
    C2 = 2 * C  # y_kv has 2C channels
    batch = TB * heads

    @T.prim_func
    def main(
        y_kv_nhwc:  T.Tensor((TB * spatial_kv, C2), io_dtype),
        x_q_nhwc:   T.Tensor((TB * spatial_q, C), io_dtype),
        attn_out:   T.Tensor((batch * spatial_kv, spatial_q), io_dtype),
    ):
        with T.Kernel(
            T.ceildiv(spatial_q, block_N), T.ceildiv(spatial_kv, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(hd, block_K), num_stages=num_stages):
                # Load A = K^T: A[n_kv, d] = y_kv[tb*spatial_kv + n_kv, head*2*hd + d]
                # y_kv channels: [head0_K(hd) | head0_V(hd) | head1_K(hd) | head1_V(hd) | ...]
                for i, j in T.Parallel(block_M, block_K):
                    n_kv = by * block_M + i
                    d = k_iter * block_K + j
                    if n_kv < spatial_kv and d < hd:
                        A_shared[i, j] = y_kv_nhwc[tb * spatial_kv + n_kv, head * 2 * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # Load B = Q: B[d, n_q] = x_q[tb*spatial_q + n_q, head*hd + d]
                for i, j in T.Parallel(block_K, block_N):
                    d = k_iter * block_K + i
                    n_q = bx * block_N + j
                    if d < hd and n_q < spatial_q:
                        B_shared[i, j] = x_q_nhwc[tb * spatial_q + n_q, head * hd + d]
                    else:
                        B_shared[i, j] = io_dtype(0)

                T.gemm(A_shared, B_shared, acc)

            # Write attn_out contiguous: [bz * spatial_kv + n_kv, n_q]
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n_kv = by * block_M + i
                n_q = bx * block_N + j
                if n_kv < spatial_kv and n_q < spatial_q:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                n_kv = by * block_M + i
                n_q = bx * block_N + j
                if n_kv < spatial_kv and n_q < spatial_q:
                    attn_out[bz * spatial_kv + n_kv, n_q] = out_shared[i, j]

    return main


@tilelang.jit(out_idx=[-1])
def dssa_v_attn_kernel(
    TB, heads, hd, spatial_kv, spatial_q, H_out, W_out,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM2 for DSSA: out[b,head] = V[b,head] @ attn[b,head].

    Reads V from y_kv NHWC — second C channels (offset by C).
    Reads attn from contiguous (TB*heads, spatial_kv, spatial_q) — GEMM1 output.
    Writes output to NHWC (TB*spatial_q, C) — matches successor Conv/LIF layout.

    GEMM per (tb, head): C(hd, spatial_q) = V(hd, spatial_kv) @ attn(spatial_kv, spatial_q)
    Scale2 applied separately (per-head tensor).
    """
    C = heads * hd
    C2 = 2 * C
    batch = TB * heads

    @T.prim_func
    def main(
        y_kv_nhwc:  T.Tensor((TB * spatial_kv, C2), io_dtype),
        attn_in:    T.Tensor((batch * spatial_kv, spatial_q), io_dtype),
        out_nhwc:   T.Tensor((TB * spatial_q, C), io_dtype),
    ):
        with T.Kernel(
            T.ceildiv(spatial_q, block_N), T.ceildiv(hd, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(spatial_kv, block_K), num_stages=num_stages):
                # Load A = V: A[d, n_kv] = y_kv[tb*spatial_kv + n_kv, head*2*hd + hd + d]
                # y_kv channels: [head0_K(hd) | head0_V(hd) | head1_K(hd) | head1_V(hd) | ...]
                for i, j in T.Parallel(block_M, block_K):
                    d = by * block_M + i
                    n_kv = k_iter * block_K + j
                    if d < hd and n_kv < spatial_kv:
                        A_shared[i, j] = y_kv_nhwc[tb * spatial_kv + n_kv, head * 2 * hd + hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # Load B = attn: bounds-checked for non-aligned spatial dims
                for i, j in T.Parallel(block_K, block_N):
                    n_kv = k_iter * block_K + i
                    n_q = bx * block_N + j
                    if n_kv < spatial_kv and n_q < spatial_q:
                        B_shared[i, j] = attn_in[bz * spatial_kv + n_kv, n_q]
                    else:
                        B_shared[i, j] = io_dtype(0)

                T.gemm(A_shared, B_shared, acc)

            # Write to NHWC: out[tb*spatial_q + n_q, head*hd + d]
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                d = by * block_M + i
                n_q = bx * block_N + j
                if d < hd and n_q < spatial_q:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                d = by * block_M + i
                n_q = bx * block_N + j
                if d < hd and n_q < spatial_q:
                    out_nhwc[tb * spatial_q + n_q, head * hd + d] = out_shared[i, j]

    return main
