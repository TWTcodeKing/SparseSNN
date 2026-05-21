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
    Per (tb, head): acc(hd, hd) = K^T(hd, N) @ V(N, hd).

    K and V are (TB*N, C) row-major. Per-head channels [head*hd : (head+1)*hd]
    are contiguous within each row → load as (N_block, hd_block) sub-tile with
    T.copy, then use transpose_A=True for K^T.
    """
    C = heads * hd
    batch = TB * heads

    @T.prim_func
    def main(
        K_nhwc: T.Tensor((TB * N, C), io_dtype),
        V_nhwc: T.Tensor((TB * N, C), io_dtype),
        kv_out: T.Tensor((batch * hd, hd), io_dtype),
    ):
        with T.Kernel(
            T.ceildiv(hd, block_N), T.ceildiv(hd, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            # K_block (block_K, block_M) loaded as (N_block, hd_block) then transposed by gemm
            K_shared = T.alloc_shared((block_K, block_M), io_dtype)
            V_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(N, block_K), num_stages=num_stages):
                # K tile (N_block, hd_block): contiguous sub-rect of (TB*N, C)
                T.copy(K_nhwc[tb * N + k_iter * block_K, head * hd + by * block_M],
                       K_shared)
                # V tile (N_block, hd_block): contiguous sub-rect of (TB*N, C)
                T.copy(V_nhwc[tb * N + k_iter * block_K, head * hd + bx * block_N],
                       V_shared)
                # acc(hd, hd) += K^T(hd, N) @ V(N, hd)
                T.gemm(K_shared, V_shared, acc, transpose_A=True)

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

    Reads Q from NHWC (TB*N, C) — per-head channels contiguous → T.copy.
    Reads kv from contiguous (TB*heads, hd, hd) — T.copy.
    Writes output to NHWC (TB*N, C) — per-head channels contiguous → T.copy.

    GEMM per (tb, head): acc(N, hd) = Q(N, hd) @ kv(hd, hd) * scale.
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
            Q_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(hd, block_K), num_stages=num_stages):
                # Q tile (N_block, hd_block): contiguous sub-rect of (TB*N, C)
                T.copy(Q_nhwc[tb * N + by * block_M, head * hd + k_iter * block_K],
                       Q_shared)
                # kv tile: contiguous
                T.copy(kv_in[bz * hd + k_iter * block_K, bx * block_N], B_shared)
                T.gemm(Q_shared, B_shared, acc)

            # Epilogue: scale + write to NHWC via T.copy
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n = by * block_M + i
                d = bx * block_N + j
                if n < N and d < hd:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), io_dtype)

            # Write contiguous sub-tile to NHWC (per-head channels contiguous)
            T.copy(out_shared, out_nhwc[tb * N + by * block_M,
                                        head * hd + bx * block_N])

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
    """GEMM1 for DSSA: attn[b,head] = K[b,head] @ Q[b,head]^T.

    Reads K from y_kv NHWC — first C channels (per-head contiguous → T.copy).
    Reads Q from x_query NHWC — per-head contiguous → T.copy + transpose_B.
    Output: attn (TB*heads, spatial_kv, spatial_q) contiguous.

    GEMM per (tb, head): acc(spatial_kv, spatial_q) = K(spatial_kv, hd) @ Q(spatial_q, hd)^T
    """
    C = heads * hd
    C2 = 2 * C
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
            # K: (spatial_kv_block, hd_block), Q: (spatial_q_block, hd_block)
            K_shared = T.alloc_shared((block_M, block_K), io_dtype)
            Q_shared = T.alloc_shared((block_N, block_K), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(hd, block_K), num_stages=num_stages):
                # K tile: contiguous sub-rect of y_kv (per-head K channels)
                T.copy(y_kv_nhwc[tb * spatial_kv + by * block_M,
                                  head * 2 * hd + k_iter * block_K],
                       K_shared)
                # Q tile: contiguous sub-rect of x_q (per-head channels)
                T.copy(x_q_nhwc[tb * spatial_q + bx * block_N,
                                 head * hd + k_iter * block_K],
                       Q_shared)
                # acc(spatial_kv, spatial_q) += K(spatial_kv, hd) @ Q(spatial_q, hd)^T
                T.gemm(K_shared, Q_shared, acc, transpose_B=True)

            # Write attn_out contiguous
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n_kv = by * block_M + i
                n_q = bx * block_N + j
                if n_kv < spatial_kv and n_q < spatial_q:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)
            T.copy(out_shared, attn_out[bz * spatial_kv + by * block_M,
                                         bx * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def dssa_v_attn_kernel(
    TB, heads, hd, spatial_kv, spatial_q, H_out, W_out,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM2 for DSSA: out[b,head] = attn[b,head]^T @ V[b,head].

    Restructured as acc(spatial_q, hd) = attn^T(spatial_q, spatial_kv) @ V(spatial_kv, hd)
    so output is (spatial_q, hd) — directly writable to NHWC with T.copy.

    All loads use T.copy (contiguous sub-tiles). Scale2 applied separately.
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
        # Restructured grid: M=spatial_q, N=hd (output is (spatial_q, hd))
        with T.Kernel(
            T.ceildiv(hd, block_N), T.ceildiv(spatial_q, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            # attn: (spatial_kv_block, spatial_q_block) — for transpose_A
            attn_shared = T.alloc_shared((block_K, block_M), io_dtype)
            # V: (spatial_kv_block, hd_block) — contiguous from y_kv
            V_shared    = T.alloc_shared((block_K, block_N), io_dtype)
            acc         = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(spatial_kv, block_K), num_stages=num_stages):
                # attn tile (spatial_kv_block, spatial_q_block): contiguous
                T.copy(attn_in[bz * spatial_kv + k_iter * block_K,
                                by * block_M],
                       attn_shared)
                # V tile (spatial_kv_block, hd_block): contiguous sub-rect of y_kv
                T.copy(y_kv_nhwc[tb * spatial_kv + k_iter * block_K,
                                  head * 2 * hd + hd + bx * block_N],
                       V_shared)
                # acc(spatial_q, hd) += attn^T(spatial_q, spatial_kv) @ V(spatial_kv, hd)
                T.gemm(attn_shared, V_shared, acc, transpose_A=True)

            # Write to NHWC: contiguous sub-tile (spatial_q_block, hd_block)
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n_q = by * block_M + i
                d = bx * block_N + j
                if n_q < spatial_q and d < hd:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)
            T.copy(out_shared, out_nhwc[tb * spatial_q + by * block_M,
                                         head * hd + bx * block_N])

    return main
