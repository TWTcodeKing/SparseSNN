"""Fused multi-head attention TileLang kernels.

Each kernel takes NHWC (or token-format) inputs directly and computes the
full attention core in a SINGLE kernel — no separate permute/transpose
kernel calls. The multi-head reshape and NCHW↔NHWC conversion are fused
into the GEMM prologue/epilogue via index math.

This eliminates ALL explicit layout_transpose_kernel calls from the
fused attention C++ dispatch.

NOTE: Attention K-reduction iterates over spatial (N) or head (hd) dims
within a packed (batch*spatial, channels) tensor. T.copy cannot be used for
these loads because partial tail tiles would read across batch boundaries.
T.Parallel with bounds checking is required. T.copy IS used for contiguous
loads (kv_in) and output writes where the sub-tile is fully within bounds.
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

    Per (tb, head): acc(hd, hd) = K^T(hd, N) @ V(N, hd).
    K, V read from NHWC with per-head channel stride (bounds-checked).
    Output: kv (TB*heads, hd, hd) contiguous.
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
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(N, block_K), num_stages=num_stages):
                # A = K^T: A[d, n] = K_nhwc[tb*N + n, head*hd + d]
                for i, j in T.Parallel(block_M, block_K):
                    d = by * block_M + i
                    n = k_iter * block_K + j
                    if d < hd and n < N:
                        A_shared[i, j] = K_nhwc[tb * N + n, head * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # B = V: B[n, d] = V_nhwc[tb*N + n, head*hd + d]
                for i, j in T.Parallel(block_K, block_N):
                    n = k_iter * block_K + i
                    d = bx * block_N + j
                    if n < N and d < hd:
                        B_shared[i, j] = V_nhwc[tb * N + n, head * hd + d]
                    else:
                        B_shared[i, j] = io_dtype(0)

                T.gemm(A_shared, B_shared, acc)

            # Output: contiguous (batch*hd, hd) → T.copy
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

    Q read from NHWC with per-head stride (bounds-checked).
    kv read from contiguous (T.copy).
    Output written to NHWC with per-head stride (bounds-checked).
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
                # A = Q: bounds-checked (spatial may exceed N at tile boundary)
                for i, j in T.Parallel(block_M, block_K):
                    n = by * block_M + i
                    d = k_iter * block_K + j
                    if n < N and d < hd:
                        A_shared[i, j] = Q_nhwc[tb * N + n, head * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # B = kv: contiguous → T.copy
                T.copy(kv_in[bz * hd + k_iter * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, acc)

            # Output: write to NHWC (per-head channels contiguous within row,
            # but spatial tiles may cross batch boundary → bounds-checked)
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n = by * block_M + i
                d = bx * block_N + j
                if n < N and d < hd:
                    out_shared[i, j] = T.cast(
                        acc[i, j] * T.float32(scale), io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                n = by * block_M + i
                d = bx * block_N + j
                if n < N and d < hd:
                    out_nhwc[tb * N + n, head * hd + d] = out_shared[i, j]

    return main


# ---------------------------------------------------------------------------
# SpikingResFormer DSSA: K@Q^T*scale1 → LIF → attn^T@V*scale2
# Two GEMM kernels that read NHWC y_kv and x_query directly.
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def dssa_kTq_kernel(
    TB, heads, hd, spatial_kv, spatial_q,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM1 for DSSA: attn[b,head] = K[b,head] @ Q[b,head]^T.

    K from y_kv NHWC — first C channels (per-head contiguous, bounds-checked).
    Q from x_query NHWC — per-head contiguous (bounds-checked).
    Output: attn (batch, spatial_kv, spatial_q) contiguous.
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
            A_shared = T.alloc_shared((block_M, block_K), io_dtype)
            B_shared = T.alloc_shared((block_N, block_K), io_dtype)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(hd, block_K), num_stages=num_stages):
                # A = K: K[n_kv, d] from y_kv (bounds-checked for spatial_kv)
                for i, j in T.Parallel(block_M, block_K):
                    n_kv = by * block_M + i
                    d = k_iter * block_K + j
                    if n_kv < spatial_kv and d < hd:
                        A_shared[i, j] = y_kv_nhwc[tb * spatial_kv + n_kv, head * 2 * hd + d]
                    else:
                        A_shared[i, j] = io_dtype(0)

                # B = Q: Q[n_q, d] from x_q (bounds-checked, loaded for transpose_B)
                for i, j in T.Parallel(block_N, block_K):
                    n_q = bx * block_N + i
                    d = k_iter * block_K + j
                    if n_q < spatial_q and d < hd:
                        B_shared[i, j] = x_q_nhwc[tb * spatial_q + n_q, head * hd + d]
                    else:
                        B_shared[i, j] = io_dtype(0)

                # acc(spatial_kv, spatial_q) += K(spatial_kv, hd) @ Q(spatial_q, hd)^T
                T.gemm(A_shared, B_shared, acc, transpose_B=True)

            # Output: bounds-checked write (spatial tiles may be partial)
            for i, j in T.Parallel(block_M, block_N):
                n_kv = by * block_M + i
                n_q = bx * block_N + j
                if n_kv < spatial_kv and n_q < spatial_q:
                    attn_out[bz * spatial_kv + n_kv, n_q] = T.cast(
                        acc[i, j], io_dtype)

    return main


@tilelang.jit(out_idx=[-1])
def dssa_v_attn_kernel(
    TB, heads, hd, spatial_kv, spatial_q, H_out, W_out,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """GEMM2 for DSSA: out[b,head] = attn[b,head]^T @ V[b,head].

    Restructured as acc(spatial_q, hd) = attn^T(spatial_q, spatial_kv) @ V(spatial_kv, hd)
    so output is (spatial_q, hd) — directly writable to NHWC per-head sub-tile.

    attn loaded with bounds check (K-reduction over spatial_kv has batch boundaries).
    V loaded with bounds check (same reason).
    Output written with bounds check (spatial_q tile may exceed spatial_q).
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
        # Grid: M=spatial_q, N=hd → output is (spatial_q, hd) for direct NHWC write
        with T.Kernel(
            T.ceildiv(hd, block_N), T.ceildiv(spatial_q, block_M), batch,
            threads=threads,
        ) as (bx, by, bz):
            # attn: (block_K, block_M) for transpose_A
            attn_shared = T.alloc_shared((block_K, block_M), io_dtype)
            V_shared    = T.alloc_shared((block_K, block_N), io_dtype)
            acc         = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            tb = bz // heads
            head = bz % heads

            for k_iter in T.Pipelined(T.ceildiv(spatial_kv, block_K), num_stages=num_stages):
                # attn tile: attn[n_kv, n_q] — bounds-checked
                for i, j in T.Parallel(block_K, block_M):
                    n_kv = k_iter * block_K + i
                    n_q = by * block_M + j
                    if n_kv < spatial_kv and n_q < spatial_q:
                        attn_shared[i, j] = attn_in[bz * spatial_kv + n_kv, n_q]
                    else:
                        attn_shared[i, j] = io_dtype(0)

                # V tile: V[n_kv, d] from y_kv second-half channels — bounds-checked
                for i, j in T.Parallel(block_K, block_N):
                    n_kv = k_iter * block_K + i
                    d = bx * block_N + j
                    if n_kv < spatial_kv and d < hd:
                        V_shared[i, j] = y_kv_nhwc[tb * spatial_kv + n_kv,
                                                     head * 2 * hd + hd + d]
                    else:
                        V_shared[i, j] = io_dtype(0)

                # acc(spatial_q, hd) += attn^T(spatial_q, spatial_kv) @ V(spatial_kv, hd)
                T.gemm(attn_shared, V_shared, acc, transpose_A=True)

            # Write to NHWC: out[tb*spatial_q + n_q, head*hd + d]
            out_shared = T.alloc_shared((block_M, block_N), io_dtype)
            for i, j in T.Parallel(block_M, block_N):
                n_q = by * block_M + i
                d = bx * block_N + j
                if n_q < spatial_q and d < hd:
                    out_shared[i, j] = T.cast(acc[i, j], io_dtype)

            for i, j in T.Parallel(block_M, block_N):
                n_q = by * block_M + i
                d = bx * block_N + j
                if n_q < spatial_q and d < hd:
                    out_nhwc[tb * spatial_q + n_q, head * hd + d] = out_shared[i, j]

    return main
