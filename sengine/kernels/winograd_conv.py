"""Winograd F(2×2, 3×3) Conv2d+BN TileLang kernels — implicit transform.

Single fused kernel: input transform computed on-the-fly in shared memory load
(like implicit GEMM for im2col). No intermediate buffer materialized.

Transform coefficients passed as small constant tensor parameters (16×9×3).
The 9 pixel reads per element are coalesced across the channel dimension.

For 3×3 stride=1 dilation=1 convolutions with C_in ≥ ~256 where K_red=9*C_in
makes im2col's K-loop deep. Winograd reduces K to C_in with 16 alpha passes.
"""

import torch
import numpy as np
import tilelang
import tilelang.language as T


_G = np.array([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]], dtype=np.float32)


def winograd_transform_weight(weight_nchw):
    """(C_out, C_in, 3, 3) → (16, C_in, C_out)"""
    G = torch.tensor(_G, device=weight_nchw.device)
    u = G @ weight_nchw.float() @ G.t()
    return u.reshape(u.shape[0], u.shape[1], 16).permute(2, 1, 0).contiguous().to(weight_nchw.dtype)


def _build_transform_tensors(device='cuda'):
    """Build constant tensors encoding sparse Winograd transform coefficients.

    Returns:
        in_terms: (16, 9, 3) int32 — (p, q, sign) per alpha, padded to 9
        in_counts: (16,) int32 — actual term count per alpha
        out_coeffs: (16, 4) float32 — AT[oi,r]*A[s,oj] for 4 output positions
    """
    BT = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
    AT = [[1, 1, 1, 0], [0, 1, -1, -1]]
    A_mat = [[1, 0], [1, 1], [1, -1], [0, -1]]

    in_terms = np.zeros((16, 9, 3), dtype=np.int32)
    in_counts = np.zeros(16, dtype=np.int32)
    out_coeffs = np.zeros((16, 4), dtype=np.float32)

    for r in range(4):
        for s in range(4):
            alpha = r * 4 + s
            idx = 0
            for p in range(4):
                for q in range(4):
                    c = BT[r][p] * BT[s][q]
                    if c != 0:
                        in_terms[alpha, idx] = [p, q, c]
                        idx += 1
            in_counts[alpha] = idx
            for oi in range(2):
                for oj in range(2):
                    out_coeffs[alpha, oi * 2 + oj] = AT[oi][r] * A_mat[s][oj]

    return (torch.tensor(in_terms, device=device),
            torch.tensor(in_counts, device=device),
            torch.tensor(out_coeffs, device=device, dtype=torch.float32))


@tilelang.jit(out_idx=[-1])
def winograd_conv2d_bn_kernel(
    TB, C_in, H, W, C_out, P,
    block_M, block_N, block_K, num_stages, threads,
    io_dtype=T.float16,
):
    """Implicit Winograd F(2×2,3×3) Conv2d+BN. Stride=1, dilation=1 only.

    Single fused kernel — input transform computed on-the-fly during shared
    memory load. No intermediate buffer. 16 alpha passes with 4 output accumulators.

    Grid: ceil(C_out/block_N) × ceil(M_tiles/block_M)
    """
    OH = H; OW = W
    tile_h = (H + 2 * P - 4) // 2 + 1
    tile_w = (W + 2 * P - 4) // 2 + 1
    M_tiles = TB * tile_h * tile_w

    @T.prim_func
    def main(
        data:       T.Tensor((TB, H, W, C_in), io_dtype),
        weight_U:   T.Tensor((16, C_in, C_out), io_dtype),
        bn_scale:   T.Tensor((C_out,), T.float32),
        bn_bias:    T.Tensor((C_out,), T.float32),
        in_terms:   T.Tensor((16, 9, 3), T.int32),
        in_counts:  T.Tensor((16,), T.int32),
        out_coeffs: T.Tensor((16, 4), T.float32),
        output:     T.Tensor((TB, OH, OW, C_out), io_dtype),
    ):
        with T.Kernel(
            T.ceildiv(C_out, block_N),
            T.ceildiv(M_tiles, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), io_dtype)
            weight_shared = T.alloc_shared((block_K, block_N), io_dtype)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)

            # 4 output accumulators for the 2×2 output tile positions
            out_00 = T.alloc_fragment((block_M, block_N), T.float32)
            out_01 = T.alloc_fragment((block_M, block_N), T.float32)
            out_10 = T.alloc_fragment((block_M, block_N), T.float32)
            out_11 = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(out_00); T.clear(out_01)
            T.clear(out_10); T.clear(out_11)

            # Fragment for implicit input transform accumulation
            v_frag = T.alloc_fragment((block_M, block_K), T.float32)

            for alpha in T.serial(16):
                T.clear(acc)
                n_terms = in_counts[alpha]

                for k_iter in T.Pipelined(
                    T.ceildiv(C_in, block_K), num_stages=num_stages,
                ):
                    # ── Implicit Winograd input transform ──
                    # V[alpha, m, c] = Σ sign * data[n, h+p, w+q, c]
                    # where (p, q, sign) from in_terms[alpha].
                    # The 9 reads are COALESCED across c (channel dim).
                    T.clear(v_frag)
                    for t_idx in T.serial(9):
                        for i, j in T.Parallel(block_M, block_K):
                            m = by * block_M + i
                            c = k_iter * block_K + j
                            if t_idx < n_terms:
                                p = in_terms[alpha, t_idx, 0]
                                q = in_terms[alpha, t_idx, 1]
                                sign = in_terms[alpha, t_idx, 2]
                                n_idx = m // (tile_h * tile_w)
                                th = (m % (tile_h * tile_w)) // tile_w
                                tw = m % tile_w
                                ih = th * 2 - P + p
                                iw = tw * 2 - P + q
                                valid = (m < M_tiles) & (c < C_in) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                                pixel = T.if_then_else(
                                    valid,
                                    T.cast(data[
                                        T.min(T.max(n_idx, 0), TB - 1),
                                        T.min(T.max(ih, 0), H - 1),
                                        T.min(T.max(iw, 0), W - 1),
                                        T.min(T.max(c, 0), C_in - 1)], T.float32),
                                    T.float32(0))
                                v_frag[i, j] = v_frag[i, j] + T.cast(sign, T.float32) * pixel
                    for i, j in T.Parallel(block_M, block_K):
                        data_shared[i, j] = T.cast(v_frag[i, j], io_dtype)

                    # Load pre-transformed weight tile
                    T.copy(weight_U[alpha, k_iter * block_K, bx * block_N],
                           weight_shared)
                    T.gemm(data_shared, weight_shared, acc)

                # ── Scatter-add into 4 output accumulators (output transform) ──
                c00 = out_coeffs[alpha, 0]
                c01 = out_coeffs[alpha, 1]
                c10 = out_coeffs[alpha, 2]
                c11 = out_coeffs[alpha, 3]
                for i, j in T.Parallel(block_M, block_N):
                    v = acc[i, j]
                    out_00[i, j] = out_00[i, j] + c00 * v
                    out_01[i, j] = out_01[i, j] + c01 * v
                    out_10[i, j] = out_10[i, j] + c10 * v
                    out_11[i, j] = out_11[i, j] + c11 * v

            # ── BN epilogue + write 2×2 output tiles ──
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M_tiles and f < C_out:
                    n  = m // (tile_h * tile_w)
                    th = (m % (tile_h * tile_w)) // tile_w
                    tw = m % tile_w
                    oh0 = th * 2; oh1 = th * 2 + 1
                    ow0 = tw * 2; ow1 = tw * 2 + 1
                    s = bn_scale[f]; b = bn_bias[f]
                    if oh0 < OH and ow0 < OW:
                        output[n, oh0, ow0, f] = T.cast(out_00[i,j] * s + b, io_dtype)
                    if oh0 < OH and ow1 < OW:
                        output[n, oh0, ow1, f] = T.cast(out_01[i,j] * s + b, io_dtype)
                    if oh1 < OH and ow0 < OW:
                        output[n, oh1, ow0, f] = T.cast(out_10[i,j] * s + b, io_dtype)
                    if oh1 < OH and ow1 < OW:
                        output[n, oh1, ow1, f] = T.cast(out_11[i,j] * s + b, io_dtype)

    return main
