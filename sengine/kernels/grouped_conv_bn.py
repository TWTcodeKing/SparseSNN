"""Grouped Conv2d + BN kernel for TileLang.

Handles Conv2d with groups > 1 and groups != C_in (i.e., NOT depthwise).
Used by SpikingResFormer GWFFN (groups = C_in // group_size).

For depthwise conv (groups == C_in), use dwconv_bn.py instead.
For standard conv (groups == 1), use conv2d_bn_if_t4.py instead.

Implementation: per-group im2col + GEMM using TileLang's T.gemm.
Each output tile computes one group's GEMM, iterating over groups
via the grid's third dimension.

Data layout: NHWC.  Accumulation: FP32.  Input/output: FP16.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def grouped_conv_bn_kernel(
    TB, C_in, H, W, C_out, K, S, D, P, groups,
    block_M, block_N, block_K, num_stages, threads,
):
    """Grouped Conv2d + BN.

    Args:
        TB: batch size (T*B or B for per-timestep)
        C_in: total input channels
        H, W: input spatial size
        C_out: total output channels
        K: kernel size (square)
        S: stride
        D: dilation
        P: padding
        groups: number of conv groups
        block_M/N/K: TileLang tile sizes
        num_stages: pipeline stages
        threads: threads per block
    """
    KH = KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = TB * OH * OW

    C_in_per_g = C_in // groups
    C_out_per_g = C_out // groups
    K_red = KH * KW * C_in_per_g  # reduction dim per group

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), T.float16),
        weight:   T.Tensor((KH, KW, C_in_per_g, C_out), T.float16),
        bn_scale: T.Tensor((C_out,), T.float32),
        bn_bias:  T.Tensor((C_out,), T.float32),
        output:   T.Tensor((TB, OH, OW, C_out), T.float16),
    ):
        output_flat = T.Tensor((M, C_out), T.float16, output.data)

        # Grid: (ceil(C_out_per_g / block_N), ceil(M / block_M), groups)
        with T.Kernel(
            T.ceildiv(C_out_per_g, block_N), T.ceildiv(M, block_M), groups,
            threads=threads,
        ) as (bx, by, g):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)

            # Weight for this group: slice [KH, KW, C_in_per_g, g*C_out_per_g : (g+1)*C_out_per_g]
            # Flatten to (K_red, C_out_per_g)
            # Global output channel offset
            co_offset = g * C_out_per_g
            ci_offset = g * C_in_per_g

            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                # im2col: read input for this group's channels
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (OH * OW)
                    hw_idx = m % (OH * OW)
                    oh_val = hw_idx // OW
                    ow_val = hw_idx % OW
                    kh_val = k // (KW * C_in_per_g)
                    kw_val = (k // C_in_per_g) % KW
                    cin_local = k % C_in_per_g
                    cin_global = ci_offset + cin_local
                    ah = oh_val * S + kh_val * D - P
                    aw = ow_val * S + kw_val * D - P
                    ib = ((ah >= 0) and (aw >= 0) and (ah < H)
                          and (aw < W) and (m < M) and (k < K_red))
                    data_shared[i, j] = T.if_then_else(
                        ib, data[n_idx, ah, aw, cin_global], T.float16(0))

                # Weight tile: (K_red, C_out_per_g) — need to index into
                # weight[kh, kw, cin_local, co_offset + bx*block_N + ...]
                # Flatten weight to (K_red, C_out) then slice columns
                for i, j in T.Parallel(block_K, block_N):
                    k = k_iter * block_K + i
                    co = bx * block_N + j
                    kh_val = k // (KW * C_in_per_g)
                    kw_val = (k // C_in_per_g) % KW
                    cin_local = k % C_in_per_g
                    co_global = co_offset + co
                    ib = (k < K_red) and (co < C_out_per_g)
                    weight_shared[i, j] = T.if_then_else(
                        ib, weight[kh_val, kw_val, cin_local, co_global],
                        T.float16(0))

                T.gemm(data_shared, weight_shared, acc)

            # BN epilogue — write to correct output channel offset
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                co = bx * block_N + j
                co_global = co_offset + co
                if m < M and co < C_out_per_g:
                    val = acc[i, j] * bn_scale[co_global] + bn_bias[co_global]
                    out_shared[i, j] = T.cast(val, T.float16)
            # Write to the correct output channel slice
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                co = bx * block_N + j
                co_global = co_offset + co
                if m < M and co < C_out_per_g:
                    output_flat[m, co_global] = out_shared[i, j]
    return main
