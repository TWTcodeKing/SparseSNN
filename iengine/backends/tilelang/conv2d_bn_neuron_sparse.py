"""2:4 Sparse Conv2d + BN + IF/LIF fused TileLang kernels.

Uses ``T.gemm_sp()`` for 2:4 structured sparse tensor core GEMM.

GEMM orientation
----------------
Standard Conv GEMM:  ``out[M,F] = im2col[M,K] @ weight[K,F]``
``T.gemm_sp`` requires the **first** operand to be sparse.  Since the
weight is sparse (from SBC pruning), we transpose the problem:

    ``out^T[F,M] = weight^T_sp[F, K//2] @ im2col^T[K, M]``

The im2col data is generated on-the-fly in shared memory with transposed
indexing (rows=K, cols=M), so no physical transpose is needed.  The
output accumulator is in (F, M) order; the epilogue writes it back as
(M, F) into the output tensor.

Weight preparation: use ``weight_compress.compress_conv_weight()`` to
convert SBC-pruned dense weights into TileLang's compressed format.

Calling convention (same as dense variant):

    spikes = kernel(data_nhwc, W_sparse, E_meta, state, bn_scale, bn_bias)
"""

import tilelang
import tilelang.language as T
from tilelang.layout import make_cutlass_metadata_layout
from tilelang.contrib import nvcc

_ARCH = nvcc.get_target_compute_version()
_ARCH_INFO = {"8.0": (16, "int16"), "8.9": (16, "int16"), "9.0": (8, "uint8")}


def _is_hopper() -> bool:
    import torch
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (9, 0)


_HOPPER = _is_hopper()


# ---------------------------------------------------------------------------
# 2:4 Sparse Conv2d + BN + IF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_if_sparse_kernel(
    N, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0,
    policy=T.GemmWarpPolicy.Square,
):
    """Fused sparse Conv2d + BN + IF neuron.

    Parameters
    ----------
    N, C_in, H, W, F, K, S, D, P : int
        Same as dense variant.
    block_M : int
        Tile along F (output channels) — the M dimension of the transposed GEMM.
    block_N : int
        Tile along spatial M = B*OH*OW — the N dimension of the transposed GEMM.
    block_K : int
        Tile along K_red = KH*KW*C_in (reduction).

    Notes
    -----
    Grid mapping is transposed vs the dense kernel:
    - bx indexes spatial tiles (ceildiv(M_spatial, block_N))
    - by indexes output-channel tiles (ceildiv(F, block_M))
    """
    KH = K
    KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M_spatial = N * OH * OW
    K_red = KH * KW * C_in

    e_factor, e_dtype = _ARCH_INFO[_ARCH]

    @T.prim_func
    def main(
        data:     T.Tensor((N, H, W, C_in), T.float16),
        W_sparse: T.Tensor((F, K_red // 2), T.float16),
        E_meta:   T.Tensor((F, K_red // e_factor), e_dtype),
        state:    T.Tensor((N, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((N, OH, OW, F), T.float16),
    ):
        # Transposed GEMM: out^T[F, M_spatial] = W^T_sp[F, K//2] @ im2col^T[K, M_spatial]
        # Grid: bx tiles M_spatial (spatial), by tiles F (channels)
        with T.Kernel(
            T.ceildiv(M_spatial, block_N),
            T.ceildiv(F, block_M),
            threads=threads,
        ) as (bx, by):
            # A = W_sparse tile: (block_M, block_K//2)
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            # B = im2col^T tile: (block_K, block_N)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)

            spikes_flat = T.Tensor((M_spatial, F), T.float16, spikes.data)
            state_flat  = T.Tensor((M_spatial, F), T.float32, state.data)

            T.clear(acc)
            T.annotate_layout({
                E_meta: make_cutlass_metadata_layout(
                    E_meta, mma_dtype=T.float16, block_k=block_K, arch=_ARCH,
                ),
                E_shared: make_cutlass_metadata_layout(
                    E_shared, mma_dtype=T.float16, block_k=block_K, arch=_ARCH,
                ),
            })

            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                # Load sparse weight tile
                T.copy(W_sparse[by * block_M, k_iter * block_K // 2], A_shared)
                T.copy(E_meta[by * block_M, k_iter * block_K // e_factor], E_shared)

                # im2col^T: rows=K_red, cols=M_spatial
                # Load tile (block_K, block_N) from im2col^T
                for i, j in T.Parallel(block_K, block_N):
                    k = k_iter * block_K + i     # reduction index
                    m = bx * block_N + j         # spatial index
                    access_h = (m % (OH * OW) // OW * S
                                + k // (KW * C_in) * D - P)
                    access_w = (m % OW * S
                                + k // C_in % KW * D - P)
                    in_bound = ((access_h >= 0) and (access_w >= 0)
                                and (access_h < H) and (access_w < W))
                    B_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[m // (OH * OW), access_h, access_w, k % C_in],
                        T.float16(0),
                    )

                T.gemm_sp(A_shared, E_shared, B_shared, acc, policy=policy)

            # ----- BN + IF neuron epilogue -----
            # acc[i,j] = out^T[by*block_M+i, bx*block_N+j]
            #          = out[spatial=bx*block_N+j, channel=by*block_M+i]
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                f = by * block_M + i       # output channel
                m = bx * block_N + j       # spatial position
                if f < F and m < M_spatial:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    v = state_flat[m, f]
                    h = v + bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[m, f] = v_new
                    # Write to (m, f) in row-major output — scattered write
                    spikes_flat[m, f] = T.cast(spike, T.float16)

    return main


# ---------------------------------------------------------------------------
# 2:4 Sparse Conv2d + BN + LIF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_lif_sparse_kernel(
    N, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
    policy=T.GemmWarpPolicy.Square,
):
    """Fused sparse Conv2d + BN + LIF neuron."""
    KH = K
    KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M_spatial = N * OH * OW
    K_red = KH * KW * C_in
    one_sub_recip = 1.0 - recip_tau

    e_factor, e_dtype = _ARCH_INFO[_ARCH]

    @T.prim_func
    def main(
        data:     T.Tensor((N, H, W, C_in), T.float16),
        W_sparse: T.Tensor((F, K_red // 2), T.float16),
        E_meta:   T.Tensor((F, K_red // e_factor), e_dtype),
        state:    T.Tensor((N, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((N, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(M_spatial, block_N),
            T.ceildiv(F, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), T.float16)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc      = T.alloc_fragment((block_M, block_N), T.float32)

            spikes_flat = T.Tensor((M_spatial, F), T.float16, spikes.data)
            state_flat  = T.Tensor((M_spatial, F), T.float32, state.data)

            T.clear(acc)
            T.annotate_layout({
                E_meta: make_cutlass_metadata_layout(
                    E_meta, mma_dtype=T.float16, block_k=block_K, arch=_ARCH,
                ),
                E_shared: make_cutlass_metadata_layout(
                    E_shared, mma_dtype=T.float16, block_k=block_K, arch=_ARCH,
                ),
            })

            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                T.copy(W_sparse[by * block_M, k_iter * block_K // 2], A_shared)
                T.copy(E_meta[by * block_M, k_iter * block_K // e_factor], E_shared)

                for i, j in T.Parallel(block_K, block_N):
                    k = k_iter * block_K + i
                    m = bx * block_N + j
                    access_h = (m % (OH * OW) // OW * S
                                + k // (KW * C_in) * D - P)
                    access_w = (m % OW * S
                                + k // C_in % KW * D - P)
                    in_bound = ((access_h >= 0) and (access_w >= 0)
                                and (access_h < H) and (access_w < W))
                    B_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[m // (OH * OW), access_h, access_w, k % C_in],
                        T.float16(0),
                    )

                T.gemm_sp(A_shared, E_shared, B_shared, acc, policy=policy)

            # ----- BN + LIF neuron epilogue -----
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                f = by * block_M + i
                m = bx * block_N + j
                if f < F and m < M_spatial:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    v = state_flat[m, f]
                    h = (T.float32(one_sub_recip) * v
                         + T.float32(recip_tau) * bn_out)
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[m, f] = v_new
                    spikes_flat[m, f] = T.cast(spike, T.float16)

    return main
