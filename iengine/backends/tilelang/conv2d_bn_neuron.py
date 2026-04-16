"""Dense Conv2d + BN + IF/LIF fused TileLang kernels.

Each kernel performs a complete Conv2d (via im2col + pipelined GEMM on
tensor cores), followed by a register-resident BN scale+bias epilogue
and IF or LIF spiking neuron dynamics — all within a single GPU kernel
launch with zero intermediate DRAM round-trips.

Data layout: NHWC (matches tensor core preferences and CUTLASS convention).
Accumulation: FP32.  Input/output: FP16.  Membrane state: FP32.

Usage
-----
>>> kernel = conv2d_bn_if_kernel(
...     N=1, C_in=64, H=56, W=56, F=64, K=3, S=1, D=1, P=1,
...     block_M=64, block_N=128, block_K=32, num_stages=3, threads=256,
...     v_threshold=1.0, v_reset=0.0,
... )
>>> spikes, state_out = kernel(data_nhwc, weight_hwcf, spikes_buf,
...                            state_buf, bn_scale, bn_bias)
"""

import tilelang
import tilelang.language as T


def _is_hopper() -> bool:
    """Check if current GPU is Hopper (SM90) for TMA im2col."""
    import torch
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (9, 0)


_HOPPER = _is_hopper()


# ---------------------------------------------------------------------------
# Dense Conv2d + BN + IF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_if_kernel(
    N, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0,
):
    """Fused Conv2d + BN scale/bias + IF neuron.

    Call with concrete ints to get a compiled kernel, then invoke with
    tensors: ``spikes = kernel(data, weight, state, bn_scale, bn_bias)``.

    ``state`` is modified in-place (membrane potential persists across
    timesteps).  ``spikes`` is allocated and returned by TileLang.
    """
    KH = K
    KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = N * OH * OW
    K_red = KH * KW * C_in
    is_hopper = _HOPPER

    @T.prim_func
    def main(
        data:     T.Tensor((N, H, W, C_in), T.float16),
        weight:   T.Tensor((KH, KW, C_in, F), T.float16),
        state:    T.Tensor((N, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((N, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            # ----- allocations -----
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)

            # flat views for 2-D indexing into 4-D tensors
            weight_flat = T.Tensor((K_red, F), T.float16, weight.data)
            spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
            state_flat  = T.Tensor((M, F), T.float32, state.data)

            T.clear(acc)

            # ----- pipelined im2col GEMM -----
            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                # im2col: load receptive-field patch into shared memory
                if is_hopper:
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = (m % (OH * OW) // OW * S
                                    + k // (KW * C_in) * D - P)
                        access_w = (m % OW * S
                                    + k // C_in % KW * D - P)
                        in_bound = ((access_h >= 0) and (access_w >= 0)
                                    and (access_h < H) and (access_w < W))
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            data[m // (OH * OW), access_h, access_w, k % C_in],
                            T.float16(0),
                        )

                # weight tile copy (global -> shared)
                T.copy(weight_flat[k_iter * block_K, bx * block_N],
                       weight_shared)

                # tensor-core GEMM (shared -> fragment accumulator)
                T.gemm(data_shared, weight_shared, acc)

            # ----- BN + IF neuron epilogue (register-resident) -----
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    # BN: y = conv_out * scale + bias
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    # IF integrate: h = v + bn_out
                    v = state_flat[m, f]
                    h = v + bn_out
                    # Fire
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    # Reset
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[m, f] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Dense Conv2d + BN + LIF neuron
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_lif_kernel(
    N, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, recip_tau=0.5,
):
    """Fused Conv2d + BN scale/bias + LIF neuron.

    Same as :func:`conv2d_bn_if_kernel` but with leaky integration:
    ``h = (1 - 1/tau)*v + (1/tau)*bn_out``.

    ``state`` is modified in-place.  ``spikes`` is allocated and returned.
    """
    KH = K
    KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = N * OH * OW
    K_red = KH * KW * C_in
    is_hopper = _HOPPER
    one_sub_recip = 1.0 - recip_tau

    @T.prim_func
    def main(
        data:     T.Tensor((N, H, W, C_in), T.float16),
        weight:   T.Tensor((KH, KW, C_in, F), T.float16),
        state:    T.Tensor((N, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((N, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)

            weight_flat = T.Tensor((K_red, F), T.float16, weight.data)
            spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
            state_flat  = T.Tensor((M, F), T.float32, state.data)

            T.clear(acc)

            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                if is_hopper:
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    for i, j in T.Parallel(block_M, block_K):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = (m % (OH * OW) // OW * S
                                    + k // (KW * C_in) * D - P)
                        access_w = (m % OW * S
                                    + k // C_in % KW * D - P)
                        in_bound = ((access_h >= 0) and (access_w >= 0)
                                    and (access_h < H) and (access_w < W))
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            data[m // (OH * OW), access_h, access_w, k % C_in],
                            T.float16(0),
                        )

                T.copy(weight_flat[k_iter * block_K, bx * block_N],
                       weight_shared)
                T.gemm(data_shared, weight_shared, acc)

            # ----- BN + LIF neuron epilogue -----
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    # LIF integrate: h = (1 - 1/tau)*v + (1/tau)*bn_out
                    v = state_flat[m, f]
                    h = (T.float32(one_sub_recip) * v
                         + T.float32(recip_tau) * bn_out)
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0),
                    )
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[m, f] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Builder helper for the compilation pipeline
# ---------------------------------------------------------------------------

# Default tile configs for SM89 (RTX 4090), FP16
_DEFAULT_TILE_CONFIG = {
    "block_M": 64,
    "block_N": 128,
    "block_K": 32,
    "num_stages": 3,
    "threads": 256,
}


def build_conv_bn_neuron_kernel(
    fusion_group: dict,
    dag,
    *,
    sparse: bool = False,
    tile_config: dict | None = None,
):
    """Build a TileLang fused kernel from a fusion-group descriptor.

    Parameters
    ----------
    fusion_group : dict
        Fusion group from ``_build_fusion_groups()``.  Must have keys
        ``'conv'``, ``'bn'``, ``'neuron'`` pointing to node IDs.
    dag : OperatorDAG
        The operator DAG containing shape/param info per node.
    sparse : bool
        If True, build the 2:4 sparse variant (Phase 3).
    tile_config : dict, optional
        Override tile sizes / pipeline stages.

    Returns
    -------
    Compiled TileLang kernel.
    """
    cfg = {**_DEFAULT_TILE_CONFIG, **(tile_config or {})}

    conv_node = dag.nodes[fusion_group["conv"]]
    neuron_node = dag.nodes[fusion_group["neuron"]]

    cp = conv_node.params
    N, C_in = conv_node.input_shape[0], conv_node.input_shape[1]
    H, W = conv_node.input_shape[2], conv_node.input_shape[3]
    F = cp["out_channels"]
    K = cp["kernel_size"] if isinstance(cp["kernel_size"], int) else cp["kernel_size"][0]
    S = cp.get("stride", 1)
    if isinstance(S, (list, tuple)):
        S = S[0]
    D = cp.get("dilation", 1)
    if isinstance(D, (list, tuple)):
        D = D[0]
    P = cp.get("padding", 0)
    if isinstance(P, (list, tuple)):
        P = P[0]

    np_ = neuron_node.params
    v_threshold = np_.get("v_threshold", 1.0)
    v_reset = np_.get("v_reset", 0.0)
    neuron_type = neuron_node.op_type  # "if_neuron" or "lif_neuron"

    if neuron_type == "lif_neuron":
        recip_tau = np_.get("recip_tau", 0.5)
        return conv2d_bn_lif_kernel(
            N=N, C_in=C_in, H=H, W=W, F=F,
            K=K, S=S, D=D, P=P,
            v_threshold=v_threshold, v_reset=v_reset,
            recip_tau=recip_tau,
            **cfg,
        )
    else:
        return conv2d_bn_if_kernel(
            N=N, C_in=C_in, H=H, W=W, F=F,
            K=K, S=S, D=D, P=P,
            v_threshold=v_threshold, v_reset=v_reset,
            **cfg,
        )
