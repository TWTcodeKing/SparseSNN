"""Fused Conv2d+BN+IF kernel with T-batching and autotuning.

Key differences from conv2d_bn_neuron.py:
  1. N=T*B — temporal dim merged into batch (matches TRT's approach)
  2. IF epilogue processes T frames sequentially over shared membrane
  3. @tilelang.autotune explores a large config space automatically

The Conv GEMM processes all T frames in a single launch (M = T*B*OH*OW),
then the epilogue iterates T times over the accumulator to update the
shared membrane state. This gives 4x more tiles for SM occupancy while
keeping the neuron dynamics correct.

Data layout: NHWC.  Accumulation: FP32.  Input/output: FP16.  Membrane: FP32.
"""

import tilelang
import tilelang.language as T


def _is_hopper() -> bool:
    import torch
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (9, 0)


_HOPPER = _is_hopper()


# ---------------------------------------------------------------------------
# Config space for autotuning (SM89 / RTX 4090)
# ---------------------------------------------------------------------------

def _make_configs_conv(C_in, F, K, OH, OW, TB):
    """Generate tile config space based on problem dimensions."""
    K_red = K * K * C_in
    M = TB * OH * OW
    configs = []
    for block_M in [32, 64, 128, 256]:
        for block_N in [32, 64, 128, 256]:
            for block_K in [32, 64]:
                for num_stages in [2, 3, 4]:
                    for threads in [128, 256]:
                        # Validity constraints
                        if block_K > K_red:
                            continue
                        if block_N > F * 2:
                            continue
                        if block_M > M:
                            continue
                        # Shared memory estimate (rough): must fit in 100KB
                        smem = (block_M * block_K + block_K * block_N) * 2 * num_stages
                        if smem > 100 * 1024:
                            continue
                        # Thread count must be enough for the tile
                        min_threads = max(block_M, block_N) // 2
                        if threads < min_threads:
                            continue
                        configs.append({
                            "block_M": block_M,
                            "block_N": block_N,
                            "block_K": block_K,
                            "num_stages": num_stages,
                            "threads": threads,
                        })
    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Fused Conv2d + BN + IF with T-batching (autotuned)
# ---------------------------------------------------------------------------

def make_conv2d_bn_if_t4(
    TB, C_in, H, W, F, K, S, D, P,
    v_threshold=1.0, v_reset=0.0, T_steps=4,
):
    """Build an autotuned fused Conv2d+BN+IF kernel with T-batching.

    Parameters
    ----------
    TB : int
        Batch size = T * B (e.g., 4 for T=4, B=1).
    T_steps : int
        Number of temporal steps (for epilogue loop).

    The Conv GEMM runs on the full (TB, H, W, C_in) input.
    The epilogue processes T_steps frames sequentially, each of size
    (B, OH, OW, F), updating the shared membrane state.
    """
    KH = KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = TB * OH * OW           # total spatial positions across all T frames
    B = TB // T_steps          # actual batch size (1 for our use case)
    spatial = B * OH * OW      # spatial positions per timestep
    K_red = KH * KW * C_in
    is_hopper = _HOPPER

    configs = _make_configs_conv(C_in, F, K, OH, OW, TB)

    @tilelang.autotune(
        configs=configs,
        warmup=10,
        rep=50,
        skip_check=True,
    )
    @tilelang.jit(out_idx=[-1])
    def kernel(
        TB=TB, C_in=C_in, H=H, W=W, F=F,
        KH=KH, KW=KW, S=S, D=D, P=P,
        OH=OH, OW=OW, M=M, K_red=K_red,
        T_steps=T_steps, B=B, spatial=spatial,
        v_threshold=v_threshold, v_reset=v_reset,
        block_M=64, block_N=64, block_K=32,
        num_stages=3, threads=128,
    ):
        @T.prim_func
        def main(
            data:     T.Tensor((TB, H, W, C_in), T.float16),
            weight:   T.Tensor((KH, KW, C_in, F), T.float16),
            state:    T.Tensor((B, OH, OW, F), T.float32),
            bn_scale: T.Tensor((F,), T.float32),
            bn_bias:  T.Tensor((F,), T.float32),
            spikes:   T.Tensor((TB, OH, OW, F), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(F, block_N),
                T.ceildiv(M, block_M),
                threads=threads,
            ) as (bx, by):
                # ── allocations ──
                data_shared   = T.alloc_shared((block_M, block_K), T.float16)
                weight_shared = T.alloc_shared((block_K, block_N), T.float16)
                acc           = T.alloc_fragment((block_M, block_N), T.float32)

                weight_flat = T.Tensor((K_red, F), T.float16, weight.data)
                spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
                state_flat  = T.Tensor((B * OH * OW, F), T.float32, state.data)

                T.clear(acc)

                # ── pipelined im2col GEMM (all T frames at once) ──
                for k_iter in T.Pipelined(
                    T.ceildiv(K_red, block_K), num_stages=num_stages,
                ):
                    if is_hopper:
                        T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                    else:
                        for i, j in T.Parallel(block_M, block_K):
                            k = k_iter * block_K + j
                            m = by * block_M + i
                            # im2col index decomposition
                            n_idx = m // (OH * OW)
                            hw_idx = m % (OH * OW)
                            oh = hw_idx // OW
                            ow = hw_idx % OW
                            kh = k // (KW * C_in)
                            kw = (k // C_in) % KW
                            cin = k % C_in
                            access_h = oh * S + kh * D - P
                            access_w = ow * S + kw * D - P
                            in_bound = (
                                (access_h >= 0) and (access_w >= 0)
                                and (access_h < H) and (access_w < W)
                                and (m < M) and (k < K_red)
                            )
                            data_shared[i, j] = T.if_then_else(
                                in_bound,
                                data[n_idx, access_h, access_w, cin],
                                T.float16(0),
                            )

                    T.copy(weight_flat[k_iter * block_K, bx * block_N],
                           weight_shared)
                    T.gemm(data_shared, weight_shared, acc)

                # ── BN + IF neuron epilogue with T-sequential membrane ──
                out_shared = T.alloc_shared((block_M, block_N), T.float16)

                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    f = bx * block_N + j
                    if m < M and f < F:
                        # BN: y = conv_out * scale + bias
                        bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]

                        # Which timestep and spatial position is this?
                        t_idx = m // spatial    # 0..T-1
                        s_idx = m % spatial     # position within one frame

                        # Load membrane (shared across T)
                        v = state_flat[s_idx, f]

                        # IF integrate
                        h = v + bn_out

                        # Fire
                        spike = T.if_then_else(
                            h >= v_threshold, T.float32(1), T.float32(0))

                        # Reset
                        v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)

                        # Write back membrane (will be read by next T frame
                        # in the same tile — tiles are ordered M-major so
                        # t=0 positions come before t=1 positions)
                        state_flat[s_idx, f] = v_new

                        out_shared[i, j] = T.cast(spike, T.float16)

                T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Non-autotuned variant (for quick testing / CUDA Graph capture)
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_if_t4_kernel(
    TB, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, T_steps=4,
):
    """Non-autotuned variant — call with explicit tile config."""
    KH = KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = TB * OH * OW
    B = TB // T_steps
    spatial = B * OH * OW
    K_red = KH * KW * C_in
    is_hopper = _HOPPER

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), T.float16),
        weight:   T.Tensor((KH, KW, C_in, F), T.float16),
        state:    T.Tensor((B, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, F), T.float16),
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
            state_flat  = T.Tensor((B * OH * OW, F), T.float32, state.data)

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
                        n_idx = m // (OH * OW)
                        hw_idx = m % (OH * OW)
                        oh = hw_idx // OW
                        ow = hw_idx % OW
                        kh = k // (KW * C_in)
                        kw = (k // C_in) % KW
                        cin = k % C_in
                        access_h = oh * S + kh * D - P
                        access_w = ow * S + kw * D - P
                        in_bound = (
                            (access_h >= 0) and (access_w >= 0)
                            and (access_h < H) and (access_w < W)
                            and (m < M) and (k < K_red)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound,
                            data[n_idx, access_h, access_w, cin],
                            T.float16(0),
                        )

                T.copy(weight_flat[k_iter * block_K, bx * block_N],
                       weight_shared)
                T.gemm(data_shared, weight_shared, acc)

            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    t_idx = m // spatial
                    s_idx = m % spatial
                    v = state_flat[s_idx, f]
                    h = v + bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[s_idx, f] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# SEW-ResNet18 layer specs (ImageNet, T=4, B=1)
# ---------------------------------------------------------------------------

SEW_RESNET18_LAYERS = [
    # (name, C_in, C_out, H, W, K, S, P)
    # stem is skipped (C_in=3, 7x7 — use cuDNN)
    ("layer1.0.conv1", 64, 64, 56, 56, 3, 1, 1),
    ("layer1.0.conv2", 64, 64, 56, 56, 3, 1, 1),
    ("layer1.1.conv1", 64, 64, 56, 56, 3, 1, 1),
    ("layer1.1.conv2", 64, 64, 56, 56, 3, 1, 1),
    ("layer2.0.conv1", 64, 128, 56, 56, 3, 2, 1),
    ("layer2.0.ds",    64, 128, 56, 56, 1, 2, 0),
    ("layer2.0.conv2", 128, 128, 28, 28, 3, 1, 1),
    ("layer2.1.conv1", 128, 128, 28, 28, 3, 1, 1),
    ("layer2.1.conv2", 128, 128, 28, 28, 3, 1, 1),
    ("layer3.0.conv1", 128, 256, 28, 28, 3, 2, 1),
    ("layer3.0.ds",    128, 256, 28, 28, 1, 2, 0),
    ("layer3.0.conv2", 256, 256, 14, 14, 3, 1, 1),
    ("layer3.1.conv1", 256, 256, 14, 14, 3, 1, 1),
    ("layer3.1.conv2", 256, 256, 14, 14, 3, 1, 1),
    ("layer4.0.conv1", 256, 512, 14, 14, 3, 2, 1),
    ("layer4.0.ds",    256, 512, 14, 14, 1, 2, 0),
    ("layer4.0.conv2", 512, 512, 7, 7, 3, 1, 1),
    ("layer4.1.conv1", 512, 512, 7, 7, 3, 1, 1),
    ("layer4.1.conv2", 512, 512, 7, 7, 3, 1, 1),
]


# ---------------------------------------------------------------------------
# Kernel 2: 1×1 Conv+BN+IF — pure GEMM, no im2col
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_if_t4_kernel(
    TB, C_in, H, W, F, S,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, T_steps=4,
):
    """Fused 1×1 Conv+BN+IF with stride support. Pure GEMM — no im2col.

    For stride=2: input is subsampled by stride before GEMM.
    """
    OH = (H + S - 1) // S  # ceil(H/S) for 1x1 with stride
    OW = (W + S - 1) // S
    M = TB * OH * OW
    B = TB // T_steps
    spatial = B * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), T.float16),
        weight:   T.Tensor((C_in, F), T.float16),
        state:    T.Tensor((B, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)

            spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
            state_flat  = T.Tensor((B * OH * OW, F), T.float32, state.data)

            T.clear(acc)

            for k_iter in T.Pipelined(
                T.ceildiv(C_in, block_K), num_stages=num_stages,
            ):
                # 1×1 gather: data[n, oh*S, ow*S, cin]
                for i, j in T.Parallel(block_M, block_K):
                    cin = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (OH * OW)
                    hw = m % (OH * OW)
                    oh = hw // OW
                    ow = hw % OW
                    in_bound = (m < M) and (cin < C_in)
                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[n_idx, oh * S, ow * S, cin],
                        T.float16(0),
                    )

                T.copy(weight[k_iter * block_K, bx * block_N],
                       weight_shared)
                T.gemm(data_shared, weight_shared, acc)

            # BN + IF epilogue
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    s_idx = m % spatial
                    v = state_flat[s_idx, f]
                    h = v + bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[s_idx, f] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# Kernel 3: Stem Conv(3→64, 7×7, s=2) + BN + IF — padded C_in
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def stem_conv_bn_if_t4_kernel(
    TB, H, W,
    block_M, block_N, block_K, num_stages, threads,
    v_threshold=1.0, v_reset=0.0, T_steps=4,
    C_in_padded=16, C_in_real=3, F=64, KH=7, KW=7, S=2, P=3,
):
    """Fused stem Conv(3→64, 7×7, s=2, p=3) + BN + IF.

    C_in is padded from 3 to 16 for tensor core alignment.
    The kernel reads only the first 3 channels and treats the rest as zero.
    """
    OH = (H + 2 * P - KH) // S + 1  # 112
    OW = (W + 2 * P - KW) // S + 1  # 112
    M = TB * OH * OW
    B = TB // T_steps
    spatial = B * OH * OW
    K_red = KH * KW * C_in_padded  # 7*7*16 = 784

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in_padded), T.float16),
        weight:   T.Tensor((KH, KW, C_in_padded, F), T.float16),
        state:    T.Tensor((B, OH, OW, F), T.float32),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        spikes:   T.Tensor((TB, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)

            weight_flat = T.Tensor((K_red, F), T.float16, weight.data)
            spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
            state_flat  = T.Tensor((B * OH * OW, F), T.float32, state.data)

            T.clear(acc)

            for k_iter in T.Pipelined(
                T.ceildiv(K_red, block_K), num_stages=num_stages,
            ):
                # im2col with padded C_in
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (OH * OW)
                    hw = m % (OH * OW)
                    oh = hw // OW
                    ow = hw % OW
                    kh = k // (KW * C_in_padded)
                    kw = (k // C_in_padded) % KW
                    cin = k % C_in_padded
                    access_h = oh * S + kh - P
                    access_w = ow * S + kw - P
                    # Only read real channels (0..2), zero-pad 3..15
                    in_bound = (
                        (access_h >= 0) and (access_w >= 0)
                        and (access_h < H) and (access_w < W)
                        and (cin < C_in_real) and (m < M) and (k < K_red)
                    )
                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[n_idx, access_h, access_w, cin],
                        T.float16(0),
                    )

                T.copy(weight_flat[k_iter * block_K, bx * block_N],
                       weight_shared)
                T.gemm(data_shared, weight_shared, acc)

            # BN + IF epilogue
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    bn_out = acc[i, j] * bn_scale[f] + bn_bias[f]
                    s_idx = m % spatial
                    v = state_flat[s_idx, f]
                    h = v + bn_out
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[s_idx, f] = v_new
                    out_shared[i, j] = T.cast(spike, T.float16)

            T.copy(out_shared, spikes_flat[by * block_M, bx * block_N])

    return main


# =========================================================================
# DECOMPOSED KERNELS — for two-stream compute/memory scheduling
# =========================================================================

# ---------------------------------------------------------------------------
# Kernel D1: Conv+BN ONLY (no IF) — compute-bound, 3×3
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv2d_bn_t4_kernel(
    TB, C_in, H, W, F, K, S, D, P,
    block_M, block_N, block_K, num_stages, threads,
):
    """Conv2d+BN without neuron. Writes BN output (FP16) to DRAM."""
    KH = KW = K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    M = TB * OH * OW
    K_red = KH * KW * C_in
    is_hopper = _HOPPER

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), T.float16),
        weight:   T.Tensor((KH, KW, C_in, F), T.float16),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        output:   T.Tensor((TB, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)
            weight_flat = T.Tensor((K_red, F), T.float16, weight.data)
            output_flat = T.Tensor((M, F), T.float16, output.data)
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
                        n_idx = m // (OH * OW)
                        hw_idx = m % (OH * OW)
                        oh_val = hw_idx // OW
                        ow_val = hw_idx % OW
                        kh_val = k // (KW * C_in)
                        kw_val = (k // C_in) % KW
                        cin_val = k % C_in
                        ah = oh_val * S + kh_val * D - P
                        aw = ow_val * S + kw_val * D - P
                        ib = ((ah >= 0) and (aw >= 0) and (ah < H)
                              and (aw < W) and (m < M) and (k < K_red))
                        data_shared[i, j] = T.if_then_else(
                            ib, data[n_idx, ah, aw, cin_val], T.float16(0))
                T.copy(weight_flat[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, acc)
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    val = acc[i, j] * bn_scale[f] + bn_bias[f]
                    out_shared[i, j] = T.cast(val, T.float16)
            T.copy(out_shared, output_flat[by * block_M, bx * block_N])
    return main


# ---------------------------------------------------------------------------
# Kernel D2: 1×1 Conv+BN ONLY (no IF) — compute-bound
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def conv1x1_bn_t4_kernel(
    TB, C_in, H, W, F, S,
    block_M, block_N, block_K, num_stages, threads,
):
    """1×1 Conv+BN without neuron."""
    OH = (H + S - 1) // S
    OW = (W + S - 1) // S
    M = TB * OH * OW

    @T.prim_func
    def main(
        data:     T.Tensor((TB, H, W, C_in), T.float16),
        weight:   T.Tensor((C_in, F), T.float16),
        bn_scale: T.Tensor((F,), T.float32),
        bn_bias:  T.Tensor((F,), T.float32),
        output:   T.Tensor((TB, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            data_shared   = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            acc           = T.alloc_fragment((block_M, block_N), T.float32)
            output_flat = T.Tensor((M, F), T.float16, output.data)
            T.clear(acc)
            for k_iter in T.Pipelined(
                T.ceildiv(C_in, block_K), num_stages=num_stages,
            ):
                for i, j in T.Parallel(block_M, block_K):
                    cin = k_iter * block_K + j
                    m = by * block_M + i
                    n_idx = m // (OH * OW)
                    hw = m % (OH * OW)
                    oh = hw // OW
                    ow = hw % OW
                    ib = (m < M) and (cin < C_in)
                    data_shared[i, j] = T.if_then_else(
                        ib, data[n_idx, oh * S, ow * S, cin], T.float16(0))
                T.copy(weight[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, acc)
            out_shared = T.alloc_shared((block_M, block_N), T.float16)
            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    val = acc[i, j] * bn_scale[f] + bn_bias[f]
                    out_shared[i, j] = T.cast(val, T.float16)
            T.copy(out_shared, output_flat[by * block_M, bx * block_N])
    return main


# ---------------------------------------------------------------------------
# Kernel D3: Standalone IF neuron — memory-bound
# ---------------------------------------------------------------------------

@tilelang.jit(out_idx=[-1])
def if_neuron_t4_kernel(
    TB, OH, OW, F,
    block_M, block_N, threads=128,
    v_threshold=1.0, v_reset=0.0, T_steps=4,
):
    """Standalone IF neuron. Memory-bound: read FP16 → update FP32 membrane → write FP16 spikes."""
    M = TB * OH * OW
    B = TB // T_steps
    spatial = B * OH * OW

    @T.prim_func
    def main(
        bn_out:  T.Tensor((TB, OH, OW, F), T.float16),
        state:   T.Tensor((B, OH, OW, F), T.float32),
        spikes:  T.Tensor((TB, OH, OW, F), T.float16),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            bn_flat     = T.Tensor((M, F), T.float16, bn_out.data)
            spikes_flat = T.Tensor((M, F), T.float16, spikes.data)
            state_flat  = T.Tensor((spatial, F), T.float32, state.data)

            for i, j in T.Parallel(block_M, block_N):
                m = by * block_M + i
                f = bx * block_N + j
                if m < M and f < F:
                    x_val = T.cast(bn_flat[m, f], T.float32)
                    s_idx = m % spatial
                    v = state_flat[s_idx, f]
                    h = v + x_val
                    spike = T.if_then_else(
                        h >= v_threshold, T.float32(1), T.float32(0))
                    v_new = (T.float32(1) - spike) * h + spike * T.float32(v_reset)
                    state_flat[s_idx, f] = v_new
                    spikes_flat[m, f] = T.cast(spike, T.float16)
    return main
