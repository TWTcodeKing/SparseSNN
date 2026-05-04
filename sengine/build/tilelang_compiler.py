"""TileLang kernel compilation from EngineIR nodes.

Bridges the gap between EngineIR (abstract op descriptions) and concrete
TileLang kernel functions. Groups nodes by unique shape key so identical
layers share the same compiled kernel module.

Usage:
    compiler = TileLangCompiler(ir, T=4, batch_size=16)
    kernels = compiler.compile_all()
    # kernels: dict[node_id → callable kernel function]
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import torch

from sengine.ir import (
    OpType, KernelVariant, Node, EngineIR,
)
from sengine.logger import logger


# ─── Lazy-loaded TileLang kernel constructors ───

_conv2d_bn_t4 = None
_conv1x1_bn_t4 = None
_stem_conv_bn_if_t4 = None
_linear_bn = None
_linear_bn_lif_t4 = None
_matmul_kernel = None
_matmul_lif_kernel = None
_ext_if = None


def _load_conv_kernels():
    global _conv2d_bn_t4, _conv1x1_bn_t4, _stem_conv_bn_if_t4
    if _conv2d_bn_t4 is not None:
        return
    from sengine.kernels.conv2d_bn_if_t4 import (
        conv2d_bn_t4_kernel,
        conv1x1_bn_t4_kernel,
        stem_conv_bn_if_t4_kernel,
    )
    _conv2d_bn_t4 = conv2d_bn_t4_kernel
    _conv1x1_bn_t4 = conv1x1_bn_t4_kernel
    _stem_conv_bn_if_t4 = stem_conv_bn_if_t4_kernel


def _load_linear_kernels():
    global _linear_bn, _linear_bn_lif_t4, _matmul_kernel, _matmul_lif_kernel
    if _linear_bn is not None:
        return
    from sengine.kernels.spikformer_kernels import (
        linear_bn_kernel,
        linear_bn_lif_t4_kernel,
        matmul_kernel,
        matmul_lif_kernel,
    )
    _linear_bn = linear_bn_kernel
    _linear_bn_lif_t4 = linear_bn_lif_t4_kernel
    _matmul_kernel = matmul_kernel
    _matmul_lif_kernel = matmul_lif_kernel


def _load_cuda_if():
    global _ext_if
    if _ext_if is not None:
        return
    # Auto-detect CUDA home and GPU arch
    import torch
    if 'CUDA_HOME' not in os.environ:
        for p in ['/usr/local/cuda', '/usr/local/cuda-12.8', '/usr/local/cuda-12']:
            if os.path.exists(os.path.join(p, 'bin', 'nvcc')):
                os.environ['CUDA_HOME'] = p
                break
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        props = torch.cuda.get_device_properties(0)
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{props.major}.{props.minor}'
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), '..', 'csrc', 'green_context', 'if_neuron.cu')
    _ext_if = load(
        name='if_neuron_ext',
        sources=[src],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False,
    )


def get_cuda_if():
    """Return the compiled CUDA vec4 IF neuron extension."""
    _load_cuda_if()
    return _ext_if


# ─── Hardware detection ───

_HW_INFO = None

def _get_hw_info() -> dict:
    """Detect GPU hardware capabilities (cached)."""
    global _HW_INFO
    if _HW_INFO is not None:
        return _HW_INFO
    try:
        props = torch.cuda.get_device_properties(0)
        _HW_INFO = {
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_arch': f"sm_{props.major}{props.minor}",
            'sm_count': props.multi_processor_count,
            'max_smem': props.max_shared_memory_per_block_optin
                        if hasattr(props, 'max_shared_memory_per_block_optin')
                        else 100 * 1024,
        }
    except Exception:
        _HW_INFO = {
            'gpu_name': 'unknown', 'gpu_arch': 'sm_89',
            'sm_count': 128, 'max_smem': 100 * 1024,
        }
    return _HW_INFO


# ─── Config selection ───

def _pick_config(M: int, K_red: int, F: int) -> dict:
    """Pick a reasonable default tile config for a GEMM problem.

    Returns dict with block_M, block_N, block_K, num_stages, threads.
    """
    if M >= 100000:
        bm = 128
    elif M >= 10000:
        bm = 64
    else:
        bm = 32

    bn = min(64, F) if F > 0 else 64
    bk = min(32, K_red) if K_red > 0 else 32
    ns = 2
    thr = 128

    return dict(block_M=bm, block_N=bn, block_K=bk, num_stages=ns, threads=thr)


def _autotune_config(compile_fn, profile_args: tuple, M: int, K_red: int, F: int,
                     n_profile: int = 200) -> dict:
    """Try multiple tile configs and return the fastest.

    Hardware-adaptive: uses detected SM count and max smem to prune config space.
    """
    hw = _get_hw_info()
    smem_limit = min(hw['max_smem'], 100 * 1024)  # conservative
    sm_count = hw['sm_count']

    candidates = []
    for bm in [32, 64, 128, 256]:
        for bn in [32, 64, 128]:
            for bk in [32, 64]:
                for ns in [2, 3]:
                    for thr in [128, 256]:
                        if bk > K_red or bn > F * 2 or bm > M:
                            continue
                        smem = (bm * bk + bk * bn) * 2 * ns
                        if smem > smem_limit:
                            continue
                        # Require minimum parallelism: at least sm_count/4 tiles
                        n_tiles = ((M + bm - 1) // bm) * ((F + bn - 1) // bn)
                        if n_tiles < sm_count // 4:
                            continue
                        min_threads = max(bm, bn) // 2
                        if thr < min_threads:
                            continue
                        candidates.append(dict(block_M=bm, block_N=bn, block_K=bk,
                                               num_stages=ns, threads=thr))

    best_us = float('inf')
    best_cfg = _pick_config(M, K_red, F)

    for cfg in candidates:
        try:
            kern = compile_fn(cfg)
            # Warmup
            for _ in range(20):
                kern(*profile_args)
            torch.cuda.synchronize()
            # Profile
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_profile):
                kern(*profile_args)
            e.record()
            torch.cuda.synchronize()
            us = s.elapsed_time(e) / n_profile * 1000
            if us < best_us:
                best_us = us
                best_cfg = cfg
        except Exception:
            continue

    best_cfg['latency_us'] = best_us
    return best_cfg


# ─── Shape key generation ───

def _conv_shape_key(cp, H: int, W: int, TB: int) -> str:
    """Unique key for a Conv layer shape."""
    return (f"conv{cp.kernel_h}x{cp.kernel_w}_{cp.in_channels}_{cp.out_channels}"
            f"_{H}x{W}_s{cp.stride_h}_TB{TB}")


def _linear_shape_key(K: int, N: int, T: int = 0, B: int = 0) -> str:
    return f"linear_{K}_{N}_T{T}_B{B}"


def _matmul_shape_key(M: int, K: int, N: int) -> str:
    return f"matmul_{M}_{K}_{N}"


# ─── Compiler ───

class TileLangCompiler:
    """Compile TileLang kernels for all nodes in an EngineIR.

    Nodes with identical shapes share the same compiled kernel.
    """

    def __init__(self, ir: EngineIR, T: int, batch_size: int,
                 autotune: bool = False, tuning_cache=None):
        self.ir = ir
        self.T = T
        self.B = batch_size
        self.TB = T * batch_size
        self.autotune = autotune
        self.tuning_cache = tuning_cache

        # shape_key → compiled kernel callable
        self._kernel_cache: dict[str, object] = {}
        # shape_key → config dict
        self._config_cache: dict[str, dict] = {}

    def compile_all(self) -> dict[int, object]:
        """Compile kernels for all TileLang-assigned nodes.

        Returns:
            dict mapping node_id → callable kernel function (or the ext_if module
            for CUDAVec4IF nodes).
        """
        t0 = time.time()
        kernels: dict[int, object] = {}
        compiled_count = 0
        cached_count = 0

        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            kv = node.assigned_kernel

            if kv == KernelVariant.TileLangConvBN:
                kern, is_new = self._get_conv_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangConv1x1BN:
                kern, is_new = self._get_conv1x1_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangStemConvBN:
                kern, is_new = self._get_stem_conv_bn(node)
                kernels[nid] = kern
            elif kv in (KernelVariant.TileLangFusedConvBNIF,
                        KernelVariant.TileLangFusedConv1x1BNIF):
                kern, is_new = self._get_fused_conv_bn_if(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangDWConvBN:
                kern, is_new = self._get_dwconv_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangFusedDWConvBNIF:
                kern, is_new = self._get_dwconv_bn_if(node)
                kernels[nid] = kern
            elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                kernels[nid] = get_cuda_if()
                is_new = False
            elif kv == KernelVariant.TileLangLinearBN:
                kern, is_new = self._get_linear_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangLinearBNLIF:
                kern, is_new = self._get_linear_bn_lif(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangGroupedConvBN:
                kern, is_new = self._get_grouped_conv_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangMatMul:
                kern, is_new = self._get_matmul(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangMatMulScale:
                kern, is_new = self._get_matmul_scale(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangFusedMatMulLIF:
                kern, is_new = self._get_fused_matmul_lif(node)
                kernels[nid] = kern
            elif kv in (KernelVariant.FusedSpikformerAttn,
                        KernelVariant.FusedMaxformerAttn,
                        KernelVariant.FusedDSSAAttn,
                        KernelVariant.FusedTokenQKAttn):
                result = self._get_fused_attn_kernels(node)
                kernels[nid] = result  # (gemm1, gemm2) tuple or None
                is_new = result is not None
            else:
                is_new = False

            if is_new:
                compiled_count += 1
            elif nid in kernels:
                cached_count += 1

        elapsed = time.time() - t0
        logger.phase("COMPILE", "Compiled %d unique kernels, %d cache hits (%.1fs)",
                     compiled_count, cached_count, elapsed)
        return kernels

    def get_config(self, node: Node) -> Optional[dict]:
        """Get the tile config used for a node (after compile_all)."""
        return node.tilelang_config

    # ─── Conv+BN (3×3 and general) ───

    def _get_conv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = _conv_shape_key(cp, H, W, self.TB)

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_conv_kernels()
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M = self.TB * OH * OW
        K_red = cp.kernel_h * cp.kernel_w * cp.in_channels

        cfg = self._resolve_config(key, M, K_red, cp.out_channels)
        logger.debug("  Conv+BN %d→%d %dx%d s=%d M=%d cfg=%dx%dx%d",
                     cp.in_channels, cp.out_channels, H, W, cp.stride_h, M,
                     cfg['block_M'], cfg['block_N'], cfg['block_K'])

        kern = _conv2d_bn_t4(
            TB=self.TB, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
            K=cp.kernel_h, S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Per-timestep fused Conv+BN+IF (T=1 per launch, large batch) ───

    def _get_fused_conv_bn_if(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"fused_conv_if_{cp.in_channels}_{cp.out_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_B{self.B}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        from sengine.kernels.conv2d_bn_if_t4 import (
            conv2d_bn_if_t4_kernel, conv1x1_bn_if_t4_kernel,
        )

        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M = self.B * OH * OW
        K_red = cp.kernel_h * cp.kernel_w * cp.in_channels

        cfg = self._resolve_config(key, M, K_red, cp.out_channels)
        logger.debug("  Fused Conv+BN+IF %d→%d %dx%d B=%d M=%d cfg=%dx%dx%d",
                     cp.in_channels, cp.out_channels, H, W, self.B, M,
                     cfg['block_M'], cfg['block_N'], cfg['block_K'])

        # Use the existing fused T4 kernels with T_steps=1 (no temporal race)
        if cp.kernel_h == 1 and cp.kernel_w == 1:
            kern = conv1x1_bn_if_t4_kernel(
                TB=self.B, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                S=cp.stride_h, T_steps=1,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        else:
            kern = conv2d_bn_if_t4_kernel(
                TB=self.B, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                K=cp.kernel_h, S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                T_steps=1,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── 1×1 Conv+BN ───

    def _get_conv1x1_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"conv1x1_{cp.in_channels}_{cp.out_channels}_{H}x{W}_s{cp.stride_h}_TB{self.TB}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_conv_kernels()
        OH = (H + cp.stride_h - 1) // cp.stride_h
        OW = (W + cp.stride_w - 1) // cp.stride_w
        M = self.TB * OH * OW

        cfg = self._resolve_config(key, M, cp.in_channels, cp.out_channels)

        kern = _conv1x1_bn_t4(
            TB=self.TB, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
            S=cp.stride_h,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Stem Conv+BN (7×7, padded C_in) ───

    def _get_stem_conv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"stem_{cp.in_channels}_{cp.out_channels}_{H}x{W}_TB{self.TB}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_conv_kernels()
        C_padded = 16  # pad C_in=3 to 16 for tensor core alignment
        OH = (H + 2 * cp.pad_h - cp.kernel_h) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.kernel_w) // cp.stride_w + 1
        M = self.TB * OH * OW
        K_red = cp.kernel_h * cp.kernel_w * C_padded

        cfg = self._resolve_config(key, M, K_red, cp.out_channels)

        kern = _stem_conv_bn_if_t4(
            TB=self.TB, H=H, W=W,
            C_in_padded=C_padded, C_in_real=cp.in_channels, F=cp.out_channels,
            KH=cp.kernel_h, KW=cp.kernel_w, S=cp.stride_h, P=cp.pad_h,
            T_steps=self.T,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Linear+BN (transformer) ───

    def _get_linear_bn(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        key = _linear_shape_key(K, N, self.T, self.B)

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        kern = _linear_bn(
            M=M, K=K, N_out=N,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Linear+BN+LIF (fused, for MLP chains) ───

    def _get_linear_bn_lif(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        spatial = M // self.T
        key = f"linear_bn_lif_{M}_{K}_{N}_T{self.T}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        kern = _linear_bn_lif_t4(
            M=M, K=K, N_out=N,
            T_steps=self.T, spatial=spatial,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Pure MatMul ───

    @staticmethod
    def _matmul_dims_ok(M, K, N, min_dim=8):
        """Check if matmul dimensions satisfy TileLang tensor-core requirements.

        Each dimension must be at least 8 (half-warp MMA tile) and the
        reduction dimension K must be a multiple of 8 for FP16 tensor cores.
        """
        return M >= min_dim and K >= min_dim and N >= min_dim and K % 8 == 0

    def _get_matmul(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        key = _matmul_shape_key(M, K, N)

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        try:
            kern = _matmul_kernel(
                M=M, K=K, N=N,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    # ─── MatMul + Scale (attention Q@K^T) ───

    def _get_matmul_scale(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        scale = node.extra_attrs.get("scale_value", 1.0)
        key = f"matmul_scale_{M}_{K}_{N}_s{scale}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        try:
            kern = _matmul_kernel(
                M=M, K=K, N=N, scale=scale,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    # ─── Fused MatMul + LIF (attention attn@V + neuron) ───

    def _get_fused_matmul_lif(self, node: Node) -> tuple[object, bool]:
        M_full, K, N = self._get_linear_dims(node)
        M = M_full // self.T if self.T > 0 else M_full  # per-timestep
        spatial = M
        key = f"fused_matmul_lif_{M}_{K}_{N}_B{self.B}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        np = node.neuron_params
        v_threshold = np.v_threshold if np else 1.0
        v_reset = np.v_reset if np else 0.0
        recip_tau = 1.0 / np.tau if (np and np.tau and np.tau > 0) else 0.5

        try:
            kern = _matmul_lif_kernel(
                M=M, K=K, N=N,
                T_steps=1, spatial=spatial,
                v_threshold=v_threshold, v_reset=v_reset, recip_tau=recip_tau,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    # ─── Fused Attention sub-kernels (batched MatMul) ───

    def _get_fused_attn_kernels(self, node: Node):
        """Compile batched MatMul kernels for fused attention.

        Returns (gemm1, gemm2) tuple of TileLang JITKernel objects,
        or None if compilation fails.

        gemm1: first matmul (e.g., Q@K^T with scale for SpikFormer)
        gemm2: second matmul (e.g., attn@V for SpikFormer)
        """
        ap = node.attention_params
        if ap is None:
            return None

        # Derive dimensions from attention params + input shapes
        heads = ap.num_heads
        hd = ap.head_dim
        C = heads * hd
        if not node.input_shapes or not node.input_shapes[0]:
            return None
        shape0 = node.input_shapes[0]
        TB = shape0[0] if shape0 else self.TB
        if TB <= 0 or C <= 0:
            return None
        batch = TB * heads

        if ap.variant == "spikformer":
            # Input: (TB, N, C). N = total_elems / (TB * C)
            total = 1
            for d in shape0:
                total *= d
            N = total // (TB * C) if (TB * C) > 0 else 0
            if N <= 0:
                return None
            # GEMM1: attn_scores = Q @ K^T, per head: (N,hd)@(hd,N)=(N,N)
            g1_batch, g1_M, g1_K, g1_N = batch, N, hd, N
            g1_scale = ap.scale
            # GEMM2: out = attn_scores @ V, per head: (N,N)@(N,hd)=(N,hd)
            g2_batch, g2_M, g2_K, g2_N = batch, N, N, hd
            g2_scale = 1.0
        elif ap.variant == "maxformer":
            H, W = ap.H, ap.W
            N = H * W
            # GEMM1: kv = K^T @ V, per head: (hd,N)@(N,hd)=(hd,hd)
            g1_batch, g1_M, g1_K, g1_N = batch, hd, N, hd
            g1_scale = 1.0
            # GEMM2: out = Q @ kv * scale, per head: (N,hd)@(hd,hd)=(N,hd)
            g2_batch, g2_M, g2_K, g2_N = batch, N, hd, hd
            g2_scale = ap.scale
        elif ap.variant == "dssa":
            N_q = ap.H * ap.W
            # Derive spatial_kv from first input shape
            s1 = node.input_shapes[1] if len(node.input_shapes) > 1 else shape0
            spatial_kv = 1
            for d in s1[1:]:
                spatial_kv *= d
            spatial_kv = spatial_kv // C if C > 0 else 1
            # GEMM1: attn = K^T @ Q, per head: (spatial,hd)^T@(hd,N_q) = (spatial,N_q)? No.
            # K^T: (hd, spatial)^T = (spatial, hd). Actually:
            # K is (heads, hd, spatial), K^T is (heads, spatial, hd)
            # K^T @ Q: (spatial,hd) @ (hd,N_q) = (spatial, N_q)
            g1_batch, g1_M, g1_K, g1_N = batch, spatial_kv, hd, N_q
            g1_scale = 1.0  # scale1 is a tensor, applied separately
            # GEMM2: out = V @ attn, per head: (hd,spatial)@(spatial,N_q) = (hd,N_q)
            g2_batch, g2_M, g2_K, g2_N = batch, hd, spatial_kv, N_q
            g2_scale = 1.0  # scale2 is a tensor, applied separately
        elif ap.variant == "token_qk":
            # TokenQK: no matmul, only sum+mul. No TileLang GEMM needed.
            return None
        else:
            return None

        if ap.variant == "maxformer":
            from sengine.kernels.fused_attention_kernels import (
                maxformer_kTv_kernel, maxformer_qkv_kernel)

            key1 = f"maxformer_kTv_{TB}_{heads}_{hd}_{N}_{H}_{W}"
            key2 = f"maxformer_qkv_lif_{TB}_{heads}_{hd}_{N}_{H}_{W}_s{ap.scale}"

            gemm1 = self._kernel_cache.get(key1)
            if gemm1 is None:
                cfg1 = self._resolve_config(key1, hd, N, hd)
                try:
                    gemm1 = maxformer_kTv_kernel(
                        TB=TB, heads=heads, hd=hd, N=N, H=H, W=W,
                        **{k: cfg1[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                except Exception:
                    gemm1 = None
                self._kernel_cache[key1] = gemm1

            gemm2 = self._kernel_cache.get(key2)
            if gemm2 is None:
                cfg2 = self._resolve_config(key2, N, hd, hd)
                try:
                    gemm2 = maxformer_qkv_kernel(
                        TB=TB, heads=heads, hd=hd, N=N, H=H, W=W,
                        scale=ap.scale,
                        **{k: cfg2[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                except Exception:
                    gemm2 = None
                self._kernel_cache[key2] = gemm2

        elif ap.variant == "spikformer":
            from sengine.kernels.spikformer_kernels import (
                batched_matmul_kernel, batched_matmul_bt_kernel)

            key1 = f"batched_mm_bt_{batch}_{g1_M}_{g1_K}_{g1_N}_s{g1_scale}"
            gemm1 = self._kernel_cache.get(key1)
            if gemm1 is None:
                cfg1 = self._resolve_config(key1, g1_M, g1_K, g1_N)
                try:
                    gemm1 = batched_matmul_bt_kernel(
                        batch=batch, M=g1_M, K=g1_K, N=g1_N, scale=g1_scale,
                        **{k: cfg1[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                except Exception:
                    gemm1 = None
                self._kernel_cache[key1] = gemm1

            key2 = f"batched_mm_{batch}_{g2_M}_{g2_K}_{g2_N}_s{g2_scale}"
            gemm2 = self._kernel_cache.get(key2)
            if gemm2 is None:
                cfg2 = self._resolve_config(key2, g2_M, g2_K, g2_N)
                try:
                    gemm2 = batched_matmul_kernel(
                        batch=batch, M=g2_M, K=g2_K, N=g2_N, scale=g2_scale,
                        **{k: cfg2[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                except Exception:
                    gemm2 = None
                self._kernel_cache[key2] = gemm2
        else:
            # DSSA and TokenQK: TODO — for now return None (Python dispatch)
            gemm1 = gemm2 = None

        if gemm1 is None or gemm2 is None:
            return None
        return (gemm1, gemm2)

    # ─── Depthwise Conv+BN ───

    def _get_dwconv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"dwconv_{cp.in_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_TB{self.TB}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.dwconv_bn import dwconv_bn_kernel
        logger.debug("  DWConv+BN %d %dx%d s=%d",
                     cp.in_channels, H, W, cp.stride_h)

        kern = dwconv_bn_kernel(
            TB=self.TB, C=cp.in_channels, H=H, W=W,
            K=cp.kernel_h, S=cp.stride_h, P=cp.pad_h)

        self._kernel_cache[key] = kern
        return kern, True

    def _get_dwconv_bn_if(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"dwconv_if_{cp.in_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_B{self.B}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.dwconv_bn import dwconv_bn_if_kernel
        logger.debug("  DWConv+BN+IF %d %dx%d s=%d",
                     cp.in_channels, H, W, cp.stride_h)

        kern = dwconv_bn_if_kernel(
            TB=self.B, C=cp.in_channels, H=H, W=W,
            K=cp.kernel_h, S=cp.stride_h, P=cp.pad_h,
            T_steps=1)

        self._kernel_cache[key] = kern
        return kern, True

    # ─── Grouped Conv+BN ───

    def _get_grouped_conv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"gconv_{cp.in_channels}_{cp.out_channels}_{cp.kernel_h}x{cp.kernel_w}" \
              f"_s{cp.stride_h}_g{cp.groups}_{H}x{W}_TB{self.TB}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache.get(key, {})
            return self._kernel_cache[key], False

        C_in_per_g = cp.in_channels // cp.groups
        K_red = cp.kernel_h * cp.kernel_w * C_in_per_g
        C_out_per_g = cp.out_channels // cp.groups
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M = self.TB * OH * OW

        cfg = self._resolve_config(key, M, K_red, C_out_per_g)

        from sengine.kernels.grouped_conv_bn import grouped_conv_bn_kernel

        try:
            kern = grouped_conv_bn_kernel(
                TB=self.TB, C_in=cp.in_channels, H=H, W=W,
                C_out=cp.out_channels, K=cp.kernel_h,
                S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                groups=cp.groups,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                        'num_stages', 'threads')})
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        return kern, kern is not None

    # ─── Helpers ───

    def _get_spatial(self, node: Node) -> tuple[int, int]:
        """Extract (H, W) from a Conv node's input shape."""
        if node.input_shapes:
            s = node.input_shapes[0]
            if len(s) == 4:
                return s[2], s[3]  # NCHW: (N, C, H, W)
        return 0, 0

    def _get_linear_dims(self, node: Node) -> tuple[int, int, int]:
        """Extract (M, K, N) for a Linear/MatMul node.

        For Linear (weight in second input): K = weight shape[0]
        For dynamic MatMul (both activations): K = first input last dim

        All leading dims are flattened into M for the 2D GEMM kernel.
        """
        M, K, N = 0, 0, 0

        if len(node.input_shapes) >= 2:
            s0 = node.input_shapes[0]   # first input (activation or Q)
            s1 = node.input_shapes[1]   # second input (weight or K^T)

            if len(s1) == 2:
                # Second input is 2D weight (K_in, N_out) — Linear layer
                K = s1[0]
                N = s1[1]
                M = 1
                for d in s0:
                    M *= d
                M = M // K if K > 0 else M  # M = total_elements / K
            else:
                # Both inputs are activations — batched MatMul
                # K = last dim of first input = second-to-last of second
                K = s0[-1]
                N = s1[-1]
                M = 1
                for d in s0[:-1]:
                    M *= d
        elif node.input_shapes:
            s = node.input_shapes[0]
            if len(s) >= 2:
                K = s[-1]
                M = 1
                for d in s[:-1]:
                    M *= d

        if node.gemm_params:
            N = node.gemm_params.get('N', N)
        elif N == 0 and node.output_shapes and len(node.output_shapes[0]) >= 2:
            N = node.output_shapes[0][-1]

        return M, K, N

    def _resolve_config(self, key: str, M: int, K_red: int, F: int,
                        compile_fn=None, profile_args=None) -> dict:
        """Get tile config from cache, autotune, or heuristic.

        Priority: cache hit → autotune (if enabled) → heuristic default.
        """
        hw = _get_hw_info()
        gpu_name = hw['gpu_name']
        gpu_arch = hw['gpu_arch']

        # Check tuning cache first
        if self.tuning_cache:
            cached = self.tuning_cache.get(key, gpu_name, gpu_arch, self.T, self.B)
            if cached is not None:
                return cached

        # Autotune if enabled and compile_fn provided
        if self.autotune and compile_fn is not None and profile_args is not None:
            logger.debug("  Autotuning %s (M=%d, K=%d, F=%d)...", key, M, K_red, F)
            cfg = _autotune_config(compile_fn, profile_args, M, K_red, F)
            if self.tuning_cache:
                self.tuning_cache.put(key, gpu_name, cfg, gpu_arch, self.T, self.B)
            return cfg

        # Heuristic default
        return _pick_config(M, K_red, F)
