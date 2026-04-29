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
    global _linear_bn, _linear_bn_lif_t4, _matmul_kernel
    if _linear_bn is not None:
        return
    from sengine.kernels.spikformer_kernels import (
        linear_bn_kernel,
        linear_bn_lif_t4_kernel,
        matmul_kernel,
    )
    _linear_bn = linear_bn_kernel
    _linear_bn_lif_t4 = linear_bn_lif_t4_kernel
    _matmul_kernel = matmul_kernel


def _load_cuda_if():
    global _ext_if
    if _ext_if is not None:
        return
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    from torch.utils.cpp_extension import load
    _ext_if = load(
        name='if_neuron_ext',
        sources=['sengine/csrc/green_context/if_neuron.cu'],
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
            elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                kernels[nid] = get_cuda_if()
                is_new = False
            elif kv == KernelVariant.TileLangLinearBN:
                kern, is_new = self._get_linear_bn(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangLinearBNLIF:
                kern, is_new = self._get_linear_bn_lif(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangMatMul:
                # Dynamic matmul (attention) — use torch.matmul fallback
                # TileLang requires dimensions aligned to tensor core tiles,
                # which attention dims (e.g. N_patches=196) often violate.
                kernels[nid] = None  # signals torch.matmul fallback in runtime
                is_new = False
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

    def _get_matmul(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        key = _matmul_shape_key(M, K, N)

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        kern = _matmul_kernel(
            M=M, K=K, N=N,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Helpers ───

    def _get_spatial(self, node: Node) -> tuple[int, int]:
        """Extract (H, W) from a Conv node's input shape."""
        if node.input_shapes:
            s = node.input_shapes[0]
            if len(s) == 4:
                return s[2], s[3]  # NCHW: (N, C, H, W)
        return 0, 0

    def _get_linear_dims(self, node: Node) -> tuple[int, int, int]:
        """Extract (M, K, N) for a Linear/MatMul node."""
        M, K = 0, 0
        if node.input_shapes:
            s = node.input_shapes[0]
            if len(s) == 2:
                M, K = s
            elif len(s) == 3:
                # 3D: (batch, seq_len, features) → flatten to M = batch*seq
                M = s[0] * s[1]
                K = s[2]
            elif len(s) == 4:
                M = s[0] * s[2] * s[3]
                K = s[1]
        if node.gemm_params:
            N = node.gemm_params.get('N', 0)
        elif node.output_shapes and len(node.output_shapes[0]) >= 2:
            N = node.output_shapes[0][-1]
        else:
            N = 0
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
