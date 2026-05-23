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
    OpType, KernelVariant, BoundType, Node, EngineIR,
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
        for p in ['/usr/local/cuda-12.8', '/usr/local/cuda-12.6', '/usr/local/cuda-12', '/usr/local/cuda']:
            if os.path.exists(os.path.join(p, 'bin', 'nvcc')):
                os.environ['CUDA_HOME'] = p
                break
    # Ensure CUDA_HOME/bin is on PATH so TileLang's nvcc finds the right version
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        cuda_bin = os.path.join(cuda_home, 'bin')
        if cuda_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = cuda_bin + ':' + os.environ.get('PATH', '')
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


# ─── Occupancy estimation ───

def estimate_occupancy(block_M, block_N, block_K, num_stages, threads,
                       n_membranes=1):
    """Estimate max CTAs per SM for an interleaved kernel.

    Auto-detects GPU resource limits — works on any arch (A100, RTX 4090, etc.).
    The estimate is conservative — actual occupancy may differ due to compiler
    register allocation.
    """
    hw = _get_hw_info()

    # Shared memory: pipeline buffers + output staging
    smem_pipeline = (block_M * block_K + block_K * block_N) * 2 * num_stages
    smem_output = block_M * block_N * 2  # output shared buffer
    smem_total = smem_pipeline + smem_output

    # Register estimate: accumulator + membrane fragments + temporaries
    elems_per_thread = (block_M * block_N) // threads
    regs_acc = elems_per_thread       # FP32 accumulator fragment
    regs_mem = elems_per_thread * n_membranes  # membrane fragment(s)
    regs_tmp = 10                     # BN, spike, h, indices, etc.
    regs_per_thread = regs_acc + regs_mem + regs_tmp

    # Read limits from detected hardware
    max_smem_per_sm = hw.get('max_smem', 100 * 1024)
    max_regs_per_sm = 65536  # same for all SM 8.x+
    max_blocks_per_sm = 16   # same for SM 8.x, 32 for SM 9.x

    # max_threads_per_sm: read from calibration cache if available
    try:
        from sengine.tuning.hw_calibrate import calibrate
        cal = calibrate()
        max_threads_per_sm = cal.get('max_threads_per_sm', 1536)
        max_regs_per_sm = cal.get('max_regs_per_sm', 65536)
        max_blocks_per_sm = cal.get('max_blocks_per_sm', 16)
    except Exception:
        max_threads_per_sm = 1536

    ctas_by_smem = max_smem_per_sm // max(smem_total, 1)
    ctas_by_regs = max_regs_per_sm // max(regs_per_thread * threads, 1)
    ctas_by_threads = max_threads_per_sm // max(threads, 1)

    return min(ctas_by_smem, ctas_by_regs, ctas_by_threads, max_blocks_per_sm)


# ─── Config selection ───

def _pick_interleaved_config(M: int, K_red: int, F: int,
                              sm_count: int = 128,
                              n_membranes: int = 1) -> dict:
    """Pick tile config optimized for interleaved kernels.

    Architecture-adaptive: uses larger tiles, deeper pipeline, and more threads
    on GPUs with higher bandwidth and shared memory (A100, H100).
    """
    hw = _get_hw_info()
    max_smem = hw.get('max_smem', 100 * 1024)
    is_high_bw = max_smem >= 160 * 1024  # A100 (164KB) / H100 (228KB)

    if is_high_bw:
        # A100+: prefer larger tiles + deeper pipeline for high BW utilization
        tile_prefs = [(64, 128), (128, 64), (64, 64), (32, 64)]
        bk = min(64, K_red)
        ns = 3
        thr = 128
    else:
        # RTX 4090 / consumer: moderate tiles, 2-stage pipeline
        tile_prefs = [(32, 64), (64, 64), (64, 32)]
        bk = min(32, K_red)
        ns = 2
        thr = 128

    for bm, bn in tile_prefs:
        if bn > max(F, 32):
            continue
        smem = (bm * bk + bk * bn) * 2 * ns + bm * bn * 2
        if smem > max_smem:
            continue
        grid = ((M + bm - 1) // bm) * ((F + bn - 1) // bn)
        if grid >= sm_count:
            return dict(block_M=bm, block_N=bn, block_K=bk,
                        num_stages=ns, threads=thr)

    return dict(block_M=32, block_N=64, block_K=min(32, K_red),
                num_stages=ns, threads=128)


def _pick_config(M: int, K_red: int, F: int, bpe: int = 2) -> dict:
    """Pick a reasonable default tile config for a GEMM problem.

    Architecture-adaptive: A100+ gets larger tiles, deeper pipeline, wider threads.

    Args:
        bpe: bytes per element (2 for fp16, 4 for fp32).

    Returns dict with block_M, block_N, block_K, num_stages, threads.
    """
    hw = _get_hw_info()
    max_smem = hw.get('max_smem', 100 * 1024)
    is_high_bw = max_smem >= 160 * 1024

    if M >= 100000:
        bm = 128
    elif M >= 10000:
        bm = 64
    else:
        bm = 32

    bn = min(64, F) if F > 0 else 64
    bk = min(64 if is_high_bw else 32, K_red) if K_red > 0 else 32
    ns = 3 if is_high_bw else 2
    thr = 128

    # On high-BW GPUs, prefer wider output tile
    if is_high_bw and F >= 128 and bpe == 2:
        bn = min(128, F)

    # FP32: TileLang can't generate float32xN vector types for N>8.
    if bpe == 4:
        while (bm * bk // thr > 8 or bn * bk // thr > 8) and bn > 32:
            bn //= 2
        while bm * bk // thr > 8 and bm > 16:
            bm //= 2
        while bm * bk // thr > 8 and bk > 16:
            bk //= 2

    # Verify smem fits
    smem = (bm * bk + bk * bn) * bpe * ns + bm * bn * bpe
    if smem > max_smem:
        ns = 2
        bk = min(32, K_red) if K_red > 0 else 32

    return dict(block_M=bm, block_N=bn, block_K=bk, num_stages=ns, threads=thr)


def _autotune_config(compile_fn, profile_args: tuple, M: int, K_red: int, F: int,
                     n_profile: int = 200) -> dict:
    """Try multiple tile configs and return the fastest.

    Hardware-adaptive: uses detected SM count and max smem to prune config space.
    """
    hw = _get_hw_info()
    smem_limit = hw['max_smem']  # use real limit, no artificial cap
    sm_count = hw['sm_count']
    is_high_bw = smem_limit >= 160 * 1024

    candidates = []
    bm_choices = [32, 64, 128, 256]
    bn_choices = [32, 64, 128, 256]
    # K-divisibility-aware block_K choices
    bk_base = {32, 64, 128}
    for bk in [48, 96]:
        if K_red > 0 and K_red % bk == 0 and bk <= K_red:
            bk_base.add(bk)
    bk_choices = sorted(bk_base)
    ns_choices = [2, 3, 4] if is_high_bw else [2, 3]
    thr_choices = [128, 256, 512] if is_high_bw else [128, 256]

    for bm in bm_choices:
        for bn in bn_choices:
            for bk in bk_choices:
                for ns in ns_choices:
                    for thr in thr_choices:
                        if bk > K_red or bn > F * 2 or bm > M:
                            continue
                        if bm * bn < thr:
                            continue
                        smem = (bm * bk + bk * bn) * 2 * ns
                        if smem > smem_limit:
                            continue
                        n_tiles = ((M + bm - 1) // bm) * ((F + bn - 1) // bn)
                        if n_tiles < sm_count // 4:
                            continue
                        candidates.append(dict(block_M=bm, block_N=bn, block_K=bk,
                                               num_stages=ns, threads=thr))

    best_us = float('inf')
    best_cfg = _pick_config(M, K_red, F)

    for cfg in candidates:
        try:
            kern = compile_fn(cfg)
            # Reset CUDA error state before profiling
            torch.cuda.synchronize()
            # Warmup with error detection
            for _ in range(5):
                kern(*profile_args)
            torch.cuda.synchronize()  # catch async errors from warmup
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


def _autotune_interleaved(compile_fn, profile_args: tuple,
                           M: int, K_red: int, F: int,
                           n_membranes: int = 1,
                           n_profile: int = 200) -> dict:
    """Autotune interleaved kernels with occupancy-aware scoring.

    Expands search space to include high-occupancy configs (smaller tiles),
    and uses a composite score: measured_latency × occupancy_penalty.
    Configs with ≥2 CTAs/SM get a bonus; configs with <1 CTA/SM are penalized.
    """
    hw = _get_hw_info()
    smem_limit = hw['max_smem']  # use real limit
    sm_count = hw['sm_count']
    is_high_bw = smem_limit >= 160 * 1024

    candidates = []
    bm_choices = [16, 32, 64, 128, 256]
    bn_choices = [32, 64, 128, 256]
    # K-divisibility-aware block_K choices
    bk_base = {32, 64, 128}
    for bk in [48, 96]:
        if K_red > 0 and K_red % bk == 0 and bk <= K_red:
            bk_base.add(bk)
    bk_choices = sorted(bk_base)
    ns_choices = [2, 3, 4] if is_high_bw else [2, 3]
    thr_choices = [128, 256, 512] if is_high_bw else [128]

    for bm in bm_choices:
        for bn in bn_choices:
            for bk in bk_choices:
                for ns in ns_choices:
                    for thr in thr_choices:
                        if bk > K_red or bn > F * 2 or bm > M:
                            continue
                        if bm * bn < thr:
                            continue
                        smem = (bm * bk + bk * bn) * 2 * ns + bm * bn * 2
                        if smem > smem_limit:
                            continue
                        n_tiles = ((M + bm - 1) // bm) * ((F + bn - 1) // bn)
                        if n_tiles < sm_count // 8:
                            continue
                        occ = estimate_occupancy(bm, bn, bk, ns, thr, n_membranes)
                        candidates.append((dict(block_M=bm, block_N=bn, block_K=bk,
                                                num_stages=ns, threads=thr), occ, n_tiles))

    best_score = float('inf')
    best_cfg = _pick_interleaved_config(M, K_red, F, sm_count, n_membranes)

    for cfg, occ, n_tiles in candidates:
        try:
            kern = compile_fn(cfg)
            torch.cuda.synchronize()
            for _ in range(5):
                kern(*profile_args)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n_profile):
                kern(*profile_args)
            e.record()
            torch.cuda.synchronize()
            us = s.elapsed_time(e) / n_profile * 1000

            # Occupancy-aware scoring:
            # Configs with higher occupancy get a bonus because the E2E
            # pipeline benefits from compute↔memory overlap between CTAs.
            # overlap_ratio ≈ (occ - 1) / occ for occ ≥ 1
            if occ >= 2:
                overlap_bonus = 0.85  # 15% bonus for good overlap
            elif occ >= 1:
                overlap_bonus = 1.0   # neutral
            else:
                overlap_bonus = 1.15  # 15% penalty for poor occupancy

            score = us * overlap_bonus

            if score < best_score:
                best_score = score
                best_cfg = cfg
                best_cfg['latency_us'] = us
        except Exception:
            continue

    if 'latency_us' not in best_cfg:
        best_cfg['latency_us'] = float('inf')
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
                 autotune: bool = False, tuning_cache=None,
                 precision: str = "fp16"):
        self.ir = ir
        self.T = T
        self.B = batch_size
        self.TB = T * batch_size
        self.autotune = autotune
        self.tuning_cache = tuning_cache
        self.precision = precision
        # Derive dtype constants from global precision
        import tilelang.language as _TL
        self.io_dtype_tl = _TL.float32 if precision == "fp32" else _TL.float16
        self.io_dtype_torch = torch.float32 if precision == "fp32" else torch.float16

        # shape_key → compiled kernel callable
        self._kernel_cache: dict[str, object] = {}
        # shape_key → config dict
        self._config_cache: dict[str, dict] = {}

    def compile_all(self, skip_nids: set = None) -> dict[int, object]:
        """Compile kernels for all TileLang-assigned nodes.

        Args:
            skip_nids: Set of node IDs to skip (already compiled externally,
                       e.g., by the fusion validator).

        Returns:
            dict mapping node_id → callable kernel function (or the ext_if module
            for CUDAVec4IF nodes).
        """
        t0 = time.time()
        kernels: dict[int, object] = {}
        compiled_count = 0
        cached_count = 0
        skip_nids = skip_nids or set()

        for nid in self.ir.topo_order:
            if nid in skip_nids:
                continue
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
            elif kv == KernelVariant.TileLangFusedAddLIF:
                kern, is_new = self._get_fused_add_lif(node)
                kernels[nid] = kern
            elif kv == KernelVariant.TileLangFusedPoolLIF:
                kern, is_new = self._get_fused_pool_lif(node)
                kernels[nid] = kern
            elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                # Native CUDA IF/LIF kernels are in cpp_executor.cu — no compilation needed.
                # The Python runtime loads the torch extension lazily only if needed.
                kernels[nid] = None
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
            elif kv == KernelVariant.TileLangFusedGroupedConvBNLIF:
                kern, is_new = self._get_fused_grouped_conv_bn_lif(node)
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
        key = _conv_shape_key(cp, H, W, self.TB) + f"_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_conv_kernels()
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M = self.TB * OH * OW
        K_red = cp.kernel_h * cp.kernel_w * cp.in_channels

        def _compile(cfg):
            return _conv2d_bn_t4(
                TB=self.TB, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                K=cp.kernel_h, S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        _profile_args = (
            torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(cp.kernel_h, cp.kernel_w, cp.in_channels, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
            torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, M, K_red, cp.out_channels,
                                    compile_fn=_compile, profile_args=_profile_args)
        logger.debug("  Conv+BN %d→%d %dx%d s=%d M=%d cfg=%dx%dx%d",
                     cp.in_channels, cp.out_channels, H, W, cp.stride_h, M,
                     cfg['block_M'], cfg['block_N'], cfg['block_K'])

        kern = _compile(cfg)

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Per-timestep fused Conv+BN+IF (T=1 per launch, large batch) ───

    def _get_fused_conv_bn_if(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        is_1x1 = (cp.kernel_h == 1 and cp.kernel_w == 1)

        # For 1x1 conv: use INTERLEAVED kernel (per-CTA T-loop, processes
        # full TB in one call, membrane in registers, no cross-CTA race).
        # For 3x3 conv: use per-timestep kernel (old approach, TB=B per call).
        if is_1x1:
            key = f"interleaved_conv1x1_if_{cp.in_channels}_{cp.out_channels}_{H}x{W}_s{cp.stride_h}_TB{self.TB}_{self.precision}"
        else:
            key = f"fused_conv_if_{cp.in_channels}_{cp.out_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_B{self.B}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1

        if is_1x1:
            # Interleaved kernel: grid covers B*OH*OW spatial (one timestep),
            # each CTA loops over T. Takes full TB input, writes full TB output.
            M_per_t = self.B * OH * OW

            # Detect neuron type from absorbed nodes (IF vs LIF)
            absorbed = node.extra_attrs.get("absorbed_nids", [])
            is_lif = any(self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF
                         for a in absorbed)

            if is_lif:
                from sengine.kernels.interleaved_templates import conv1x1_bn_lif
                lif_node = next((self.ir.nodes[a] for a in absorbed
                                 if self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF), None)
                np_ = lif_node.neuron_params if lif_node else None
                recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5
                def _compile_inter(cfg):
                    return conv1x1_bn_lif(
                        B=self.B, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                        T_steps=self.T, S=cp.stride_h, recip_tau=recip_tau,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
            else:
                from sengine.kernels.interleaved_templates import conv1x1_bn_if
                def _compile_inter(cfg):
                    return conv1x1_bn_if(
                        B=self.B, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                        T_steps=self.T, S=cp.stride_h,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
            _inter_state = torch.zeros(M_per_t, cp.out_channels, dtype=torch.float32, device='cuda')
            _profile_args_inter = (
                torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
                torch.empty(cp.in_channels, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
                _inter_state,
                torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
                torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
            )
            cfg = self._resolve_config(key, M_per_t, cp.in_channels, cp.out_channels,
                                        compile_fn=_compile_inter, profile_args=_profile_args_inter,
                                        interleaved=True, n_membranes=1)
            logger.debug("  Interleaved Conv1x1+BN+IF %d→%d %dx%d B=%d T=%d cfg=%dx%dx%d occ=%d",
                         cp.in_channels, cp.out_channels, H, W, self.B, self.T,
                         cfg['block_M'], cfg['block_N'], cfg['block_K'],
                         estimate_occupancy(cfg['block_M'], cfg['block_N'], cfg['block_K'],
                                            cfg['num_stages'], cfg['threads']))

            kern = _compile_inter(cfg)
        elif cp.in_channels < 4:
            # Stem conv: C_in < 4, pad to 16 for tensor core alignment.
            # Use interleaved stem kernel with per-CTA T-loop.
            from sengine.kernels.conv2d_bn_if_t4 import stem_conv_bn_lif_interleaved_kernel

            C_padded = 16
            M_per_t = self.B * OH * OW
            K_red = cp.kernel_h * cp.kernel_w * C_padded

            # Detect LIF params
            absorbed = node.extra_attrs.get("absorbed_nids", [])
            lif_node = next((self.ir.nodes[a] for a in absorbed
                             if self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF), None)
            np_ = lif_node.neuron_params if lif_node else None
            recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5

            def _compile_stem(cfg):
                return stem_conv_bn_lif_interleaved_kernel(
                    B=self.B, H=H, W=W,
                    C_in_padded=C_padded, C_in_real=cp.in_channels, F=cp.out_channels,
                    KH=cp.kernel_h, KW=cp.kernel_w, S=cp.stride_h, P=cp.pad_h,
                    T_steps=self.T, recip_tau=recip_tau,
                    io_dtype=self.io_dtype_tl,
                    **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
            _state_stem = torch.zeros(M_per_t, cp.out_channels, dtype=torch.float32, device='cuda')
            _profile_args_stem = (
                torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
                torch.empty(cp.kernel_h, cp.kernel_w, C_padded, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
                _state_stem,
                torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
                torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
            )
            cfg = self._resolve_config(key, M_per_t, K_red, cp.out_channels,
                                        compile_fn=_compile_stem, profile_args=_profile_args_stem,
                                        interleaved=True, n_membranes=1)
            logger.debug("  Interleaved Stem Conv+BN+LIF %d→%d %dx%d B=%d T=%d cfg=%dx%dx%d",
                         cp.in_channels, cp.out_channels, H, W, self.B, self.T,
                         cfg['block_M'], cfg['block_N'], cfg['block_K'])

            kern = _compile_stem(cfg)
        else:
            # 3x3: interleaved (per-CTA T-loop with im2col)
            from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_if_interleaved_kernel

            M_per_t = self.B * OH * OW
            K_red = cp.kernel_h * cp.kernel_w * cp.in_channels

            def _compile_3x3(cfg):
                return conv2d_bn_if_interleaved_kernel(
                    B=self.B, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                    K=cp.kernel_h, S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                    T_steps=self.T,
                    io_dtype=self.io_dtype_tl,
                    **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
            _state_3x3 = torch.zeros(M_per_t, cp.out_channels, dtype=torch.float32, device='cuda')
            _profile_args_3x3 = (
                torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
                torch.empty(cp.kernel_h, cp.kernel_w, cp.in_channels, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
                _state_3x3,
                torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
                torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
            )
            cfg = self._resolve_config(key, M_per_t, K_red, cp.out_channels,
                                        compile_fn=_compile_3x3, profile_args=_profile_args_3x3)
            logger.debug("  Interleaved Conv3x3+BN+IF %d→%d %dx%d B=%d T=%d cfg=%dx%dx%d",
                         cp.in_channels, cp.out_channels, H, W, self.B, self.T,
                         cfg['block_M'], cfg['block_N'], cfg['block_K'])

            kern = _compile_3x3(cfg)

        # NOTE: No mid-compile fallback to decomposed. The fusion strategy
        # decides WHAT to fuse; the roofline tuner decides HOW to tile.
        # Un-fusing during compilation corrupts the IR state (schedule,
        # buffer planner, absorbed nodes tracking all assume the fusion
        # decision is final).

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── 1×1 Conv+BN ───

    def _get_conv1x1_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"conv1x1_{cp.in_channels}_{cp.out_channels}_{H}x{W}_s{cp.stride_h}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_conv_kernels()
        OH = (H + cp.stride_h - 1) // cp.stride_h
        OW = (W + cp.stride_w - 1) // cp.stride_w
        M = self.TB * OH * OW

        def _compile(cfg):
            return _conv1x1_bn_t4(
                TB=self.TB, C_in=cp.in_channels, H=H, W=W, F=cp.out_channels,
                S=cp.stride_h,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        _profile_args = (
            torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(cp.in_channels, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
            torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, M, cp.in_channels, cp.out_channels,
                                    compile_fn=_compile, profile_args=_profile_args)

        kern = _compile(cfg)

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Stem Conv+BN (7×7, padded C_in) ───

    def _get_stem_conv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"stem_{cp.in_channels}_{cp.out_channels}_{H}x{W}_TB{self.TB}_{self.precision}"

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
            io_dtype=self.io_dtype_tl,
            **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Linear+BN (transformer) ───

    def _get_linear_bn(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        key = _linear_shape_key(K, N, self.T, self.B) + f"_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_linear_kernels()

        def _compile(cfg):
            return _linear_bn(
                M=M, K=K, N_out=N,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        _profile_args = (
            torch.empty(M, K, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(K, N, dtype=self.io_dtype_torch, device='cuda'),
            torch.ones(N, dtype=torch.float32, device='cuda'),
            torch.zeros(N, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, M, K, N,
                                    compile_fn=_compile, profile_args=_profile_args)

        kern = _compile(cfg)

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

    # ─── Linear+BN+LIF (fused, for MLP chains) ───

    def _get_linear_bn_lif(self, node: Node) -> tuple[object, bool]:
        M, K, N = self._get_linear_dims(node)
        spatial = M // self.T
        key = f"linear_bn_lif_{M}_{K}_{N}_T{self.T}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        _load_linear_kernels()
        cfg = self._resolve_config(key, M, K, N)

        kern = _linear_bn_lif_t4(
            M=M, K=K, N_out=N,
            T_steps=self.T, spatial=spatial,
            io_dtype=self.io_dtype_tl,
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
        key = _matmul_shape_key(M, K, N) + f"_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        _load_linear_kernels()

        def _compile(cfg):
            return _matmul_kernel(
                M=M, K=K, N=N,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        _profile_args = (
            torch.empty(M, K, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(K, N, dtype=self.io_dtype_torch, device='cuda'),
        )
        cfg = self._resolve_config(key, M, K, N,
                                    compile_fn=_compile, profile_args=_profile_args)

        try:
            kern = _compile(cfg)
        except Exception:
            cfg = dict(block_M=64, block_N=64, block_K=32, num_stages=2, threads=128)
            try:
                kern = _compile(cfg)
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
        key = f"matmul_scale_{M}_{K}_{N}_s{scale}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        _load_linear_kernels()

        def _compile_ms(cfg):
            return _matmul_kernel(
                M=M, K=K, N=N, scale=scale,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        _profile_args = (
            torch.empty(M, K, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(K, N, dtype=self.io_dtype_torch, device='cuda'),
        )
        cfg = self._resolve_config(key, M, K, N,
                                    compile_fn=_compile_ms, profile_args=_profile_args)

        try:
            kern = _compile_ms(cfg)
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    # ─── Fused MatMul + LIF (attention attn@V + neuron) ───

    def _get_fused_matmul_lif(self, node: Node) -> tuple[object, bool]:
        """Compile interleaved MatMul+BN+LIF kernel.

        MatMul (TB, M_spatial, K) @ (K, N) is equivalent to Conv1x1
        (TB, H, W, C_in) @ (C_in, C_out) where H*W = M_spatial/B, W=1.
        Reuse the Conv1x1 interleaved template.
        """
        M_full, K, N = self._get_linear_dims(node)
        M_per_t = M_full // self.T if self.T > 0 else M_full
        # M_per_t = B * spatial_per_sample
        spatial_per_sample = M_per_t // self.B if self.B > 0 else M_per_t

        key = f"interleaved_matmul_lif_{K}_{N}_{spatial_per_sample}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache[key]
            return self._kernel_cache[key], False

        if not self._matmul_dims_ok(M_per_t, K, N):
            self._kernel_cache[key] = None
            self._config_cache[key] = {}
            return None, False

        # Detect IF vs LIF from absorbed nodes
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        is_lif = any(self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF
                     for a in absorbed)

        # Use H = spatial_per_sample, W = 1 to map MatMul to Conv1x1 template
        H = spatial_per_sample
        W = 1

        if is_lif:
            from sengine.kernels.interleaved_templates import conv1x1_bn_lif
            lif_node = next((self.ir.nodes[a] for a in absorbed
                             if self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF), None)
            np_ = lif_node.neuron_params if lif_node else None
            recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5
            def _compile_inter(cfg):
                return conv1x1_bn_lif(
                    B=self.B, C_in=K, H=H, W=W, F=N, T_steps=self.T,
                    recip_tau=recip_tau,
                    io_dtype=self.io_dtype_tl,
                    **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})
        else:
            from sengine.kernels.interleaved_templates import conv1x1_bn_if
            def _compile_inter(cfg):
                return conv1x1_bn_if(
                    B=self.B, C_in=K, H=H, W=W, F=N, T_steps=self.T,
                    io_dtype=self.io_dtype_tl,
                    **{k: cfg[k] for k in ('block_M','block_N','block_K','num_stages','threads')})

        _inter_state = torch.zeros(M_per_t, N, dtype=torch.float32, device='cuda')
        _profile_args = (
            torch.empty(self.TB, H, W, K, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(K, N, dtype=self.io_dtype_torch, device='cuda'),
            _inter_state,
            torch.ones(N, dtype=torch.float32, device='cuda'),
            torch.zeros(N, dtype=torch.float32, device='cuda'),
        )

        cfg = self._resolve_config(key, M_per_t, K, N,
                                    compile_fn=_compile_inter, profile_args=_profile_args,
                                    interleaved=True, n_membranes=1)

        kern = _compile_inter(cfg)

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, True

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
        shape0 = node.input_shapes[0] if node.input_shapes else ()
        TB = shape0[0] if shape0 else self.TB
        batch = TB * heads

        if ap.variant == "spikformer":
            # Input: (TB, N, C). N = total_elems / (TB * C)
            total = 1
            for d in shape0:
                total *= d
            N = total // (TB * C) if (TB * C) > 0 else 1
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
            H, W = ap.H, ap.W
            spatial_q = H * W
            # Derive spatial_kv from y_kv (input[0]) which has 2C channels
            s0 = node.input_shapes[0] if node.input_shapes else shape0
            spatial_kv = 1
            for d in s0[2:]:  # skip (TB, 2C), multiply spatial dims
                spatial_kv *= d
            # GEMM1: attn = K^T @ Q: (spatial_kv, hd) @ (hd, spatial_q) = (spatial_kv, spatial_q)
            g1_batch, g1_M, g1_K, g1_N = batch, spatial_kv, hd, spatial_q
            g1_scale = 1.0  # scale1 is a tensor, applied separately
            # GEMM2 (restructured): out = attn^T @ V: (spatial_q, spatial_kv) @ (spatial_kv, hd) = (spatial_q, hd)
            g2_batch, g2_M, g2_K, g2_N = batch, spatial_q, spatial_kv, hd
            g2_scale = 1.0  # scale2 is a tensor, applied separately
        elif ap.variant == "token_qk":
            # TokenQK: no matmul, only sum+mul. No TileLang GEMM needed.
            return None
        else:
            return None

        if ap.variant == "maxformer":
            from sengine.kernels.fused_attention_kernels import (
                maxformer_kTv_kernel, maxformer_qkv_kernel)

            C = heads * hd
            key1 = f"maxformer_kTv_{TB}_{heads}_{hd}_{N}_{H}_{W}_{self.precision}"
            key2 = f"maxformer_qkv_lif_{TB}_{heads}_{hd}_{N}_{H}_{W}_s{ap.scale}_{self.precision}"

            # Clamp tile sizes: T.copy in these kernels accesses hd×hd
            # output/intermediate tensors. block_M/block_N must not exceed hd
            # to avoid out-of-bounds T.copy (which causes TileLang divide-by-zero).
            def _clamp_attn_cfg(cfg, max_M, max_N, max_K):
                c = dict(cfg)
                if c['block_M'] > max_M:
                    c['block_M'] = max_M
                if c['block_N'] > max_N:
                    c['block_N'] = max_N
                if c['block_K'] > max_K:
                    c['block_K'] = max_K
                return c

            gemm1 = self._kernel_cache.get(key1)
            if gemm1 is None:
                def _compile_mf_g1(cfg):
                    c = _clamp_attn_cfg(cfg, hd, hd, N)
                    return maxformer_kTv_kernel(
                        TB=TB, heads=heads, hd=hd, N=N, H=H, W=W,
                        io_dtype=self.io_dtype_tl,
                        **{k: c[k] for k in ('block_M', 'block_N', 'block_K',
                                                'num_stages', 'threads')})
                _profile_mf_g1 = (
                    torch.empty(TB * N, C, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(TB * N, C, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg1 = self._resolve_config(key1, hd, N, hd,
                                            compile_fn=_compile_mf_g1,
                                            profile_args=_profile_mf_g1)
                cfg1 = _clamp_attn_cfg(cfg1, hd, hd, N)
                try:
                    gemm1 = _compile_mf_g1(cfg1)
                except Exception:
                    gemm1 = None
                self._kernel_cache[key1] = gemm1
                self._config_cache[key1] = cfg1

            gemm2 = self._kernel_cache.get(key2)
            if gemm2 is None:
                def _compile_mf_g2(cfg):
                    c = _clamp_attn_cfg(cfg, N, hd, hd)
                    return maxformer_qkv_kernel(
                        TB=TB, heads=heads, hd=hd, N=N, H=H, W=W,
                        scale=ap.scale,
                        io_dtype=self.io_dtype_tl,
                        **{k: c[k] for k in ('block_M', 'block_N', 'block_K',
                                                'num_stages', 'threads')})
                _profile_mf_g2 = (
                    torch.empty(TB * N, C, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(batch * hd, hd, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg2 = self._resolve_config(key2, N, hd, hd,
                                            compile_fn=_compile_mf_g2,
                                            profile_args=_profile_mf_g2)
                cfg2 = _clamp_attn_cfg(cfg2, N, hd, hd)
                try:
                    gemm2 = _compile_mf_g2(cfg2)
                except Exception:
                    gemm2 = None
                self._kernel_cache[key2] = gemm2
                self._config_cache[key2] = cfg2

        elif ap.variant == "spikformer":
            from sengine.kernels.spikformer_kernels import (
                batched_matmul_kernel, batched_matmul_bt_kernel)

            key1 = f"batched_mm_bt_{batch}_{g1_M}_{g1_K}_{g1_N}_s{g1_scale}_{self.precision}"
            gemm1 = self._kernel_cache.get(key1)
            if gemm1 is None:
                def _compile_sf_g1(cfg):
                    return batched_matmul_bt_kernel(
                        batch=batch, M=g1_M, K=g1_K, N=g1_N, scale=g1_scale,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                                'num_stages', 'threads')})
                _profile_sf_g1 = (
                    torch.empty(batch * g1_M, g1_K, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(batch * g1_N, g1_K, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg1 = self._resolve_config(key1, g1_M, g1_K, g1_N,
                                            compile_fn=_compile_sf_g1,
                                            profile_args=_profile_sf_g1)
                try:
                    gemm1 = _compile_sf_g1(cfg1)
                except Exception:
                    gemm1 = None
                self._kernel_cache[key1] = gemm1
                self._config_cache[key1] = cfg1

            key2 = f"batched_mm_{batch}_{g2_M}_{g2_K}_{g2_N}_s{g2_scale}_{self.precision}"
            gemm2 = self._kernel_cache.get(key2)
            if gemm2 is None:
                def _compile_sf_g2(cfg):
                    return batched_matmul_kernel(
                        batch=batch, M=g2_M, K=g2_K, N=g2_N, scale=g2_scale,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                                'num_stages', 'threads')})
                _profile_sf_g2 = (
                    torch.empty(batch * g2_M, g2_K, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(batch * g2_K, g2_N, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg2 = self._resolve_config(key2, g2_M, g2_K, g2_N,
                                            compile_fn=_compile_sf_g2,
                                            profile_args=_profile_sf_g2)
                try:
                    gemm2 = _compile_sf_g2(cfg2)
                except Exception:
                    gemm2 = None
                self._kernel_cache[key2] = gemm2
                self._config_cache[key2] = cfg2
        elif ap.variant == "dssa":
            from sengine.kernels.fused_attention_kernels import (
                dssa_kTq_kernel, dssa_v_attn_kernel)

            C_full = heads * hd
            key1 = f"dssa_kTq_{TB}_{heads}_{hd}_{spatial_kv}_{spatial_q}_{self.precision}"
            key2 = f"dssa_v_attn_{TB}_{heads}_{hd}_{spatial_kv}_{spatial_q}_{H}x{W}_{self.precision}"

            gemm1 = self._kernel_cache.get(key1)
            if gemm1 is None:
                def _compile_g1(cfg):
                    return dssa_kTq_kernel(
                        TB=TB, heads=heads, hd=hd,
                        spatial_kv=spatial_kv, spatial_q=spatial_q,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                _profile_g1 = (
                    torch.empty(TB * spatial_kv, 2 * C_full, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(TB * spatial_q, C_full, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg1 = self._resolve_config(key1, spatial_kv, hd, spatial_q,
                                            compile_fn=_compile_g1, profile_args=_profile_g1)
                try:
                    gemm1 = _compile_g1(cfg1)
                except Exception:
                    gemm1 = None
                self._kernel_cache[key1] = gemm1
                self._config_cache[key1] = cfg1

            gemm2 = self._kernel_cache.get(key2)
            if gemm2 is None:
                def _compile_g2(cfg):
                    return dssa_v_attn_kernel(
                        TB=TB, heads=heads, hd=hd,
                        spatial_kv=spatial_kv, spatial_q=spatial_q,
                        H_out=H, W_out=W,
                        io_dtype=self.io_dtype_tl,
                        **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                                 'num_stages', 'threads')})
                _profile_g2 = (
                    torch.empty(TB * spatial_kv, 2 * C_full, dtype=self.io_dtype_torch, device='cuda'),
                    torch.empty(batch * spatial_kv, spatial_q, dtype=self.io_dtype_torch, device='cuda'),
                )
                cfg2 = self._resolve_config(key2, spatial_q, spatial_kv, hd,
                                            compile_fn=_compile_g2, profile_args=_profile_g2)
                try:
                    gemm2 = _compile_g2(cfg2)
                except Exception:
                    gemm2 = None
                self._kernel_cache[key2] = gemm2
                self._config_cache[key2] = cfg2

        else:
            # TokenQK: no matmul, handled by Python runtime
            gemm1 = gemm2 = None

        if gemm1 is None or gemm2 is None:
            return None
        return (gemm1, gemm2)

    # ─── Depthwise Conv+BN ───

    def _get_dwconv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"dwconv_{cp.in_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.dwconv_bn import dwconv_bn_kernel
        logger.debug("  DWConv+BN %d %dx%d s=%d",
                     cp.in_channels, H, W, cp.stride_h)

        # FP32: cap block_HW so vectorization stays within float32x8
        block_HW = 64 if self.precision == 'fp32' else 128
        kern = dwconv_bn_kernel(
            TB=self.TB, C=cp.in_channels, H=H, W=W,
            K=cp.kernel_h, S=cp.stride_h, P=cp.pad_h,
            block_HW=block_HW,
            io_dtype=self.io_dtype_tl)

        self._kernel_cache[key] = kern
        return kern, True

    def _get_dwconv_bn_if(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"dwconv_if_{cp.in_channels}_{cp.kernel_h}x{cp.kernel_w}_{H}x{W}_s{cp.stride_h}_B{self.B}_{self.precision}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.dwconv_bn import dwconv_bn_if_kernel
        logger.debug("  DWConv+BN+IF %d %dx%d s=%d",
                     cp.in_channels, H, W, cp.stride_h)

        block_HW = 64 if self.precision == 'fp32' else 128
        kern = dwconv_bn_if_kernel(
            TB=self.B, C=cp.in_channels, H=H, W=W,
            K=cp.kernel_h, S=cp.stride_h, P=cp.pad_h,
            block_HW=block_HW,
            T_steps=1,
            io_dtype=self.io_dtype_tl)

        self._kernel_cache[key] = kern
        return kern, True

    # ─── Grouped Conv+BN ───

    # ─── Fused Add+LIF ───

    def _get_fused_add_lif(self, node: Node) -> tuple[object, bool]:
        """Compile fused Add+LIF kernel (per-CTA T-loop, 5-arg interface)."""
        if not node.output_shapes or len(node.output_shapes[0]) != 4:
            return None, False

        TB_out, C_out, OH, OW = node.output_shapes[0]
        F = C_out
        B = TB_out // self.T
        spatial = B * OH * OW
        key = f"add_lif_{F}_{OH}x{OW}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.add_lif_fused import add_lif_fused_kernel
        logger.debug("  Add+LIF fused %d %dx%d TB=%d", F, OH, OW, self.TB)

        def _compile(cfg):
            return add_lif_fused_kernel(
                TB=self.TB, OH=OH, OW=OW, F=F, T_steps=self.T,
                block_M=cfg['block_M'], block_N=cfg['block_N'],
                threads=cfg['threads'],
                io_dtype=self.io_dtype_tl)
        _state = torch.zeros(spatial, F, dtype=torch.float32, device='cuda')
        _profile_args = (
            torch.empty(self.TB, OH, OW, F, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(self.TB, OH, OW, F, dtype=self.io_dtype_torch, device='cuda'),
            _state,
            torch.zeros(F, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, spatial, 1, F,
                                    compile_fn=_compile, profile_args=_profile_args)
        kern = _compile(cfg)
        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        return kern, True

    def _get_fused_pool_lif(self, node: Node) -> tuple[object, bool]:
        """Compile fused MaxPool+LIF kernel (per-CTA T-loop, 5-arg interface)."""
        if not node.output_shapes or len(node.output_shapes[0]) != 4:
            return None, False
        if not node.pool_params:
            return None, False

        TB_out, C_out, OH, OW = node.output_shapes[0]
        pp = node.pool_params  # dict with kernel_shape, strides, pads
        pk = pp.get('kernel_shape', [2, 2])[0]
        ps = pp.get('strides', [2, 2])[0]
        pad = pp.get('pads', [0, 0, 0, 0])[0]
        H_in = (OH - 1) * ps + pk - 2 * pad
        W_in = (OW - 1) * ps + pk - 2 * pad
        key = f"pool_lif_{C_out}_{H_in}x{W_in}_k{pk}_s{ps}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            return self._kernel_cache[key], False

        from sengine.kernels.pool_lif_fused import maxpool_lif_fused_kernel
        logger.debug("  Pool+LIF fused %d %dx%d->%dx%d TB=%d",
                     C_out, H_in, W_in, OH, OW, self.TB)

        kern = maxpool_lif_fused_kernel(
            TB=self.TB, C=C_out, H=H_in, W=W_in,
            pool_k=pk, pool_s=ps, pool_p=pad,
            T_steps=self.T, block_M=64, block_N=64, threads=128,
            io_dtype=self.io_dtype_tl)

        self._kernel_cache[key] = kern
        return kern, True

    # ─── Grouped Conv+BN ───

    def _get_grouped_conv_bn(self, node: Node) -> tuple[object, bool]:
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"gconv_{cp.in_channels}_{cp.out_channels}_{cp.kernel_h}x{cp.kernel_w}" \
              f"_s{cp.stride_h}_g{cp.groups}_{H}x{W}_TB{self.TB}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache.get(key, {})
            return self._kernel_cache[key], False

        C_in_per_g = cp.in_channels // cp.groups
        K_red = cp.kernel_h * cp.kernel_w * C_in_per_g
        C_out_per_g = cp.out_channels // cp.groups
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M = self.TB * OH * OW

        from sengine.kernels.grouped_conv_bn import grouped_conv_bn_kernel

        def _compile(cfg):
            return grouped_conv_bn_kernel(
                TB=self.TB, C_in=cp.in_channels, H=H, W=W,
                C_out=cp.out_channels, K=cp.kernel_h,
                S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                groups=cp.groups,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                        'num_stages', 'threads')})
        _profile_args = (
            torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(cp.kernel_h, cp.kernel_w, C_in_per_g, cp.out_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
            torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, M, K_red, C_out_per_g,
                                    compile_fn=_compile, profile_args=_profile_args)

        try:
            kern = _compile(cfg)
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    def _get_fused_grouped_conv_bn_lif(self, node: Node) -> tuple[object, bool]:
        """Compile interleaved Grouped Conv+BN+LIF kernel."""
        cp = node.conv_params
        H, W = self._get_spatial(node)
        key = f"fused_gconv_lif_{cp.in_channels}_{cp.out_channels}_{cp.kernel_h}x{cp.kernel_w}" \
              f"_s{cp.stride_h}_g{cp.groups}_{H}x{W}_B{self.B}_{self.precision}"

        if key in self._kernel_cache:
            node.tilelang_config = self._config_cache.get(key, {})
            return self._kernel_cache[key], False

        C_in_per_g = cp.in_channels // cp.groups
        C_out_per_g = cp.out_channels // cp.groups
        K_red = cp.kernel_h * cp.kernel_w * C_in_per_g
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        M_per_t = self.B * OH * OW

        # Detect neuron params from absorbed nodes
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        lif_node = next((self.ir.nodes[a] for a in absorbed
                         if self.ir.nodes.get(a) and self.ir.nodes[a].op_type == OpType.LIF), None)
        np_ = lif_node.neuron_params if lif_node else None
        recip_tau = 1.0 / np_.tau if (np_ and np_.tau and np_.tau > 0) else 0.5

        from sengine.kernels.grouped_conv_bn import grouped_conv_bn_lif_kernel

        def _compile(cfg):
            return grouped_conv_bn_lif_kernel(
                B=self.B, C_in=cp.in_channels, H=H, W=W,
                C_out=cp.out_channels, K=cp.kernel_h,
                S=cp.stride_h, D=cp.dilation_h, P=cp.pad_h,
                groups=cp.groups, T_steps=self.T,
                recip_tau=recip_tau,
                io_dtype=self.io_dtype_tl,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K',
                                        'num_stages', 'threads')})

        _state = torch.zeros(M_per_t, cp.out_channels, dtype=torch.float32, device='cuda')
        _profile_args = (
            torch.empty(self.TB, H, W, cp.in_channels, dtype=self.io_dtype_torch, device='cuda'),
            torch.empty(cp.kernel_h, cp.kernel_w, C_in_per_g, cp.out_channels,
                        dtype=self.io_dtype_torch, device='cuda'),
            _state,
            torch.ones(cp.out_channels, dtype=torch.float32, device='cuda'),
            torch.zeros(cp.out_channels, dtype=torch.float32, device='cuda'),
        )
        cfg = self._resolve_config(key, M_per_t, K_red, C_out_per_g,
                                    compile_fn=_compile, profile_args=_profile_args,
                                    interleaved=True, n_membranes=1)

        try:
            kern = _compile(cfg)
        except Exception:
            kern = None

        self._kernel_cache[key] = kern
        self._config_cache[key] = cfg
        node.tilelang_config = cfg
        node.est_latency_us = cfg.get('latency_us', 0.0)
        return kern, kern is not None

    # ─── Helpers ───

    def _get_spatial(self, node: Node) -> tuple[int, int]:
        """Extract (H, W) from a Conv node's input shape."""
        if node.input_shapes:
            s = node.input_shapes[0]
            if len(s) == 4:
                return s[2], s[3]  # NCHW: (N, C, H, W)
            if len(s) == 5:
                return s[3], s[4]  # (T, B, C, H, W) or (B, 1, C, H, W)
        # Fallback: try output shapes
        if node.output_shapes:
            s = node.output_shapes[0]
            cp = node.conv_params
            if cp and len(s) == 4:
                OH, OW = s[2], s[3]
                H = (OH - 1) * cp.stride_h - 2 * cp.pad_h + cp.kernel_h
                W = (OW - 1) * cp.stride_w - 2 * cp.pad_w + cp.kernel_w
                return H, W
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
                        compile_fn=None, profile_args=None,
                        interleaved: bool = False,
                        n_membranes: int = 1,
                        bpe: int = 0) -> dict:
        """Get tile config from cache, autotune, or heuristic.

        Priority: cache hit → autotune (if enabled) → heuristic default.

        Args:
            interleaved: If True, use occupancy-aware tuning for interleaved
                         kernels (targets ≥2 CTAs/SM for compute↔memory overlap).
            n_membranes: Number of neuron membranes in the epilogue chain
                         (1 for Conv+IF, 2 for Conv+IF+Add+LIF).
        """
        hw = _get_hw_info()
        gpu_name = hw['gpu_name']
        gpu_arch = hw['gpu_arch']

        # Check tuning cache first
        if self.tuning_cache:
            cached = self.tuning_cache.get(key, gpu_name, gpu_arch, self.T, self.B)
            if cached is not None:
                return cached

        # Autotune: roofline-guided pruning + GPU profiling
        # Works for both decomposed and interleaved kernels.
        if self.autotune and compile_fn is not None and profile_args is not None:
            from sengine.tuning.roofline import select_config_roofline
            T_steps = self.T if interleaved else 1
            logger.debug("  Roofline tuning %s (M=%d, K=%d, F=%d, T=%d)...",
                         key, M, K_red, F, T_steps)
            _bpe = bpe if bpe > 0 else (4 if self.precision == 'fp32' else 2)
            cfg = select_config_roofline(
                M, K_red, F, T_steps,
                compile_fn=compile_fn,
                profile_args=profile_args,
                n_membranes=n_membranes,
                top_k=5, n_profile=100,
                bpe=_bpe)
            if self.tuning_cache:
                self.tuning_cache.put(key, gpu_name, cfg, gpu_arch, self.T, self.B)
            return cfg

        # No autotuning: use heuristic default
        _bpe = bpe if bpe > 0 else (4 if self.precision == 'fp32' else 2)
        if interleaved:
            return _pick_interleaved_config(M, K_red, F,
                                             sm_count=hw['sm_count'],
                                             n_membranes=n_membranes)
        return _pick_config(M, K_red, F, bpe=_bpe)
