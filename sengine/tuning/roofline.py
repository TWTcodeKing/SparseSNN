"""Roofline-Guided Kernel Tuning Strategy.

Classifies each GEMM shape as compute-bound or memory-bound using the
roofline model, prunes the config space to top-K candidates, then profiles
only those on GPU.

Hardware-adaptive: auto-detects GPU specs via CUDA runtime. Supports all
NVIDIA GPUs (Ampere, Ada Lovelace, Hopper, Blackwell).

Typical tuning time: 2-5 seconds per shape (5-7 profiles × 0.3s each).
"""

from __future__ import annotations
import math
import time
import torch

from sengine.logger import logger


# ─── Hardware detection (calibrated via microbenchmarks) ───

_GPU_SPECS = None


def _detect_gpu():
    """Auto-detect GPU specs via CUDA API + microbenchmarks.

    First call runs calibration (~2s). Results are cached per GPU UUID.
    Works on any NVIDIA GPU (desktop, server, Jetson).
    """
    global _GPU_SPECS
    if _GPU_SPECS is not None:
        return _GPU_SPECS

    from sengine.tuning.hw_calibrate import calibrate
    _GPU_SPECS = calibrate()
    return _GPU_SPECS


# ─── Shared memory and occupancy ───

def _smem_bytes(bM, bN, bK, ns, bpe=2):
    """Shared memory for pipelined GEMM + output staging.

    Args:
        bpe: bytes per element (2 for fp16, 4 for fp32).
    """
    return (bM * bK + bK * bN) * bpe * ns + bM * bN * bpe


def _occupancy(bM, bN, bK, ns, thr, n_membranes=1, bpe=2):
    """Estimate max CTAs per SM."""
    hw = _detect_gpu()
    smem = _smem_bytes(bM, bN, bK, ns, bpe)
    elems = (bM * bN) // thr
    regs = elems * (1 + n_membranes) + 10
    by_smem = hw['max_smem_per_sm'] // max(smem, 1)
    by_regs = hw['max_regs_per_sm'] // max(regs * thr, 1)
    by_thr = hw['max_threads_per_sm'] // max(thr, 1)
    return min(by_smem, by_regs, by_thr, hw['max_blocks_per_sm'])


# ─── Roofline classification ───

def classify_shape(M_per_t, K, N, T_steps, bpe=2):
    """Classify an interleaved GEMM as compute-bound or memory-bound.

    Args:
        bpe: bytes per element for data/weight I/O (2 for fp16, 4 for fp32).
             Membrane is always fp32 (4 bytes).
    """
    hw = _detect_gpu()
    flops = 2.0 * M_per_t * K * N * T_steps
    data_bytes = M_per_t * K * bpe * T_steps
    weight_bytes = K * N * bpe
    output_bytes = M_per_t * N * bpe * T_steps
    membrane_bytes = M_per_t * N * 4 * 2  # always fp32
    total_bytes = data_bytes + weight_bytes + output_bytes + membrane_bytes

    ai = flops / total_bytes
    bound = 'compute' if ai > hw['ridge_point'] else 'memory'
    return bound, ai, {'flops': flops, 'bytes': total_bytes, 'ai': ai,
                       'ridge': hw['ridge_point']}


# ─── Candidate pruning ───

def prune_candidates(M_per_t, K, N, T_steps, n_membranes=1, top_k=5, bpe=2):
    """Generate top-K tile config candidates using roofline classification.

    Args:
        bpe: bytes per element (2 for fp16, 4 for fp32).
    """
    hw = _detect_gpu()
    bound, ai, _ = classify_shape(M_per_t, K, N, T_steps, bpe)
    sm_count = hw['sm_count']

    all_cfgs = []
    # FP32 can't vectorize as wide as FP16 — TileLang's T.copy() generates
    # float32xN vector types that CUDA doesn't support when N>8.
    # Constraint: bM * bK / threads <= 8 for fp32 (and bN * bK / threads too).
    bK_choices = [32, 64] if bpe == 2 else [32]
    for bM in [16, 32, 64, 128]:
        for bN in [32, 64, 128]:
            for bK in bK_choices:
                if bK > K:
                    continue
                for ns in [2, 3]:
                    thr = 128
                    # FP32: skip configs where T.copy vectorization exceeds float32x8
                    if bpe == 4 and (bM * bK // thr > 8 or bN * bK // thr > 8):
                        continue
                    smem = _smem_bytes(bM, bN, bK, ns, bpe)
                    if smem > hw['max_smem_per_sm']:
                        continue
                    grid = math.ceil(M_per_t / bM) * math.ceil(N / bN)
                    if grid < sm_count // 8:
                        continue
                    occ = _occupancy(bM, bN, bK, ns, thr, n_membranes, bpe)
                    tile_area = bM * bN
                    cfg = dict(block_M=bM, block_N=bN, block_K=bK,
                               num_stages=ns, threads=thr)
                    all_cfgs.append((cfg, grid, occ, tile_area))

    if not all_cfgs:
        return [dict(block_M=32, block_N=64, block_K=min(32, K),
                     num_stages=2, threads=128)]

    # Unified ranking: balance tile area (TC efficiency) with grid size
    # (SM parallelism). A tile is "good" if area >= 2048 AND grid >= SM/2.
    # Penalize configs that fail either condition.
    sm_half = hw['sm_count'] // 2
    def _rank(cfg_tuple):
        cfg, grid, occ, area = cfg_tuple
        # Prefer: large area (up to 4096), sufficient grid (>= SM/2)
        area_score = min(area, 4096)
        grid_ok = 1 if grid >= sm_half else 0.5
        return -(area_score * grid_ok)

    all_cfgs.sort(key=_rank)

    # Deduplicate by (bM, bN)
    seen = set()
    pruned = []
    for cfg, grid, occ, tile_area in all_cfgs:
        key = (cfg['block_M'], cfg['block_N'])
        if key not in seen:
            seen.add(key)
            pruned.append(cfg)
        if len(pruned) >= top_k:
            break

    # Always include known-good configs
    known_good = [
        dict(block_M=64, block_N=64, block_K=32, num_stages=2, threads=128),
        dict(block_M=32, block_N=64, block_K=32, num_stages=2, threads=128),
    ]
    for kg in known_good:
        if kg['block_K'] > K:
            continue
        key = (kg['block_M'], kg['block_N'])
        if key not in seen:
            seen.add(key)
            pruned.append(kg)

    return pruned


# ─── Main tuning function ───

def select_config_roofline(M_per_t, K, N, T_steps,
                            compile_fn, profile_args,
                            n_membranes=1, top_k=5,
                            n_profile=100,
                            compile_timeout=10.0,
                            bpe=2):
    """Select best config using roofline pruning + GPU profiling.

    Args:
        M_per_t: Spatial dimension per timestep (B × OH × OW).
        K: Reduction dimension (C_in for 1x1, K*K*C_in for KxK).
        N: Output channels (C_out).
        T_steps: Number of timesteps.
        compile_fn: callable(cfg) → kernel.
        profile_args: tuple of tensors for profiling.
        n_membranes: Number of neuron membranes (1 or 2).
        top_k: Max candidates to profile.
        n_profile: Profiling iterations per candidate.
        compile_timeout: Skip configs that take longer to compile (seconds).
        bpe: bytes per element (2 for fp16, 4 for fp32).

    Returns:
        dict with block_M, block_N, block_K, num_stages, threads, latency_us.
    """
    candidates = prune_candidates(M_per_t, K, N, T_steps,
                                   n_membranes=n_membranes, top_k=top_k,
                                   bpe=bpe)

    best_us = float('inf')
    best_cfg = candidates[0] if candidates else dict(
        block_M=32, block_N=64, block_K=min(32, K), num_stages=2, threads=128)

    _dev = torch.cuda.current_device()

    for cfg in candidates:
        try:
            torch.cuda.set_device(_dev)  # restore before compilation
            t0 = time.time()
            kern = compile_fn(cfg)
            compile_s = time.time() - t0
            if compile_s > compile_timeout:
                continue

            torch.cuda.set_device(_dev)  # restore after compilation
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

            if us < best_us:
                best_us = us
                best_cfg = cfg
        except Exception:
            continue

    best_cfg['latency_us'] = best_us
    return best_cfg
