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

from sengine_edge.logger import logger


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

    from sengine_edge.tuning.hw_calibrate import calibrate
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
    """Generate top-K tile config candidates for Jetson AGX Orin (sm_87, 16 SMs).

    Orin is Ampere (supports cp.async pipeline stages 2-4) but has only 16 SMs
    and 100KB smem/SM. Grid coverage and occupancy are critical — a single wave
    miss wastes 6.25% of GPU (1/16) vs 0.78% on 4090 (1/128).

    Args:
        bpe: bytes per element (2 for fp16, 4 for fp32).
    """
    hw = _detect_gpu()
    bound, ai, _ = classify_shape(M_per_t, K, N, T_steps, bpe)
    sm_count = hw['sm_count']
    max_smem = hw['max_smem_per_sm']

    # Orin: Ampere ISA (cp.async), moderate tiles, 2-4 pipeline stages
    bM_choices = [16, 32, 64, 128]
    bN_choices = [32, 64, 128]
    bK_base = [32, 64] if bpe == 2 else [32]
    bK_divisors = set()
    for bk in [32, 48, 64, 96]:
        if bpe == 4 and bk > 32:
            continue
        if K > 0 and K % bk == 0 and bk <= K:
            bK_divisors.add(bk)
    bK_choices = sorted(set(bK_base) | bK_divisors)
    ns_choices = [2, 3, 4]  # Ampere cp.async supports 2-4 stages
    thr_choices = [128, 256]

    all_cfgs = []
    for bM in bM_choices:
        for bN in bN_choices:
            for bK in bK_choices:
                if bK > K:
                    continue
                for ns in ns_choices:
                    for thr in thr_choices:
                        if bpe == 4 and (bM * bK // thr > 8 or bN * bK // thr > 8):
                            continue
                        if bM * bN < thr:
                            continue
                        smem = _smem_bytes(bM, bN, bK, ns, bpe)
                        if smem > max_smem:
                            continue
                        grid = math.ceil(M_per_t / bM) * math.ceil(N / bN)
                        occ = _occupancy(bM, bN, bK, ns, thr, n_membranes, bpe)
                        # On 16 SMs, grid must fill at least 2 SMs to be worth it
                        max_possible_grid = math.ceil(M_per_t / bM_choices[0]) * math.ceil(N / bN_choices[0])
                        min_grid = min(max(sm_count // 4, 1), max_possible_grid)
                        if grid < min_grid:
                            continue
                        tile_area = bM * bN
                        cfg = dict(block_M=bM, block_N=bN, block_K=bK,
                                   num_stages=ns, threads=thr)
                        all_cfgs.append((cfg, grid, occ, tile_area))

    if not all_cfgs:
        return [dict(block_M=32, block_N=64, block_K=max(16, min(32, K)),
                     num_stages=2, threads=128)]

    # Ranking for Orin: wave efficiency, grid coverage, and occupancy dominate.
    # With 16 SMs, a grid of 17 CTAs wastes 15/16 SMs in the second wave.
    def _rank(cfg_tuple):
        cfg, grid, occ, area = cfg_tuple
        # Tile area: moderate cap — Orin can't saturate with huge tiles
        area_score = min(area, 4096)
        # Wave efficiency: fraction of SMs utilized across all waves.
        # grid=16 → 1.0, grid=17 → 17/32=0.53, grid=32 → 1.0, grid=33 → 0.52
        active_slots = sm_count * occ if occ >= 1 else sm_count
        n_waves = math.ceil(grid / active_slots)
        wave_eff = grid / (n_waves * active_slots) if n_waves > 0 else 0.5
        # Grid coverage: critical on 16 SMs — penalize hard if < sm_count
        if grid >= sm_count * 2:
            grid_ok = 1.0
        elif grid >= sm_count:
            grid_ok = 0.85
        elif grid >= sm_count // 2:
            grid_ok = 0.5
        else:
            grid_ok = 0.2
        # K-loop depth penalty
        k_iters = math.ceil(K / cfg['block_K'])
        k_eff = min(1.0, 128.0 / k_iters) if k_iters > 128 else 1.0
        # Occupancy: very important on Orin to hide LPDDR5 latency
        occ_score = 1.0 + 0.3 * min(occ, 4)
        # Pipeline depth bonus (Ampere cp.async)
        ns_bonus = 1.0 + 0.08 * (cfg['num_stages'] - 2)
        # Wave efficiency: grid=16 perfect, grid=17 wastes 47% of second wave
        return -(area_score * grid_ok * wave_eff * k_eff * occ_score * ns_bonus)

    all_cfgs.sort(key=_rank)

    seen = set()
    pruned = []
    for cfg, grid, occ, tile_area in all_cfgs:
        key = (cfg['block_M'], cfg['block_N'], cfg['block_K'],
               cfg['num_stages'], cfg['threads'])
        if key not in seen:
            seen.add(key)
            pruned.append(cfg)
        if len(pruned) >= top_k:
            break

    # Known-good configs for Orin (16 SMs, 100KB smem, Ampere cp.async)
    # Research: fewer SMs → larger tiles per CTA to amortize overhead.
    # Include 128x64/64x128 alongside moderate tiles for wave-aligned grids.
    known_good = [
        # Large tiles: fewer CTAs but each CTA does more work (good for big GEMMs)
        dict(block_M=128, block_N=64, block_K=32, num_stages=3, threads=256),
        dict(block_M=64, block_N=128, block_K=32, num_stages=3, threads=256),
        dict(block_M=128, block_N=64, block_K=64, num_stages=2, threads=256),
        dict(block_M=64, block_N=128, block_K=64, num_stages=2, threads=256),
        # Moderate tiles: good SM fill for medium GEMMs
        dict(block_M=64, block_N=64, block_K=32, num_stages=3, threads=128),
        dict(block_M=32, block_N=64, block_K=32, num_stages=3, threads=128),
        dict(block_M=64, block_N=64, block_K=64, num_stages=2, threads=128),
        # Deeper pipeline for memory-bound shapes (LPDDR5 high latency)
        dict(block_M=64, block_N=64, block_K=32, num_stages=4, threads=128),
        dict(block_M=32, block_N=64, block_K=64, num_stages=4, threads=128),
        # Asymmetric tiles for narrow GEMMs
        dict(block_M=32, block_N=128, block_K=32, num_stages=2, threads=128),
        dict(block_M=128, block_N=32, block_K=32, num_stages=2, threads=128),
        # K-divisibility for common SNN channel counts (64, 128, 256, 512)
        dict(block_M=64, block_N=64, block_K=64, num_stages=3, threads=128),
        dict(block_M=32, block_N=64, block_K=48, num_stages=3, threads=128),
    ]
    for kg in known_good:
        if kg['block_K'] > K:
            continue
        if kg['block_M'] * kg['block_N'] < kg['threads']:
            continue
        smem = _smem_bytes(kg['block_M'], kg['block_N'], kg['block_K'],
                           kg['num_stages'], bpe)
        if smem > max_smem:
            continue
        key = (kg['block_M'], kg['block_N'], kg['block_K'],
               kg['num_stages'], kg['threads'])
        if key not in seen:
            seen.add(key)
            pruned.append(kg)

    return pruned


# ─── Main tuning function ───

def select_config_roofline(M_per_t, K, N, T_steps,
                            compile_fn, profile_args,
                            n_membranes=1, top_k=7,
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
