"""Approach 1: Analytical Cost Model for Interleaved Kernel Config Selection.

tritonBLAS-inspired: models kernel latency from hardware constants and tile
parameters WITHOUT profiling on GPU. Selects the config that minimizes the
predicted cost in microseconds.

Cost model for interleaved Conv+BN+IF:
    T_total = T_steps × (T_gemm + T_epilogue) + T_mem_init + T_mem_writeback

    T_gemm = max(T_compute, T_memory)    # roofline
    T_compute = 2·M·K·N / (SM_active × TC_throughput_per_SM)
    T_memory  = (data_bytes + weight_bytes) / effective_BW

    T_epilogue = M·N·(BN_ops + IF_ops) / (SM_active × ALU_throughput)
                 + M·N·elem_size / mem_BW   (output write)

Hardware constants are auto-detected from the current GPU via hw_calibrate.
"""

from __future__ import annotations
import math


# ─── Hardware constants (auto-detected, cached) ───

_HW = None


def _get_hw() -> dict:
    """Return hardware specs, auto-detecting on first call."""
    global _HW
    if _HW is not None:
        return _HW

    try:
        from sengine.tuning.hw_calibrate import calibrate
        cal = calibrate()
        _HW = {
            'sm_count': cal['sm_count'],
            'tc_tflops': cal['tc_tflops'],
            'mem_bw_gbps': cal['mem_bw_gbps'],
            'l2_bw_gbps': cal['mem_bw_gbps'] * 4,  # ~4× DRAM as L2 estimate
            'l2_size_bytes': cal['l2_size_bytes'],
            'max_smem_per_sm': cal['max_smem_per_sm'],
            'max_regs_per_sm': cal['max_regs_per_sm'],
            'max_threads_per_sm': cal['max_threads_per_sm'],
            'max_blocks_per_sm': cal['max_blocks_per_sm'],
        }
    except Exception:
        # Fallback: conservative defaults
        _HW = {
            'sm_count': 128,
            'tc_tflops': 82.6,
            'mem_bw_gbps': 1008,
            'l2_bw_gbps': 4000,
            'l2_size_bytes': 72 * 1024 * 1024,
            'max_smem_per_sm': 100 * 1024,
            'max_regs_per_sm': 65536,
            'max_threads_per_sm': 1536,
            'max_blocks_per_sm': 16,
        }
    return _HW


def _smem_bytes(bM, bN, bK, ns):
    """Shared memory for pipelined GEMM + output staging."""
    return (bM * bK + bK * bN) * 2 * ns + bM * bN * 2


def _occupancy(bM, bN, bK, ns, thr, n_membranes=1):
    """Estimate max CTAs per SM."""
    hw = _get_hw()
    smem = _smem_bytes(bM, bN, bK, ns)
    elems_per_thread = (bM * bN) // thr
    regs = elems_per_thread * (1 + n_membranes) + 10

    by_smem = hw['max_smem_per_sm'] // max(smem, 1)
    by_regs = hw['max_regs_per_sm'] // max(regs * thr, 1)
    by_threads = hw['max_threads_per_sm'] // max(thr, 1)
    return min(by_smem, by_regs, by_threads, hw['max_blocks_per_sm'])


def predict_interleaved_us(M_per_t, K, N, T_steps, bM, bN, bK, ns, thr,
                            n_membranes=1):
    """Predict interleaved kernel latency in microseconds.

    Returns (total_us, details_dict).
    """
    hw = _get_hw()
    sm_count = hw['sm_count']
    tc_flops_per_us = hw['tc_tflops'] * 1e6  # FLOPS per microsecond
    dram_bytes_per_us = hw['mem_bw_gbps'] * 1e3  # bytes per microsecond
    l2_bytes_per_us = hw['l2_bw_gbps'] * 1e3

    # Grid size
    grid_M = math.ceil(M_per_t / bM)
    grid_N = math.ceil(N / bN)
    n_ctas = grid_M * grid_N
    occ = _occupancy(bM, bN, bK, ns, thr, n_membranes)

    # Waves: how many rounds of CTA launches
    ctas_per_round = min(n_ctas, sm_count * occ)
    n_waves = math.ceil(n_ctas / max(ctas_per_round, 1))

    # SM utilization fraction
    sm_active = min(n_ctas, sm_count) / sm_count

    # ── Per-timestep GEMM cost ──
    flops_gemm = 2.0 * M_per_t * K * N

    # Tile efficiency: tensor core MMA instructions operate on 16×16 fragments.
    # Small tiles (bM=16) have fewer fragments per CTA → worse instruction-level
    # parallelism, more pipeline bubbles, less register reuse.
    # Reference area scales with GPU: A100 benefits from larger tiles.
    tile_area = bM * bN
    ref_area = 8192 if hw['max_smem_per_sm'] >= 160 * 1024 else 4096
    tile_efficiency = min(1.0, (tile_area / ref_area) ** 0.5)
    # Clamp: even the smallest tile achieves at least 25% of peak
    tile_efficiency = max(0.25, tile_efficiency)

    t_compute = flops_gemm / (tc_flops_per_us * sm_active * tile_efficiency)

    # Memory: data + weight per timestep
    data_bytes = M_per_t * K * 2  # FP16 input tile
    weight_bytes = K * N * 2      # FP16 weight

    # Weight caching: after t=0, weight likely in L2 (if < L2 size)
    weight_in_l2 = (weight_bytes < hw['l2_size_bytes'] * 0.5)
    weight_bw = l2_bytes_per_us if weight_in_l2 else dram_bytes_per_us

    t_data = data_bytes / dram_bytes_per_us     # data always from DRAM
    t_weight = weight_bytes / weight_bw          # weight from L2 or DRAM
    t_memory_gemm = t_data + t_weight

    t_gemm = max(t_compute, t_memory_gemm) * n_waves

    # ── Epilogue cost (BN + IF per element) ──
    # BN: 1 mul + 1 add = 2 FP32 ops per element
    # IF: 1 add (integrate) + 1 compare + 2 select = 4 ops per element
    epilogue_ops_per_elem = 6  # BN + IF
    epilogue_ops_per_elem += 4 * (n_membranes - 1)  # extra neurons
    total_epilogue_ops = M_per_t * N * epilogue_ops_per_elem
    # FP32 ALU throughput: roughly half of TC throughput
    fp32_flops_per_us = hw['tc_tflops'] * 0.5e6
    t_epilogue_compute = total_epilogue_ops / (fp32_flops_per_us * sm_active)

    # Epilogue memory: write output (FP16)
    output_bytes = M_per_t * N * 2
    t_epilogue_mem = output_bytes / dram_bytes_per_us

    t_epilogue = max(t_epilogue_compute, t_epilogue_mem) * n_waves

    # ── Membrane init/writeback ──
    mem_bytes = M_per_t * N * 4 * n_membranes  # FP32 read + write
    t_membrane = 2 * mem_bytes / dram_bytes_per_us

    # ── Total ──
    t_total = T_steps * (t_gemm + t_epilogue) + t_membrane

    details = {
        'grid': n_ctas, 'waves': n_waves, 'occ': occ,
        'sm_active': sm_active, 'tile_efficiency': tile_efficiency,
        't_compute': t_compute, 't_memory_gemm': t_memory_gemm,
        't_gemm': t_gemm, 't_epilogue': t_epilogue,
        't_membrane': t_membrane, 't_total': t_total,
        'weight_in_l2': weight_in_l2,
    }
    return t_total, details


def predict_decomposed_us(M_total, K, N):
    """Predict decomposed Conv+BN (single GEMM) latency in microseconds."""
    hw = _get_hw()
    sm_count = hw['sm_count']
    tc_flops_per_us = hw['tc_tflops'] * 1e6
    dram_bytes_per_us = hw['mem_bw_gbps'] * 1e3

    flops = 2.0 * M_total * K * N
    t_compute = flops / (tc_flops_per_us * 1.0)  # full SM utilization assumed

    data_bytes = M_total * K * 2
    weight_bytes = K * N * 2
    output_bytes = M_total * N * 2
    t_memory = (data_bytes + weight_bytes + output_bytes) / dram_bytes_per_us

    return max(t_compute, t_memory)


def select_config_analytical(M_per_t, K, N, T_steps,
                              n_membranes=1,
                              return_all=False):
    """Select best tile config using analytical cost model.

    Returns dict with block_M, block_N, block_K, num_stages, threads, latency_us.
    """
    hw = _get_hw()
    is_high_bw = hw['max_smem_per_sm'] >= 160 * 1024  # A100+

    candidates = []

    bM_choices = [16, 32, 64, 128, 256]
    bN_choices = [32, 64, 128, 256]
    bK_choices = [32, 64, 128]
    ns_choices = [2, 3, 4] if is_high_bw else [2, 3]
    thr_choices = [128, 256, 512] if is_high_bw else [128, 256]

    for bM in bM_choices:
        for bN in bN_choices:
            for bK in bK_choices:
                if bK > K:
                    continue
                for ns in ns_choices:
                    for thr in thr_choices:
                        if bM * bN < thr:
                            continue
                        smem = _smem_bytes(bM, bN, bK, ns)
                        if smem > hw['max_smem_per_sm']:
                            continue
                        grid = math.ceil(M_per_t / bM) * math.ceil(N / bN)
                        if grid < hw['sm_count'] // 8:
                            continue

                        us, details = predict_interleaved_us(
                            M_per_t, K, N, T_steps, bM, bN, bK, ns, thr,
                            n_membranes)

                        cfg = dict(block_M=bM, block_N=bN, block_K=bK,
                                   num_stages=ns, threads=thr,
                                   predicted_us=us)
                        candidates.append((us, cfg, details))

    candidates.sort(key=lambda x: x[0])

    if return_all:
        return candidates

    if candidates:
        best_us, best_cfg, _ = candidates[0]
        best_cfg['latency_us'] = best_us
        return best_cfg

    # Fallback
    return dict(block_M=32, block_N=64, block_K=min(32, K),
                num_stages=2, threads=128, latency_us=float('inf'))
