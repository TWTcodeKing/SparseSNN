"""Hardware calibration: measure actual GPU throughput for roofline model.

Reads what it can from CUDA API (SM count, L2 size, registers, max threads).
Measures memory bandwidth and tensor core throughput with microbenchmarks.
Results are cached per GPU UUID — calibration runs once (~2 seconds).
"""

from __future__ import annotations
import os
import json
import torch

_CACHE_DIR = os.path.expanduser('~/.cache/sengine_hw')
_CALIBRATION = None


def calibrate(device_id: int = 0) -> dict:
    """Return hardware specs for the current GPU.

    First call runs microbenchmarks (~2s). Results are cached by GPU UUID.
    """
    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION

    torch.cuda.set_device(device_id)
    props = torch.cuda.get_device_properties(device_id)

    # Check cache
    cache_file = os.path.join(_CACHE_DIR, f'{props.uuid}.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                _CALIBRATION = json.load(f)
            return _CALIBRATION
        except Exception:
            pass

    # ── Read from API ──
    hw = {
        'gpu_name': props.name,
        'gpu_uuid': str(props.uuid),
        'arch': f'sm_{props.major}{props.minor}',
        'sm_count': props.multi_processor_count,
        'l2_size_bytes': props.L2_cache_size,
        'max_regs_per_sm': props.regs_per_multiprocessor,
        'max_threads_per_sm': props.max_threads_per_multi_processor,
        'total_mem_bytes': props.total_memory,
        'warp_size': props.warp_size,
    }

    # Max blocks per SM (from arch)
    if props.major >= 9:
        hw['max_blocks_per_sm'] = 32
    elif props.major >= 8:
        hw['max_blocks_per_sm'] = 16
    else:
        hw['max_blocks_per_sm'] = 16

    # Max shared memory per SM (configurable)
    hw['max_smem_per_sm'] = getattr(props, 'max_shared_memory_per_block_optin',
                                     100 * 1024)

    # ── Microbenchmark: memory bandwidth ──
    hw['mem_bw_gbps'] = _measure_mem_bw(device_id)

    # ── Microbenchmark: FP16 tensor core throughput ──
    hw['tc_tflops'] = _measure_tc_throughput(device_id)

    # ── Derived ──
    hw['ridge_point'] = (hw['tc_tflops'] * 1e12) / (hw['mem_bw_gbps'] * 1e9)

    # Cache
    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        with open(cache_file, 'w') as f:
            json.dump(hw, f, indent=2)
    except Exception:
        pass

    _CALIBRATION = hw
    return hw


def _measure_mem_bw(device_id: int, size_mb: int = 256) -> float:
    """Measure DRAM bandwidth with a large memcpy (GB/s)."""
    n = size_mb * 1024 * 1024 // 2  # FP16 elements
    src = torch.randn(n, dtype=torch.float16, device=f'cuda:{device_id}')
    dst = torch.empty_like(src)

    # Warmup
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    iters = 20
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    bytes_moved = n * 2 * 2  # read + write
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    return round(gbps, 1)


def _measure_tc_throughput(device_id: int) -> float:
    """Measure FP16 tensor core throughput with a large GEMM (TFLOPS)."""
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, dtype=torch.float16, device=f'cuda:{device_id}')
    b = torch.randn(K, N, dtype=torch.float16, device=f'cuda:{device_id}')

    # Warmup
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    iters = 20
    s.record()
    for _ in range(iters):
        torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    flops = 2.0 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    return round(tflops, 1)
