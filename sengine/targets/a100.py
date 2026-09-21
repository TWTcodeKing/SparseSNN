"""A100 (Ampere data-center, sm_80): the former `is_ampere_plus` / `is_high_bw` branch."""

from dataclasses import replace

from sengine.targets.ada import ADA
from sengine.targets.base import HeuristicPick, HwFallback, RooflineRank, TileSpace

A100 = replace(
    ADA,
    name='a100',
    arch='sm_80',
    hw=HwFallback(gpu_name='NVIDIA A100', sm_count=108, max_smem=164 * 1024,
                  tc_tflops=312.0, mem_bw_gbps=2039, l2_size_bytes=40 * 1024 * 1024),
    l2_bw_multiplier=4.0,

    roofline_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                             bK=(32, 64, 128), ns=(2, 3, 4, 5), thr=(128, 256, 512),
                             bK_divisors=(32, 48, 64, 96, 128), bK_fp32_max=64,
                             min_grid_divisor=16),
    roofline_rank=RooflineRank(area_cap=16384, grid_ok_mode='half', k_iters_ref=256,
                               occ_bonus=0.15, ns_bonus=0.05, wave_efficiency=False),
    known_good=(
        # cuBLAS A100 typical configs: square & rectangular
        dict(block_M=128, block_N=128, block_K=64, num_stages=3, threads=256),
        dict(block_M=128, block_N=128, block_K=32, num_stages=4, threads=256),
        dict(block_M=64, block_N=128, block_K=64, num_stages=4, threads=256),
        dict(block_M=128, block_N=64, block_K=64, num_stages=4, threads=256),
        dict(block_M=64, block_N=64, block_K=64, num_stages=4, threads=128),
        dict(block_M=256, block_N=64, block_K=64, num_stages=3, threads=256),
        dict(block_M=64, block_N=256, block_K=64, num_stages=3, threads=256),
        # Asymmetric tiles for M>>N or M<<N shapes
        dict(block_M=32, block_N=128, block_K=64, num_stages=4, threads=128),
        dict(block_M=128, block_N=32, block_K=64, num_stages=4, threads=128),
        dict(block_M=256, block_N=128, block_K=64, num_stages=3, threads=256),
        dict(block_M=128, block_N=256, block_K=64, num_stages=3, threads=256),
        # Deeper pipeline variants
        dict(block_M=64, block_N=128, block_K=64, num_stages=5, threads=128),
        dict(block_M=128, block_N=64, block_K=64, num_stages=5, threads=128),
        # K-divisibility configs for common C_in values (96, 192, 384, 768)
        dict(block_M=64, block_N=128, block_K=96, num_stages=3, threads=256),
        dict(block_M=128, block_N=64, block_K=96, num_stages=3, threads=256),
        dict(block_M=64, block_N=64, block_K=48, num_stages=4, threads=128),
    ),

    analytical_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                               bK=(32, 64, 128), ns=(2, 3, 4), thr=(128, 256, 512),
                               min_grid_divisor=8),
    analytical_ref_area=8192,
    analytical_weight_l2_frac=0.5,

    pick=HeuristicPick(interleaved_prefs=((64, 128, 128), (128, 64, 128), (64, 64, 128), (32, 64, 128)),
                       interleaved_bk=64, interleaved_ns=3,
                       bm_thresholds=((100000, 128), (10000, 64)), bm_default=32,
                       bk=64, ns=3, wide_bn=True),
    autotune_space=TileSpace(bM=(32, 64, 128, 256), bN=(32, 64, 128, 256),
                             bK=(32, 64, 128), ns=(2, 3, 4), thr=(128, 256, 512),
                             bK_divisors=(48, 96), min_grid_divisor=4),
    autotune_interleaved_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                                         bK=(32, 64, 128), ns=(2, 3, 4), thr=(128, 256, 512),
                                         bK_divisors=(48, 96), min_grid_divisor=8),
)
