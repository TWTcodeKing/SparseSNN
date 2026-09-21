"""Jetson AGX Orin (Ampere GA10B, sm_87, 16 SMs, 100KB smem, 4MB L2, LPDDR5).

With 16 SMs a single wave miss idles 1/16 of the GPU, so grid coverage,
occupancy and wave efficiency dominate the tile ranking. The 3x3 interleaved
Conv+BN+IF kernel is the hand-unrolled T=4 variant (weight tile loaded once per
K iteration for all four timesteps), which requires num_stages=1. Weights are
pinned in L2 (`l2_persist`) and the native memory-bound kernels ask for the
max-L1 carveout.
"""

from dataclasses import replace

from sengine.targets.ada import ADA
from sengine.targets.base import HeuristicPick, HwFallback, RooflineRank, TileSpace

ORIN = replace(
    ADA,
    name='orin',
    arch='sm_87',
    nvcc_paths=('/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12.8/bin/nvcc',
                '/usr/local/cuda-12.6/bin/nvcc'),
    hw=HwFallback(gpu_name='Jetson AGX Orin', sm_count=16, max_smem=100 * 1024,
                  tc_tflops=5.3, mem_bw_gbps=204.8, l2_size_bytes=4 * 1024 * 1024),
    l2_bw_multiplier=2.0,

    roofline_space=TileSpace(bM=(16, 32, 64, 128), bN=(32, 64, 128),
                             bK=(32, 64), ns=(2, 3, 4), thr=(128, 256),
                             bK_divisors=(32, 48, 64, 96), bK_fp32_max=32,
                             min_grid_divisor=4),
    roofline_rank=RooflineRank(area_cap=4096, grid_ok_mode='wave', k_iters_ref=128,
                               occ_bonus=0.3, ns_bonus=0.08, wave_efficiency=True),
    known_good=(
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
    ),

    analytical_space=TileSpace(bM=(16, 32, 64, 128), bN=(32, 64, 128),
                               bK=(32, 64), ns=(2, 3, 4), thr=(128, 256),
                               min_grid_divisor=4),
    analytical_ref_area=2048,
    analytical_weight_l2_frac=0.3,

    pick=HeuristicPick(interleaved_prefs=((128, 64, 256), (64, 128, 256), (64, 64, 128),
                                          (32, 64, 128), (64, 32, 128), (32, 32, 128)),
                       interleaved_bk=32, interleaved_ns=3,
                       bm_thresholds=((10000, 64), (1000, 32)), bm_default=16,
                       bk=32, ns=3, wide_bn=False),
    autotune_space=TileSpace(bM=(32, 64, 128), bN=(32, 64, 128),
                             bK=(32, 64), ns=(2, 3, 4), thr=(128, 256),
                             bK_divisors=(48, 96), min_grid_divisor=4),
    autotune_interleaved_space=TileSpace(bM=(16, 32, 64, 128), bN=(32, 64, 128),
                                         bK=(32, 64), ns=(2, 3, 4), thr=(128, 256),
                                         bK_divisors=(48, 96), min_grid_divisor=4),
    conv3x3_interleaved_variant='t_unrolled4',
    conv3x3_num_stages_override=1,
    conv_autotune_bM=(16, 32, 64, 128),
    conv_autotune_bN=(32, 64, 128),

    calibration=dict(conv_factor=1.5, neuron_factor=1.1, bn_factor=1.1,
                     elementwise_factor=1.1, pool_factor=1.1, linear_factor=1.3,
                     matmul_factor=1.3, l2_hit_rate_conv=0.3, l2_hit_rate_neuron=0.1),
    fused_per_t_block_m_ref=32,
    fused_per_t_sm_default=16,
    l2_persist=True,
    l1_carveout_native_kernels=True,
    cache_subdir='sengine_edge_B{B}',
)
