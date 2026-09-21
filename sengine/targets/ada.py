"""RTX 4090 (Ada, sm_89): the reference target. Values are sengine's historical defaults."""

from sengine.targets.base import (HeuristicPick, HwFallback, RooflineRank,
                                  TargetProfile, TileSpace)

ADA = TargetProfile(
    name='ada',
    arch='sm_89',
    nvcc_paths=('/usr/local/cuda-12.8/bin/nvcc', '/usr/local/cuda/bin/nvcc'),
    hw=HwFallback(gpu_name='NVIDIA GeForce RTX 4090', sm_count=128, max_smem=100 * 1024,
                  tc_tflops=82.6, mem_bw_gbps=1008, l2_size_bytes=72 * 1024 * 1024),
    l2_bw_multiplier=4.0,

    roofline_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                             bK=(32, 64, 128), ns=(2, 3), thr=(128, 256),
                             bK_divisors=(32, 48, 64, 96, 128), bK_fp32_max=64,
                             min_grid_divisor=8),
    roofline_rank=RooflineRank(area_cap=4096, grid_ok_mode='half', k_iters_ref=256,
                               occ_bonus=0.0, ns_bonus=0.0, wave_efficiency=False),
    known_good=(
        dict(block_M=64, block_N=64, block_K=32, num_stages=2, threads=128),
        dict(block_M=32, block_N=64, block_K=32, num_stages=2, threads=128),
        dict(block_M=64, block_N=64, block_K=64, num_stages=2, threads=128),
        dict(block_M=64, block_N=128, block_K=32, num_stages=2, threads=128),
        dict(block_M=128, block_N=64, block_K=32, num_stages=2, threads=128),
        # Asymmetric tiles for narrow GEMM shapes
        dict(block_M=32, block_N=128, block_K=32, num_stages=2, threads=128),
        dict(block_M=128, block_N=32, block_K=32, num_stages=2, threads=128),
        dict(block_M=64, block_N=64, block_K=64, num_stages=3, threads=128),
    ),

    analytical_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                               bK=(32, 64, 128), ns=(2, 3), thr=(128, 256),
                               min_grid_divisor=8),
    analytical_ref_area=4096,
    analytical_weight_l2_frac=0.5,

    pick=HeuristicPick(interleaved_prefs=((32, 64, 128), (64, 64, 128), (64, 32, 128)),
                       interleaved_bk=32, interleaved_ns=2,
                       bm_thresholds=((100000, 128), (10000, 64)), bm_default=32,
                       bk=32, ns=2, wide_bn=False),
    autotune_space=TileSpace(bM=(32, 64, 128, 256), bN=(32, 64, 128, 256),
                             bK=(32, 64, 128), ns=(2, 3), thr=(128, 256),
                             bK_divisors=(48, 96), min_grid_divisor=4),
    autotune_interleaved_space=TileSpace(bM=(16, 32, 64, 128, 256), bN=(32, 64, 128, 256),
                                         bK=(32, 64, 128), ns=(2, 3), thr=(128,),
                                         bK_divisors=(48, 96), min_grid_divisor=8),
    conv3x3_interleaved_variant='t_loop',
    conv3x3_num_stages_override=None,
    conv_autotune_bM=(32, 64, 128, 256),
    conv_autotune_bN=(32, 64, 128, 256),

    calibration=dict(conv_factor=1.3, neuron_factor=1.0, bn_factor=1.0,
                     elementwise_factor=1.0, pool_factor=1.0, linear_factor=1.2,
                     matmul_factor=1.2, l2_hit_rate_conv=0.98, l2_hit_rate_neuron=0.50),
    fused_per_t_block_m_ref=16,
    fused_per_t_sm_default=128,
    l2_persist=False,
    l1_carveout_native_kernels=False,
    cache_subdir='sengine_B{B}',
)
