"""Target profile: every hardware-dependent tuning knob of sengine in one place.

A `TargetProfile` is pure data. The tuning modules (`tuning/roofline.py`,
`tuning/analytical.py`, `build/tilelang_compiler.py`), the IR optimizer, the
TDL cost model, the kernel autotune spaces and the runtime (L2 persistence,
L1 carveout) read their constants from the *active* profile instead of
branching on the detected GPU. Profiles live in `ada.py` (RTX 4090),
`a100.py` (A100) and `orin.py` (Jetson AGX Orin); `get_target()` picks one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HwFallback:
    """Hardware table used when the CUDA API / calibration is unavailable."""
    gpu_name: str
    sm_count: int
    max_smem: int            # bytes of shared memory per SM
    tc_tflops: float         # tensor-core FP16 TFLOPS
    mem_bw_gbps: float       # DRAM bandwidth GB/s
    l2_size_bytes: int
    max_regs_per_sm: int = 65536
    max_threads_per_sm: int = 1536
    max_blocks_per_sm: int = 16


@dataclass(frozen=True)
class TileSpace:
    """A tile-config search space (bM x bN x bK x num_stages x threads)."""
    bM: tuple[int, ...]
    bN: tuple[int, ...]
    bK: tuple[int, ...]
    ns: tuple[int, ...]
    thr: tuple[int, ...]
    bK_divisors: tuple[int, ...] = ()   # extra bK values kept only when K % bK == 0
    bK_fp32_max: int = 64               # fp32: drop bK above this (T.copy vectorization)
    min_grid_divisor: int = 8           # grid must reach sm_count // divisor


@dataclass(frozen=True)
class RooflineRank:
    """Ranking weights of `tuning/roofline.py::prune_candidates`."""
    area_cap: int = 4096
    grid_ok_mode: str = 'half'   # 'half' (>= sm/2 -> 1.0) or 'wave' (tiers at 2*sm, sm, sm/2)
    k_iters_ref: int = 256       # K-loop depth above which configs are penalized
    occ_bonus: float = 0.0       # 1 + occ_bonus * min(occ, 4)
    ns_bonus: float = 0.0        # 1 + ns_bonus * (num_stages - 2)
    wave_efficiency: bool = False


@dataclass(frozen=True)
class HeuristicPick:
    """Non-autotuned defaults (`_pick_config` / `_pick_interleaved_config`)."""
    interleaved_prefs: tuple[tuple[int, int, int], ...]   # (bM, bN, threads) in preference order
    interleaved_bk: int
    interleaved_ns: int
    bm_thresholds: tuple[tuple[int, int], ...]            # (M >= threshold -> bM), descending
    bm_default: int
    bk: int
    ns: int
    wide_bn: bool = False    # prefer bN=128 when F >= 128 (fp16)


@dataclass(frozen=True)
class TargetProfile:
    name: str
    arch: str                                   # 'sm_89' / 'sm_80' / 'sm_87'
    nvcc_paths: tuple[str, ...]
    hw: HwFallback
    l2_bw_multiplier: float                     # L2 BW estimate = DRAM BW * this

    # tuning/roofline.py
    roofline_space: TileSpace
    roofline_rank: RooflineRank
    known_good: tuple[dict, ...]

    # tuning/analytical.py
    analytical_space: TileSpace
    analytical_ref_area: int
    analytical_weight_l2_frac: float

    # build/tilelang_compiler.py
    pick: HeuristicPick
    autotune_space: TileSpace                   # _autotune_config (per-timestep kernels)
    autotune_interleaved_space: TileSpace       # _autotune_interleaved
    conv3x3_interleaved_variant: str = 't_loop'  # 't_loop' | 't_unrolled4'
    conv3x3_num_stages_override: int | None = None

    # kernels/conv2d_bn_if_t4.py::_make_configs_conv
    conv_autotune_bM: tuple[int, ...] = (32, 64, 128, 256)
    conv_autotune_bN: tuple[int, ...] = (32, 64, 128, 256)

    # tdl/cost_model.py::CalibrationFactors (field name -> value)
    calibration: dict = field(default_factory=dict)

    # optimizer.py::classify_bound_and_assign_tilelang
    fused_per_t_block_m_ref: int = 16
    fused_per_t_sm_default: int = 128

    # runtime
    l2_persist: bool = False
    l1_carveout_native_kernels: bool = False

    # kernel .so cache dir under <repo>/.cache (per batch size)
    cache_subdir: str = 'sengine_B{B}'

    def bK_for(self, space: TileSpace, K: int, bpe: int) -> list[int]:
        """Resolve a space's bK choices for a reduction size and element width."""
        base = [b for b in space.bK if bpe == 2 or b <= space.bK_fp32_max]
        divs = [b for b in space.bK_divisors
                if (bpe == 2 or b <= space.bK_fp32_max) and K > 0 and K % b == 0 and b <= K]
        return sorted(set(base) | set(divs))

    def cache_dir_for(self, batch_size: int) -> str:
        return self.cache_subdir.format(B=batch_size)

    def __str__(self) -> str:
        return (f"TargetProfile({self.name}: {self.arch}, {self.hw.sm_count} SMs, "
                f"{self.hw.max_smem // 1024}KB smem, conv3x3={self.conv3x3_interleaved_variant}, "
                f"l2_persist={self.l2_persist}, l1_carveout={self.l1_carveout_native_kernels})")
