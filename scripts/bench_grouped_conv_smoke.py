#!/usr/bin/env python3
"""Smoke test: Grouped Conv2d+BN kernel performance with autotuning.

Tests the bottleneck SpikingResFormer GWFFN shapes:
  - 1536→1536 K3 S1 28×28 groups=4 (SpikingResFormer-M stage 2)
  - 3072→3072 K3 S1 14×14 groups=4 (SpikingResFormer-M stage 3)
  - 256→256  K3 S1 56×56 groups=4 (SpikingResFormer-M stage 1)

Benchmarks with autotuning to find optimal tile config per GPU.
Also compares against cuDNN (torch.nn.functional.conv2d) as reference.

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_grouped_conv_smoke.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ensure correct CUDA
for p in ['/usr/local/cuda-12.8', '/usr/local/cuda-12.6', '/usr/local/cuda']:
    if os.path.isdir(p):
        os.environ.setdefault('CUDA_HOME', p)
        os.environ['PATH'] = os.path.join(p, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch
import torch.nn.functional as F
import tilelang.language as T

from sengine.kernels.grouped_conv_bn import grouped_conv_bn_kernel
from sengine.tuning.roofline import select_config_roofline


def benchmark(fn, warmup=100, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


# SpikingResFormer-M bottleneck shapes (GWFFN grouped conv)
SHAPES = [
    # (TB, C_in, H, W, C_out, K, S, P, groups, name)
    (16, 256,  56, 56, 256,  3, 1, 1, 4, "256_256_K3_S1_56x56_g4 (stage1)"),
    (16, 1536, 28, 28, 1536, 3, 1, 1, 4, "1536_1536_K3_S1_28x28_g4 (stage2)"),
    (16, 3072, 14, 14, 3072, 3, 1, 1, 4, "3072_3072_K3_S1_14x14_g4 (stage3)"),
    # Also test non-grouped versions for comparison
    (16, 1536, 28, 28, 1536, 3, 1, 1, 1, "1536_1536_K3_S1_28x28_g1 (standard)"),
]


def main():
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
    print(f"Testing grouped conv bottleneck shapes for SpikingResFormer-M (TB=16, T=4 B=4)")
    print()

    for TB, C_in, H, W, C_out, K, S, P, groups, name in SHAPES:
        print(f"{'='*70}")
        print(f"  {name}")
        print(f"  TB={TB} C_in={C_in} C_out={C_out} K={K} S={S} groups={groups} H×W={H}×{W}")

        OH = (H + 2*P - K) // S + 1
        OW = (W + 2*P - K) // S + 1
        C_in_per_g = C_in // groups
        C_out_per_g = C_out // groups
        K_red = K * K * C_in_per_g
        M = TB * OH * OW

        print(f"  M={M} K_red={K_red} N(per_g)={C_out_per_g} groups={groups}")
        print(f"{'='*70}")

        # Allocate test tensors
        data = torch.randn(TB, H, W, C_in, dtype=torch.float16, device=device)
        weight = torch.randn(K, K, C_in_per_g, C_out, dtype=torch.float16, device=device)
        bn_s = torch.ones(C_out, dtype=torch.float32, device=device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=device)

        # ── 1. Autotuned TileLang grouped conv ──
        print(f"\n  1. TileLang grouped conv (autotuned):")

        def _compile(cfg):
            return grouped_conv_bn_kernel(
                TB=TB, C_in=C_in, H=H, W=W, C_out=C_out,
                K=K, S=S, D=1, P=P, groups=groups,
                io_dtype=T.float16,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        profile_args = (data, weight, bn_s, bn_b)

        print(f"     Autotuning (roofline + profiling)...")
        t0 = time.time()
        best_cfg = select_config_roofline(
            M_per_t=M // 4,  # per-timestep spatial (for interleaved comparison)
            K=K_red, N=C_out_per_g, T_steps=4,
            compile_fn=_compile, profile_args=profile_args,
            top_k=7, n_profile=200, bpe=2)
        tune_time = time.time() - t0
        print(f"     Config: bM={best_cfg['block_M']} bN={best_cfg['block_N']} "
              f"bK={best_cfg['block_K']} ns={best_cfg['num_stages']} "
              f"latency={best_cfg.get('latency_us', 0):.1f}µs (tuned in {tune_time:.1f}s)")

        kern = _compile(best_cfg)
        ms_tl = benchmark(lambda: kern(data, weight, bn_s, bn_b))
        print(f"     Latency: {ms_tl:.3f} ms ({ms_tl*1000:.1f} µs)")

        # ── 2. Heuristic config (no autotuning) ──
        from sengine.build.tilelang_compiler import _pick_config
        heur_cfg = _pick_config(M, K_red, C_out_per_g, bpe=2)
        print(f"\n  2. TileLang heuristic config (no autotune):")
        print(f"     Config: bM={heur_cfg['block_M']} bN={heur_cfg['block_N']} bK={heur_cfg['block_K']}")
        try:
            kern_h = _compile(heur_cfg)
            ms_heur = benchmark(lambda: kern_h(data, weight, bn_s, bn_b))
            print(f"     Latency: {ms_heur:.3f} ms ({ms_heur*1000:.1f} µs)")
        except Exception as e:
            print(f"     FAILED: {e}")
            ms_heur = float('inf')

        # ── 3. cuDNN reference ──
        print(f"\n  3. cuDNN reference (torch.nn.functional.conv2d):")
        data_nchw = data.permute(0, 3, 1, 2).contiguous()
        # cuDNN weight: (C_out, C_in_per_g, K, K)
        w_cudnn = torch.randn(C_out, C_in_per_g, K, K, dtype=torch.float16, device=device)
        ms_cudnn = benchmark(lambda: F.conv2d(data_nchw, w_cudnn, padding=P, stride=S, groups=groups))
        print(f"     Latency: {ms_cudnn:.3f} ms ({ms_cudnn*1000:.1f} µs)")

        # ── 4. Correctness check ──
        print(f"\n  4. Correctness (TileLang vs cuDNN):")
        # Use consistent weights for comparison
        # TileLang weight: (K, K, C_in_per_g, C_out) NHWC
        # cuDNN weight: (C_out, C_in_per_g, K, K) NCHW
        w_ref = torch.randn(C_out, C_in_per_g, K, K, dtype=torch.float16, device=device)
        w_tl = w_ref.permute(2, 3, 1, 0).contiguous()  # → (K, K, C_in_per_g, C_out)

        out_tl = kern(data, w_tl, bn_s, bn_b)
        ref_nchw = F.conv2d(data_nchw, w_ref, padding=P, stride=S, groups=groups)
        # BN
        ref_nchw = ref_nchw * bn_s.view(1, -1, 1, 1) + bn_b.view(1, -1, 1, 1)
        ref_nhwc = ref_nchw.permute(0, 2, 3, 1).contiguous()

        cos = F.cosine_similarity(out_tl.float().flatten(), ref_nhwc.float().flatten(), dim=0)
        maxdiff = (out_tl.float() - ref_nhwc.float()).abs().max().item()
        print(f"     Cosine: {cos.item():.6f}, MaxDiff: {maxdiff:.4f}")

        # ── Summary ──
        print(f"\n  Summary:")
        print(f"     TileLang autotuned: {ms_tl:.3f} ms")
        print(f"     TileLang heuristic: {ms_heur:.3f} ms")
        print(f"     cuDNN:              {ms_cudnn:.3f} ms")
        speedup = ms_tl / ms_cudnn if ms_cudnn > 0 else 0
        print(f"     TileLang/cuDNN:     {speedup:.2f}x ({'faster' if speedup < 1 else 'SLOWER'})")
        print()


if __name__ == '__main__':
    main()
