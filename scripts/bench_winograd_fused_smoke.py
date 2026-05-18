#!/usr/bin/env python3
"""Smoke test: Implicit Winograd fused TileLang kernel vs im2col vs cuDNN.

Tests the single fused Winograd kernel (no intermediate buffer) on
SpikingResFormer bottleneck shapes.

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_winograd_fused_smoke.py
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

for p in ['/usr/local/cuda-12.8', '/usr/local/cuda-12.6', '/usr/local/cuda']:
    if os.path.isdir(p):
        os.environ.setdefault('CUDA_HOME', p)
        os.environ['PATH'] = os.path.join(p, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch
import torch.nn.functional as F
import tilelang.language as T

from sengine.kernels.winograd_conv import (
    winograd_conv2d_bn_kernel, winograd_transform_weight, _build_transform_tensors)
from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_t4_kernel
from sengine.tuning.roofline import select_config_roofline


def benchmark(fn, warmup=100, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


SHAPES = [
    # (TB, C_in, C_out, H, W, name)
    (4,  64,   64,  56, 56, "ResNet: 64→64 56×56"),
    (4,  128, 128,  28, 28, "ResNet: 128→128 28×28"),
    (4,  256, 256,  14, 14, "ResNet: 256→256 14×14"),
    (16, 256, 256,  56, 56, "SRF-M: 256→256 56×56"),
    (16, 1536, 1536, 28, 28, "SRF-M: 1536→1536 28×28"),
    (16, 3072, 3072, 14, 14, "SRF-M: 3072→3072 14×14"),
    (16, 2048, 2048, 28, 28, "SRF-L: 2048→2048 28×28"),
    (16, 4096, 4096, 14, 14, "SRF-L: 4096→4096 14×14"),
]


def main():
    device_id = torch.cuda.current_device()
    device = f'cuda:{device_id}'
    props = torch.cuda.get_device_properties(device_id)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
    print(f"Implicit Winograd fused TileLang vs im2col TileLang vs cuDNN")
    print()

    P = 1
    it, ic, oc = _build_transform_tensors(device)
    results = []

    for TB, C_in, C_out, H, W, name in SHAPES:
        OH, OW = H, W
        K_red_imcol = 9 * C_in
        tile_h = (H + 2*P - 4) // 2 + 1
        tile_w = (W + 2*P - 4) // 2 + 1
        M_tiles = TB * tile_h * tile_w
        M = TB * OH * OW

        print(f"{'='*70}")
        print(f"  {name} | TB={TB} C_in={C_in} M={M} M_tiles={M_tiles}")
        print(f"{'='*70}")

        data_nhwc = torch.randn(TB, H, W, C_in, dtype=torch.float16, device=device)
        data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous()
        w_nchw = torch.randn(C_out, C_in, 3, 3, dtype=torch.float16, device=device)
        w_nhwc = w_nchw.permute(2, 3, 1, 0).contiguous()
        bn_s = torch.ones(C_out, dtype=torch.float32, device=device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=device)

        # ── 1. Winograd fused TileLang (implicit transform) ──
        U = winograd_transform_weight(w_nchw)
        def _compile_wino(cfg):
            return winograd_conv2d_bn_kernel(
                TB=TB, C_in=C_in, H=H, W=W, C_out=C_out, P=P,
                io_dtype=T.float16,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        wino_profile = (data_nhwc, U, bn_s, bn_b, it, ic, oc)

        print(f"  Winograd fused: autotuning...", end='', flush=True)
        try:
            torch.cuda.set_device(device_id)
            wino_cfg = select_config_roofline(
                M_per_t=M_tiles, K=C_in, N=C_out, T_steps=1,
                compile_fn=_compile_wino, profile_args=wino_profile,
                top_k=5, n_profile=200, bpe=2)
            torch.cuda.set_device(device_id)
            kern_wino = _compile_wino(wino_cfg)
            ms_wino = benchmark(lambda: kern_wino(data_nhwc, U, bn_s, bn_b, it, ic, oc))
            print(f" {ms_wino:.3f}ms (bM={wino_cfg['block_M']} bN={wino_cfg['block_N']} bK={wino_cfg['block_K']})")
        except Exception as e:
            print(f" FAILED: {e}")
            ms_wino = float('inf')

        # ── 2. im2col TileLang (autotuned) ──
        def _compile_imcol(cfg):
            return conv2d_bn_t4_kernel(
                TB=TB, C_in=C_in, H=H, W=W, F=C_out, K=3, S=1, D=1, P=P,
                io_dtype=T.float16,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})
        imcol_profile = (data_nhwc, w_nhwc, bn_s, bn_b)

        print(f"  im2col:         autotuning...", end='', flush=True)
        try:
            torch.cuda.set_device(device_id)
            imcol_cfg = select_config_roofline(
                M_per_t=M, K=K_red_imcol, N=C_out, T_steps=1,
                compile_fn=_compile_imcol, profile_args=imcol_profile,
                top_k=5, n_profile=200, bpe=2)
            torch.cuda.set_device(device_id)
            kern_imcol = _compile_imcol(imcol_cfg)
            ms_imcol = benchmark(lambda: kern_imcol(data_nhwc, w_nhwc, bn_s, bn_b))
            print(f" {ms_imcol:.3f}ms (bM={imcol_cfg['block_M']} bN={imcol_cfg['block_N']} bK={imcol_cfg['block_K']})")
        except Exception as e:
            print(f" FAILED: {e}")
            ms_imcol = float('inf')

        # ── 3. cuDNN reference ──
        bn_s4 = bn_s.view(1, -1, 1, 1); bn_b4 = bn_b.view(1, -1, 1, 1)
        ms_cudnn = benchmark(lambda: F.conv2d(data_nchw, w_nchw, padding=P) * bn_s4 + bn_b4)
        print(f"  cuDNN:          {ms_cudnn:.3f}ms")

        # ── Correctness ──
        if ms_wino < float('inf'):
            out_wino = kern_wino(data_nhwc, U, bn_s, bn_b, it, ic, oc)
            ref = (F.conv2d(data_nchw.float(), w_nchw.float(), padding=P) * bn_s4 + bn_b4).half()
            ref_nhwc = ref.permute(0, 2, 3, 1)
            cos = F.cosine_similarity(out_wino.float().flatten(), ref_nhwc.float().flatten(), dim=0).item()
        else:
            cos = 0.0

        winner = min([('Winograd', ms_wino), ('im2col', ms_imcol), ('cuDNN', ms_cudnn)], key=lambda x: x[1])
        print(f"  → Winner: {winner[0]} | cos={cos:.4f}")

        results.append({'name': name, 'C_in': C_in, 'ms_wino': ms_wino,
                        'ms_imcol': ms_imcol, 'ms_cudnn': ms_cudnn, 'cos': cos})
        print()

    # Summary table
    print(f"\n{'='*85}")
    print(f"  SUMMARY: Implicit Winograd vs im2col vs cuDNN (3×3 s=1 Conv+BN)")
    print(f"  GPU: {props.name}")
    print(f"{'='*85}")
    print(f"  {'Shape':<28} {'Wino':>8} {'im2col':>8} {'cuDNN':>8} {'Wino/im2col':>12} {'Winner':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    for r in results:
        ratio = r['ms_wino'] / r['ms_imcol'] if r['ms_imcol'] > 0 and r['ms_wino'] < float('inf') else 999
        winner = min([('Wino', r['ms_wino']), ('im2col', r['ms_imcol']), ('cuDNN', r['ms_cudnn'])],
                     key=lambda x: x[1])
        print(f"  {r['name']:<28} {r['ms_wino']:>7.3f}ms {r['ms_imcol']:>7.3f}ms "
              f"{r['ms_cudnn']:>7.3f}ms {ratio:>10.2f}x {winner[0]:>10}")


if __name__ == '__main__':
    main()
