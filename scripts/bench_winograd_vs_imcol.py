#!/usr/bin/env python3
"""Smoke test: im2col TileLang vs Winograd vs cuDNN for 3×3 stride=1 convolutions.

Compares across a range of (C, H, W) shapes from SNN models to identify
where Winograd outperforms im2col and by how much.

For each shape:
  1. TileLang im2col Conv+BN (current sengine kernel, autotuned)
  2. Winograd transform-separate (vectorized transforms + torch.bmm 16 GEMMs)
  3. cuDNN reference (torch.nn.functional.conv2d, auto-selects best algo)

Usage:
    CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_winograd_vs_imcol.py
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
import numpy as np

from sengine.kernels.conv2d_bn_if_t4 import conv2d_bn_t4_kernel
from sengine.tuning.roofline import select_config_roofline
import tilelang.language as T


# ═══════════════════════════════════════════════════════════════
# Winograd F(2×2, 3×3) — vectorized transforms + batched GEMM
# ═══════════════════════════════════════════════════════════════

_G = torch.tensor([[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]], dtype=torch.float32)

def wino_transform_weight(w_nchw, device):
    """(C_out, C_in, 3, 3) → (16, C_in, C_out) fp16"""
    G = _G.to(device)
    u = G @ w_nchw.float() @ G.t()
    return u.reshape(u.shape[0], u.shape[1], 16).permute(2, 1, 0).contiguous().half()


def wino_forward(data_nchw, U, P, tile_h, tile_w, C_in, C_out, TB, OH, OW):
    """Full Winograd F(2×2,3×3) forward: transform → 16 GEMMs → inverse transform."""
    BT = torch.tensor([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],
                       dtype=torch.float32, device=data_nchw.device)
    B = BT.t()
    AT = torch.tensor([[1,1,1,0],[0,1,-1,-1]], dtype=torch.float32, device=data_nchw.device)

    # Input transform (vectorized)
    x = F.pad(data_nchw.float(), (P, P, P, P))
    patches = x.unfold(2, 4, 2).unfold(3, 4, 2)
    # Actual tile counts from unfold (may differ from ceil calculation for small spatial)
    actual_th = patches.shape[2]
    actual_tw = patches.shape[3]
    tile_h = min(tile_h, actual_th)
    tile_w = min(tile_w, actual_tw)
    patches = patches[:, :, :tile_h, :tile_w]
    M = TB * tile_h * tile_w
    d0 = patches[:,:,:,:,0,:]; d1 = patches[:,:,:,:,1,:]
    d2 = patches[:,:,:,:,2,:]; d3 = patches[:,:,:,:,3,:]
    r0 = d0 - d2; r1 = d1 + d2; r2 = d2 - d1; r3 = d1 - d3
    results = []
    for rv in [r0, r1, r2, r3]:
        c0, c1, c2, c3 = rv[...,0], rv[...,1], rv[...,2], rv[...,3]
        results.extend([c0 - c2, c1 + c2, c2 - c1, c1 - c3])
    V = torch.stack(results, dim=0).permute(0, 1, 3, 4, 2).reshape(16, M, C_in).half().contiguous()

    # Batched GEMM
    O = torch.bmm(V, U)

    # Output transform (vectorized)
    m = O.float().reshape(4, 4, M, C_out)
    ar0 = m[0] + m[1] + m[2]
    ar1 = m[1] - m[2] - m[3]
    o00 = (ar0[0] + ar0[1] + ar0[2]).half()
    o01 = (ar0[1] - ar0[2] - ar0[3]).half()
    o10 = (ar1[0] + ar1[1] + ar1[2]).half()
    o11 = (ar1[1] - ar1[2] - ar1[3]).half()

    out = torch.empty(TB, C_out, OH, OW, device=data_nchw.device, dtype=torch.float16)
    out[:, :, 0::2, 0::2] = o00.reshape(TB, tile_h, tile_w, C_out).permute(0, 3, 1, 2)
    out[:, :, 0::2, 1::2] = o01.reshape(TB, tile_h, tile_w, C_out).permute(0, 3, 1, 2)
    out[:, :, 1::2, 0::2] = o10.reshape(TB, tile_h, tile_w, C_out).permute(0, 3, 1, 2)
    out[:, :, 1::2, 1::2] = o11.reshape(TB, tile_h, tile_w, C_out).permute(0, 3, 1, 2)
    return out


def benchmark(fn, warmup=100, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# ═══════════════════════════════════════════════════════════════
# Test shapes: 3×3 stride=1 convolutions from SNN models
# ═══════════════════════════════════════════════════════════════

SHAPES = [
    # (TB, C_in, C_out, H, W, name)
    # SEW-ResNet / MS-ResNet
    (4,  64,   64,  56, 56, "ResNet: 64→64 56×56"),
    (4,  128, 128,  28, 28, "ResNet: 128→128 28×28"),
    (4,  256, 256,  14, 14, "ResNet: 256→256 14×14"),
    (4,  512, 512,   7,  7, "ResNet: 512→512 7×7"),
    # SpikingResFormer-M (standard conv, not grouped)
    (16, 256, 256,  56, 56, "SRF-M: 256→256 56×56"),
    (16, 1536, 1536, 28, 28, "SRF-M: 1536→1536 28×28"),
    (16, 3072, 3072, 14, 14, "SRF-M: 3072→3072 14×14"),
    # SpikingResFormer-L
    (16, 2048, 2048, 28, 28, "SRF-L: 2048→2048 28×28"),
    (16, 4096, 4096, 14, 14, "SRF-L: 4096→4096 14×14"),
    # Small channels (where im2col should win)
    (4,  32,   32,  112, 112, "Small: 32→32 112×112"),
    (4,  64,   64,  112, 112, "Small: 64→64 112×112"),
]


def main():
    device_id = torch.cuda.current_device()
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device_id)
    props = torch.cuda.get_device_properties(device_id)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)")
    print(f"Comparing im2col TileLang vs Winograd vs cuDNN for 3×3 stride=1 Conv+BN")
    print()

    # Results table
    results = []
    P = 1  # padding=1 for all 3×3 stride=1

    for TB, C_in, C_out, H, W, name in SHAPES:
        OH, OW = H, W  # stride=1, pad=1
        K_red = 9 * C_in
        M = TB * OH * OW
        # Winograd tile count: actual tiles from unfold(size=4, step=2) on padded (H+2)
        tile_h = (H + 2 * P - 4) // 2 + 1  # = (H + 2P - 4) // 2 + 1
        tile_w = (W + 2 * P - 4) // 2 + 1
        M_tiles = TB * tile_h * tile_w

        print(f"{'='*70}")
        print(f"  {name} | TB={TB} K_red={K_red} M={M} M_tiles={M_tiles}")
        print(f"{'='*70}")

        # Tensors
        data_nhwc = torch.randn(TB, H, W, C_in, dtype=torch.float16, device=device)
        data_nchw = data_nhwc.permute(0, 3, 1, 2).contiguous()
        w_nchw = torch.randn(C_out, C_in, 3, 3, dtype=torch.float16, device=device)
        w_nhwc = w_nchw.permute(2, 3, 1, 0).contiguous()
        bn_s = torch.ones(C_out, dtype=torch.float32, device=device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=device)

        # ── 1. TileLang im2col (autotuned) ──
        def _compile_imcol(cfg):
            return conv2d_bn_t4_kernel(
                TB=TB, C_in=C_in, H=H, W=W, F=C_out, K=3, S=1, D=1, P=P,
                io_dtype=T.float16,
                **{k: cfg[k] for k in ('block_M', 'block_N', 'block_K', 'num_stages', 'threads')})

        profile_args = (data_nhwc, w_nhwc, bn_s, bn_b)
        try:
            torch.cuda.set_device(device_id)
            cfg = select_config_roofline(
                M_per_t=M, K=K_red, N=C_out, T_steps=1,
                compile_fn=_compile_imcol, profile_args=profile_args,
                top_k=5, n_profile=200, bpe=2)
            torch.cuda.set_device(device_id)
            kern_imcol = _compile_imcol(cfg)
            ms_imcol = benchmark(lambda: kern_imcol(data_nhwc, w_nhwc, bn_s, bn_b))
        except Exception as e:
            print(f"  im2col FAILED: {e}")
            ms_imcol = float('inf')
            cfg = {}
        print(f"  im2col TileLang:   {ms_imcol:.3f} ms (bK={cfg.get('block_K','?')})")

        # ── 2. Winograd transform-separate ──
        U = wino_transform_weight(w_nchw, device)
        try:
            ms_wino = benchmark(lambda: wino_forward(
                data_nchw, U, P, tile_h, tile_w, C_in, C_out, TB, OH, OW))
        except Exception as e:
            print(f"  Winograd FAILED: {e}")
            ms_wino = float('inf')
        print(f"  Winograd (xform+bmm): {ms_wino:.3f} ms")

        # ── 2b. Winograd GEMM only (ceiling for fused kernel) ──
        try:
            V_pre = torch.randn(16, M_tiles, C_in, dtype=torch.float16, device=device).contiguous()
            U_local = U.contiguous()
            ms_gemm = benchmark(lambda: torch.bmm(V_pre, U_local))
        except Exception as e:
            ms_gemm = float('inf')
        print(f"  Winograd GEMM only:   {ms_gemm:.3f} ms (16× {M_tiles}×{C_in}×{C_out})")

        # ── 3. cuDNN reference ──
        bn_s4 = bn_s.view(1, -1, 1, 1)
        bn_b4 = bn_b.view(1, -1, 1, 1)
        ms_cudnn = benchmark(lambda: F.conv2d(data_nchw, w_nchw, padding=P) * bn_s4 + bn_b4)
        print(f"  cuDNN:                {ms_cudnn:.3f} ms")

        # ── Correctness ──
        try:
            out_wino = wino_forward(data_nchw, U, P, tile_h, tile_w, C_in, C_out, TB, OH, OW)
            ref = (F.conv2d(data_nchw.float(), w_nchw.float(), padding=P) * bn_s4 + bn_b4).half()
            cos = F.cosine_similarity(out_wino.float().flatten(), ref.float().flatten(), dim=0)
        except Exception:
            cos = torch.tensor(0.0)

        # ── Summary ──
        best = min(ms_imcol, ms_wino, ms_cudnn)
        imcol_ratio = ms_imcol / ms_cudnn if ms_cudnn > 0 else 0
        wino_ratio = ms_wino / ms_cudnn if ms_cudnn > 0 else 0
        winner = "im2col" if ms_imcol <= ms_wino else "WINOGRAD"
        print(f"  → im2col/cuDNN: {imcol_ratio:.2f}x | wino/cuDNN: {wino_ratio:.2f}x | "
              f"winner: {winner} | cos={cos.item():.4f}")

        results.append({
            'name': name, 'C_in': C_in, 'H': H, 'TB': TB,
            'K_red': K_red,
            'ms_imcol': ms_imcol, 'ms_wino': ms_wino,
            'ms_gemm': ms_gemm, 'ms_cudnn': ms_cudnn,
        })
        print()

    # ═══════════════════════════════════════════════════════════
    # Final comparison table
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(f"  SUMMARY: 3×3 stride=1 Conv+BN — im2col vs Winograd vs cuDNN")
    print(f"  GPU: {props.name}")
    print(f"{'='*90}")
    print(f"  {'Shape':<30} {'K_red':>6} {'im2col':>8} {'Wino':>8} {'GEMM':>8} {'cuDNN':>8} {'Winner':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in results:
        winner = "im2col" if r['ms_imcol'] <= r['ms_wino'] else "WINOGRAD"
        if r['ms_cudnn'] < min(r['ms_imcol'], r['ms_wino']):
            winner = "cuDNN"
        print(f"  {r['name']:<30} {r['K_red']:>6} {r['ms_imcol']:>7.3f}ms "
              f"{r['ms_wino']:>7.3f}ms {r['ms_gemm']:>7.3f}ms {r['ms_cudnn']:>7.3f}ms "
              f"{winner:>10}")

    print(f"\n  Note: 'Wino' = full transform-separate (transforms + 16 GEMMs)")
    print(f"  Note: 'GEMM' = Winograd GEMM only (ceiling for fused kernel)")
    print(f"  Note: cuDNN auto-selects Winograd internally for large C_in shapes")


if __name__ == '__main__':
    main()
