"""Profile individual Conv+LIF and Linear+LIF pairs with T-B UN-MERGED.

Instead of TB-merge (process T*B at once), loops over T timesteps so each
compute kernel only sees batch=B. This reveals the per-timestep alternation
between compute-bound and memory-bound kernels.

Pairs selected from Spikformer-1-512 on ImageNet (B=8 ncu profile):
  Conv+LIF : SPS proj_conv3 — Conv2d(256,512,3,pad=1) at 112x112  (3.5ms in TRT)
  Linear+LIF: MLP fc1       — Linear(512,2048) on 3136 tokens     (0.9ms in TRT)

Usage:
    # Profile Conv+LIF pair
    sudo ncu --set basic --csv python motivation/profile_pair.py --pair conv > out.csv

    # Profile Linear+LIF pair
    sudo ncu --set basic --csv python motivation/profile_pair.py --pair linear > out.csv
"""

import argparse
import torch
import torch.nn as nn

T = 4
B = 8
TAU = 2.0
V_TH = 1.0


def lif_step(x, v):
    """Single-timestep LIF neuron. Returns (spike, new_v)."""
    v = v * (1.0 - 1.0 / TAU) + x * (1.0 / TAU)
    spike = (v >= V_TH).to(x.dtype)
    v = v * (1.0 - spike)
    return spike, v


def profile_conv_lif():
    """Conv2d(256,512,3,pad=1) at 112x112 + LIF, T-B un-merged."""
    conv = nn.Conv2d(256, 512, 3, padding=1, bias=False).cuda().half()
    # Input: (T, B, 256, 112, 112)
    x = torch.randn(T, B, 256, 112, 112, device='cuda', dtype=torch.float16)
    v = torch.zeros(B, 512, 112, 112, device='cuda', dtype=torch.float16)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            for t in range(T):
                out = conv(x[t])
                _, v = lif_step(out, v)
            v.zero_()
    torch.cuda.synchronize()

    # Profiled region: one full T-loop
    v.zero_()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for t in range(T):
            out = conv(x[t])         # Compute-bound
            spike, v = lif_step(out, v)  # Memory-bound
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Conv+LIF pair profiled (T-B un-merged)")


def profile_linear_lif():
    """Linear(512,2048) on 3136 tokens + LIF, T-B un-merged."""
    linear = nn.Linear(512, 2048, bias=False).cuda().half()
    N_TOKENS = 3136  # 56x56 patches from ImageNet 224 / 4
    # Input: (T, B, N_TOKENS, 512)
    x = torch.randn(T, B, N_TOKENS, 512, device='cuda', dtype=torch.float16)
    v = torch.zeros(B, N_TOKENS, 2048, device='cuda', dtype=torch.float16)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            for t in range(T):
                out = linear(x[t].reshape(B * N_TOKENS, 512))
                out = out.reshape(B, N_TOKENS, 2048)
                _, v = lif_step(out, v)
            v.zero_()
    torch.cuda.synchronize()

    # Profiled region
    v.zero_()
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for t in range(T):
            out = linear(x[t].reshape(B * N_TOKENS, 512))  # Compute-bound
            out = out.reshape(B, N_TOKENS, 2048)
            spike, v = lif_step(out, v)  # Memory-bound
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Linear+LIF pair profiled (T-B un-merged)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', choices=['conv', 'linear'], required=True)
    args = parser.parse_args()

    if args.pair == 'conv':
        profile_conv_lif()
    else:
        profile_linear_lif()


if __name__ == '__main__':
    main()
