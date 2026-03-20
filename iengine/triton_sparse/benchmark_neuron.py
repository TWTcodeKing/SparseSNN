"""Benchmark fused Triton neuron kernels vs PyTorch baseline.

Tests correctness (output match) and latency for both LIF and IF neurons
across various tensor sizes and timesteps.

Usage:
    python -m iengine.triton_sparse.benchmark_neuron [--T 4] [--gpu-id 0]
"""

import argparse
import time
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.neurons import (
    MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron
)
from iengine.triton_sparse.neuron_kernel import (
    fused_lif_forward, fused_if_forward, replace_neuron_forward
)


def benchmark_kernel(fn, *args, warmup=20, repeat=100, **kwargs):
    """Benchmark a function with CUDA synchronization."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeat * 1000  # ms


def test_correctness(T=4, device='cuda'):
    """Verify fused Triton output matches PyTorch baseline."""
    print(f"\n{'='*60}")
    print(f"  Correctness Test (T={T})")
    print(f"{'='*60}")

    shapes = [
        (T, 8, 64),          # small: (T, B, C)
        (T, 16, 128, 8, 8),  # conv: (T, B, C, H, W)
        (T, 32, 384),        # transformer: (T, B, D)
    ]

    all_passed = True
    for shape in shapes:
        x = torch.randn(shape, device=device)

        # --- LIF ---
        lif = MultiStepLIFNeuron(tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate='atan')
        lif.neuron.reset()
        with torch.no_grad():
            spikes_ref = lif(x)

        spikes_triton = fused_lif_forward(x, tau=2.0, v_threshold=1.0, v_reset=0.0)

        match = torch.allclose(spikes_ref, spikes_triton)
        max_diff = (spikes_ref - spikes_triton).abs().max().item()
        status = 'PASS' if match else 'FAIL'
        if not match:
            all_passed = False
        print(f"  LIF {str(shape):<28} {status}  max_diff={max_diff:.6f}")

        # --- LIF soft reset ---
        lif_soft = MultiStepLIFNeuron(tau=2.0, v_threshold=1.0, v_reset=None, surrogate='atan')
        lif_soft.neuron.v_reset = None
        # Need to manually create a LIF with soft reset
        lif_soft_manual = LIFNeuron(tau=2.0, v_threshold=1.0, v_reset=None, surrogate='atan')
        lif_soft_ms = MultiStepIFNeuron.__new__(MultiStepLIFNeuron)
        nn.Module.__init__(lif_soft_ms)
        lif_soft_ms.neuron = lif_soft_manual
        lif_soft_ms.neuron.reset()
        with torch.no_grad():
            spikes_ref_soft = lif_soft_ms(x)

        spikes_triton_soft = fused_lif_forward(x, tau=2.0, v_threshold=1.0, v_reset=None)

        match_s = torch.allclose(spikes_ref_soft, spikes_triton_soft)
        max_diff_s = (spikes_ref_soft - spikes_triton_soft).abs().max().item()
        status_s = 'PASS' if match_s else 'FAIL'
        if not match_s:
            all_passed = False
        print(f"  LIF(soft) {str(shape):<23} {status_s}  max_diff={max_diff_s:.6f}")

        # --- IF ---
        ifn = MultiStepIFNeuron(v_threshold=1.0, v_reset=0.0, surrogate='atan')
        ifn.neuron.reset()
        with torch.no_grad():
            spikes_if_ref = ifn(x)

        spikes_if_triton = fused_if_forward(x, v_threshold=1.0, v_reset=0.0)

        match_if = torch.allclose(spikes_if_ref, spikes_if_triton)
        max_diff_if = (spikes_if_ref - spikes_if_triton).abs().max().item()
        status_if = 'PASS' if match_if else 'FAIL'
        if not match_if:
            all_passed = False
        print(f"  IF  {str(shape):<28} {status_if}  max_diff={max_diff_if:.6f}")

    print(f"\n  All tests: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def benchmark_latency(T=4, device='cuda'):
    """Benchmark fused Triton vs PyTorch loop for various shapes."""
    print(f"\n{'='*60}")
    print(f"  Latency Benchmark (T={T})")
    print(f"{'='*60}")

    configs = [
        # (name, shape)
        ("CIFAR Conv",     (T, 32, 128, 8, 8)),
        ("CIFAR Linear",   (T, 32, 384)),
        ("ImageNet Conv",  (T, 16, 256, 14, 14)),
        ("ImageNet Linear",(T, 16, 768)),
        ("Large Conv",     (T, 64, 512, 7, 7)),
        ("Large Linear",   (T, 64, 768)),
    ]

    print(f"\n  {'Config':<20} {'Shape':<28} {'PyTorch (ms)':>14} {'Triton (ms)':>14} {'Speedup':>10}")
    print("  " + "-" * 90)

    for cfg_name, shape in configs:
        x = torch.randn(shape, device=device)

        # PyTorch baseline (MultiStepLIFNeuron with Python loop)
        lif = MultiStepLIFNeuron(tau=2.0, v_threshold=1.0, v_reset=0.0)

        def pytorch_fn():
            lif.neuron.reset()
            return lif(x)

        def triton_fn():
            return fused_lif_forward(x, tau=2.0, v_threshold=1.0, v_reset=0.0)

        ms_pytorch = benchmark_kernel(pytorch_fn, warmup=10, repeat=50)
        ms_triton = benchmark_kernel(triton_fn, warmup=10, repeat=50)
        speedup = ms_pytorch / max(ms_triton, 1e-6)

        N = x[0].numel()
        print(f"  {cfg_name:<20} {str(shape):<28} {ms_pytorch:>14.4f} {ms_triton:>14.4f} {speedup:>9.2f}x")


def benchmark_model_replacement(T=4, device='cuda'):
    """Benchmark end-to-end model with neuron replacement."""
    print(f"\n{'='*60}")
    print(f"  Model-Level Neuron Replacement Benchmark")
    print(f"{'='*60}")

    from tengine.utils import build_model, get_dataset_config

    models_to_test = [
        ('ms_resnet18', 'cifar100', T),
    ]

    # Try to also test spikformer if config exists
    try:
        from tengine.utils import load_model_config, build_model_from_config
        config = load_model_config('configs/spikformer/spikformer_cifar.yaml')
        ds_cfg = get_dataset_config('cifar100')
        config.update(ds_cfg)
        config['T'] = T
        has_spikformer = True
    except Exception:
        has_spikformer = False

    for model_name, dataset, t in models_to_test:
        ds_cfg = get_dataset_config(dataset)
        model = build_model(model_name, num_classes=ds_cfg['num_classes'],
                           in_channels=ds_cfg['in_channels'], T=t)
        model = model.to(device).eval()

        # Create dummy input
        img_size = ds_cfg['img_size']
        x = torch.randn(1, ds_cfg['in_channels'], img_size, img_size, device=device)

        # Baseline
        from models import reset_net
        def run_baseline():
            with torch.no_grad():
                out = model(x)
            reset_net(model)
            return out

        ms_baseline = benchmark_kernel(run_baseline, warmup=5, repeat=30)

        # Get baseline output for correctness check
        out_baseline = run_baseline()

        # Replace neurons
        print(f"\n  Model: {model_name}")
        n_replaced = replace_neuron_forward(model)

        def run_triton():
            with torch.no_grad():
                out = model(x)
            reset_net(model)
            return out

        ms_triton = benchmark_kernel(run_triton, warmup=5, repeat=30)

        # Correctness
        out_triton = run_triton()
        match = torch.allclose(out_baseline.float(), out_triton.float(), atol=1e-4, rtol=1e-3)

        speedup = ms_baseline / max(ms_triton, 1e-6)
        print(f"  Baseline:  {ms_baseline:.4f} ms")
        print(f"  Triton:    {ms_triton:.4f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print(f"  Output match: {'PASS' if match else 'FAIL'}")

    if has_spikformer:
        model_spk = build_model_from_config(config).to(device).eval()
        x_spk = torch.randn(1, ds_cfg['in_channels'], img_size, img_size, device=device)

        def run_spk_baseline():
            with torch.no_grad():
                out = model_spk(x_spk)
            reset_net(model_spk)
            return out

        ms_spk_base = benchmark_kernel(run_spk_baseline, warmup=5, repeat=30)
        out_spk_base = run_spk_baseline()

        print(f"\n  Model: spikformer")
        n_replaced = replace_neuron_forward(model_spk)

        def run_spk_triton():
            with torch.no_grad():
                out = model_spk(x_spk)
            reset_net(model_spk)
            return out

        ms_spk_triton = benchmark_kernel(run_spk_triton, warmup=5, repeat=30)
        out_spk_triton = run_spk_triton()
        match_spk = torch.allclose(out_spk_base.float(), out_spk_triton.float(), atol=1e-4, rtol=1e-3)

        speedup_spk = ms_spk_base / max(ms_spk_triton, 1e-6)
        print(f"  Baseline:  {ms_spk_base:.4f} ms")
        print(f"  Triton:    {ms_spk_triton:.4f} ms")
        print(f"  Speedup:   {speedup_spk:.2f}x")
        print(f"  Output match: {'PASS' if match_spk else 'FAIL'}")


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark fused Triton neuron kernels')
    parser.add_argument('--T', type=int, default=4, help='Number of timesteps')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--skip-model', action='store_true', help='Skip model-level benchmark')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # 1. Correctness
    passed = test_correctness(T=args.T, device=device)
    if not passed:
        print("\n  WARNING: Correctness test failed! Check kernel implementation.")

    # 2. Kernel-level latency
    benchmark_latency(T=args.T, device=device)

    # 3. Model-level (end-to-end)
    if not args.skip_model:
        benchmark_model_replacement(T=args.T, device=device)


if __name__ == '__main__':
    main()
