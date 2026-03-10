"""
Verify computation density: approximate vs exact (im2col) for convolution layers.

Compares the current approximation (mean(input!=0) * mean(weight!=0)) against
exact element-wise counting via im2col (torch.nn.functional.unfold).

For 1x1 convolutions: both methods are identical.
For 3x3+ convolutions with padding: exact method accounts for:
  - Padding zeros injected at input edges
  - Per-receptive-field non-zero counting (no independence assumption)

Usage:
    CUDA_VISIBLE_DEVICES=5 uv run vis/verify_density.py \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Verify density computation')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None)
    group.add_argument('--config', type=str, default=None)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--sample-idx', type=int, default=0)
    return parser.parse_args()


def compute_exact_conv_density(input_tensor, module):
    """Compute exact convolution density using im2col (unfold).

    For each output position, unfold the input receptive field and count
    element-wise where BOTH the unfolded input AND the weight are non-zero.

    Args:
        input_tensor: (TB, C_in, H, W) — the actual input to the conv layer
        module: nn.Conv2d

    Returns:
        exact_density: float — fraction of non-zero MACs
        spatial_density: (H_out, W_out) — per-output-position density
        total_ops: int
        effective_ops: int
    """
    x = input_tensor.detach()
    weight = module.weight.detach()
    TB = x.shape[0]

    kH, kW = module.kernel_size
    padding = module.padding
    stride = module.stride
    C_in = module.in_channels
    C_out = module.out_channels

    # Unfold input: (TB, C_in*kH*kW, L) where L = H_out * W_out
    x_unfolded = F.unfold(x, kernel_size=(kH, kW), padding=padding, stride=stride)
    # x_unfolded shape: (TB, C_in*kH*kW, L)

    L = x_unfolded.shape[2]
    H_out = (x.shape[2] + 2 * padding[0] - kH) // stride[0] + 1
    W_out = (x.shape[3] + 2 * padding[1] - kW) // stride[1] + 1

    # Weight reshaped: (C_out, C_in*kH*kW)
    w_flat = weight.view(C_out, -1)  # (C_out, C_in*kH*kW)

    # For each output position l and each output channel c:
    #   MAC[c, l] = sum_k(x_unfolded[b, k, l] * w_flat[c, k])
    #   effective = sum_k((x_unfolded[b,k,l] != 0) AND (w_flat[c,k] != 0))
    #   total = C_in * kH * kW

    x_nz = (x_unfolded != 0)  # (TB, K, L) where K = C_in*kH*kW
    w_nz = (w_flat != 0)      # (C_out, K)

    K = C_in * kH * kW
    total_per_output = K  # total MACs per output element

    # Count effective ops per output position (summed over C_out and TB)
    # For each (b, c, l): effective = sum_k(x_nz[b,k,l] AND w_nz[c,k])
    # = x_nz[b,:,l] dot w_nz[c,:]
    # Efficient: x_nz.float().T @ w_nz.float().T → per (l, c)
    # x_nz: (TB, K, L), w_nz: (C_out, K)
    # For each batch: (K, L).T @ (K, C_out) = (L, C_out) → effective per (l, c)

    total_effective = 0
    spatial_effective = np.zeros(L, dtype=np.float64)

    for b in range(TB):
        x_nz_b = x_nz[b].float()  # (K, L)
        # joint_nz[l, c] = sum_k x_nz[k,l] * w_nz[c,k] = x_nz_b.T @ w_nz.T
        joint = x_nz_b.T @ w_nz.float().T  # (L, C_out)
        total_effective += joint.sum().item()
        spatial_effective += joint.sum(dim=1).cpu().numpy()  # sum over C_out → (L,)

    total_ops = TB * C_out * L * K
    exact_density = total_effective / total_ops if total_ops > 0 else 0.0

    # Spatial density: normalize per position
    spatial_total_per_pos = TB * C_out * K
    spatial_density = spatial_effective / spatial_total_per_pos
    spatial_density = spatial_density.reshape(H_out, W_out)

    return exact_density, spatial_density, total_ops, int(total_effective)


def compute_approx_conv_density(input_tensor, module):
    """Current approximate density: mean(input!=0) * mean(weight!=0)."""
    x = input_tensor.detach()
    weight = module.weight.detach()

    input_density = (x != 0).float().mean().item()
    weight_density = (weight != 0).float().mean().item()
    approx_density = input_density * weight_density

    kH, kW = module.kernel_size
    C_in, C_out = module.in_channels, module.out_channels
    H_out, W_out = 0, 0
    # Compute output size (need to run the conv or calculate)
    with torch.no_grad():
        out = module(x)
        H_out, W_out = out.shape[2], out.shape[3]

    total_ops = x.shape[0] * C_out * H_out * W_out * C_in * kH * kW
    effective_ops = int(total_ops * approx_density)

    return approx_density, total_ops, effective_ops


def main():
    args = parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(42)

    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, batch_size=1,
        img_size=img_size, num_workers=0,
    )

    if args.config:
        model_cfg = load_model_config(args.config)
        model_cfg.update({
            'num_classes': num_classes, 'T': args.T,
            'img_size': img_size, 'in_channels': in_channels,
        })
        model = build_model_from_config(model_cfg)
        model_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
            model_kwargs['connect_f'] = 'ADD'
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, **model_kwargs)
        model_name = args.model

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get sample
    image, label = val_loader.dataset[args.sample_idx]
    image_tensor = image.unsqueeze(0).to(device)

    # Register hooks to capture inputs to every Conv2d
    captured_inputs = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            captured_inputs[name] = inp[0].detach()
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
        reset_net(model)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compare densities
    print(f"\n{'='*100}")
    print(f"  Density Verification: Approximate vs Exact (im2col) — {model_name}")
    print(f"  Sample idx: {args.sample_idx}  |  Label: {label}")
    print(f"{'='*100}")
    print(f"  {'Layer':<40} {'Kernel':>6} {'Approx':>10} {'Exact':>10} {'Diff':>10} {'RelErr':>10}")
    print(f"  {'-'*96}")

    total_approx_eff = 0
    total_exact_eff = 0
    total_ops_sum = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in captured_inputs:
            x_in = captured_inputs[name]
            kH, kW = module.kernel_size

            approx_d, approx_total, approx_eff = compute_approx_conv_density(x_in, module)

            if isinstance(module.kernel_size, tuple):
                kernel_str = f"{kH}x{kW}"
            else:
                kernel_str = f"{kH}x{kH}"

            # Only compute exact for Conv2d (skip Conv1d)
            exact_d, spatial, exact_total, exact_eff = compute_exact_conv_density(x_in, module)

            diff = (approx_d - exact_d) * 100
            rel_err = abs(approx_d - exact_d) / max(exact_d, 1e-10) * 100

            total_approx_eff += approx_eff
            total_exact_eff += exact_eff
            total_ops_sum += exact_total

            short_name = name[:39]
            print(f"  {short_name:<40} {kernel_str:>6} "
                  f"{approx_d*100:>9.4f}% {exact_d*100:>9.4f}% "
                  f"{diff:>+9.4f}% {rel_err:>9.2f}%")

    print(f"  {'-'*96}")
    overall_approx = total_approx_eff / max(total_ops_sum, 1) * 100
    overall_exact = total_exact_eff / max(total_ops_sum, 1) * 100
    print(f"  {'OVERALL':<40} {'':>6} "
          f"{overall_approx:>9.4f}% {overall_exact:>9.4f}% "
          f"{overall_approx - overall_exact:>+9.4f}% "
          f"{abs(overall_approx - overall_exact) / max(overall_exact, 1e-10) * 100:>9.2f}%")
    print(f"{'='*100}")

    print(f"\nSummary:")
    print(f"  1x1 convs: Approximate == Exact (no padding, no overlap)")
    print(f"  3x3 convs with padding: Approximate slightly overestimates density")
    print(f"    (because it doesn't account for zero-padding at edges)")
    print(f"  Overall discrepancy: {abs(overall_approx - overall_exact):.4f}%")


if __name__ == '__main__':
    main()
