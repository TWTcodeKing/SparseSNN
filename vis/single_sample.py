"""
Single-sample SNN computation density visualization.

Loads a trained SNN model, runs a single image through it, and produces:
1. Per-layer computation density bar chart (conv, linear, attention)
2. Attention Q/K/V sparsity breakdown (per-block)
3. Spatial density heatmap overlaid on original image (GradCAM-style)
4. Numerical statistics table

Usage:
    # Spikformer
    uv run vis/single_sample.py \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0 --sample-idx 0

    # SEW-ResNet
    uv run vis/single_sample.py \
        --model sew_resnet34 --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/sew_resnet34_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0 --sample-idx 42
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)
from vis.density_hooks import DensityTracker, compute_overall_spatial_density


def parse_args():
    parser = argparse.ArgumentParser(description='Single-sample SNN density visualization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None)
    group.add_argument('--config', type=str, default=None)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Index of the sample in the validation set')
    parser.add_argument('--save-dir', type=str, default='./vis_output',
                        help='Directory to save output figures')
    return parser.parse_args()


def load_model_and_data(args):
    """Load the model checkpoint and get a single sample from validation set."""
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(args.seed)

    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = args.img_size or ds_cfg['img_size']
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

    # Get the specific sample
    val_dataset = val_loader.dataset
    image, label = val_dataset[args.sample_idx]
    image_tensor = image.unsqueeze(0).to(device)  # (1, C, H, W)

    return model, model_name, image_tensor, label, img_size, device, val_loader


def denormalize_image(img_tensor, dataset='cifar100'):
    """Denormalize image tensor back to [0, 1] for display."""
    # CIFAR mean/std
    if dataset in ('cifar10', 'cifar100'):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    elif dataset == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    else:
        return img_tensor.cpu().clamp(0, 1)

    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1)


def plot_layer_density_bar(records, save_path, model_name):
    """Bar chart of per-layer computation density."""
    names = []
    densities = []
    colors = []
    color_map = {'conv': '#4C72B0', 'linear': '#DD8452', 'attn_kv': '#55A868', 'attn_qr': '#C44E52'}

    for name, rec in records.items():
        if not rec.densities:
            continue
        # Shorten names for readability
        short_name = name.replace('patch_embed.', 'PE/')
        short_name = short_name.replace('block.', 'B')
        short_name = short_name.replace('.attn.', '/A/')
        short_name = short_name.replace('.mlp.', '/M/')
        names.append(short_name)
        densities.append(rec.mean_density * 100)
        colors.append(color_map.get(rec.layer_type, '#999999'))

    if not names:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 6))
    x = np.arange(len(names))
    bars = ax.bar(x, densities, color=colors, width=0.7, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Computation Density (%)', fontsize=12)
    ax.set_title(f'Per-Layer Computation Density — {model_name}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7, ha='center')
    ax.set_ylim(0, 100)
    ax.axhline(y=np.mean(densities), color='red', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(densities):.1f}%')
    ax.legend(fontsize=10)

    # Legend for layer types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='Conv'),
        Patch(facecolor='#DD8452', label='Linear'),
        Patch(facecolor='#55A868', label=r'Attn $K^T@V$'),
        Patch(facecolor='#C44E52', label=r'Attn $Q@(K^T@V)$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attention_density(records, save_path, model_name):
    """Attention-specific density breakdown per block."""
    kv_names = []
    kv_densities = []
    qr_names = []
    qr_densities = []

    for name, rec in records.items():
        if not rec.densities:
            continue
        if rec.layer_type == 'attn_kv':
            block_id = name.split('.')[1] if '.' in name else name
            kv_names.append(block_id)
            kv_densities.append(rec.mean_density * 100)
        elif rec.layer_type == 'attn_qr':
            block_id = name.split('.')[1] if '.' in name else name
            qr_names.append(block_id)
            qr_densities.append(rec.mean_density * 100)

    if not kv_names:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # K^T @ V density
    x = np.arange(len(kv_names))
    axes[0].bar(x, kv_densities, color='#55A868', width=0.6)
    axes[0].set_xlabel('Block')
    axes[0].set_ylabel('Density (%)')
    axes[0].set_title(r'$K^T @ V$ Density (binary x binary)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kv_names)
    axes[0].set_ylim(0, max(max(kv_densities) * 1.3, 1))
    for i, v in enumerate(kv_densities):
        axes[0].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=9)

    # Q @ (K^T@V) density
    x = np.arange(len(qr_names))
    axes[1].bar(x, qr_densities, color='#C44E52', width=0.6)
    axes[1].set_xlabel('Block')
    axes[1].set_ylabel('Density (%)')
    axes[1].set_title(r'$Q @ (K^T@V)$ Density (binary x real)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(qr_names)
    axes[1].set_ylim(0, max(max(qr_densities) * 1.3, 1))
    for i, v in enumerate(qr_densities):
        axes[1].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=9)

    fig.suptitle(f'Attention Computation Density — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_spatial_density_overlay(tracker, image_tensor, img_size, save_path,
                                 model_name, dataset):
    """GradCAM-style spatial density overlay on the original image."""
    # Aggregate spatial density map
    overall_map = compute_overall_spatial_density(tracker, img_size)
    layer_maps = tracker.get_spatial_maps(target_size=(img_size, img_size))

    if overall_map is None and not layer_maps:
        print("  No spatial maps available for overlay.")
        return

    # Denormalize image for display
    img_display = denormalize_image(image_tensor.squeeze(0), dataset)
    img_np = img_display.permute(1, 2, 0).numpy()  # (H, W, C)

    # Select representative layer maps (first conv, middle, last, attention)
    selected_layers = {}
    conv_layers = [(n, m) for n, m in layer_maps.items()
                   if tracker.records[n].layer_type == 'conv']
    attn_layers = [(n, m) for n, m in layer_maps.items()
                   if tracker.records[n].layer_type in ('attn_kv', 'attn_qr')]

    if conv_layers:
        selected_layers['First Conv'] = conv_layers[0]
        if len(conv_layers) > 2:
            mid = len(conv_layers) // 2
            selected_layers['Mid Conv'] = conv_layers[mid]
        selected_layers['Last Conv'] = conv_layers[-1]
    if attn_layers:
        selected_layers['Attention'] = attn_layers[0]

    n_plots = 2 + len(selected_layers)  # original + overall + selected
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontsize=11)
    axes[0].axis('off')

    # Overall density overlay
    if overall_map is not None:
        axes[1].imshow(img_np)
        heatmap = axes[1].imshow(overall_map, cmap='jet', alpha=0.5,
                                  vmin=0, vmax=max(overall_map.max(), 0.01))
        axes[1].set_title('Overall Density', fontsize=11)
        axes[1].axis('off')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(heatmap, cax=cax)

    # Selected layer overlays
    for idx, (label, (layer_name, smap)) in enumerate(selected_layers.items()):
        ax = axes[2 + idx]
        ax.imshow(img_np)
        hm = ax.imshow(smap, cmap='jet', alpha=0.5,
                        vmin=0, vmax=max(smap.max(), 0.01))
        ax.set_title(f'{label}\n({layer_name[:30]})', fontsize=9)
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(hm, cax=cax)

    fig.suptitle(f'Spatial Computation Density — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_statistics(tracker, model_name, label, pred_class):
    """Print numerical density statistics table."""
    summary = tracker.get_summary()
    print(f"\n{'='*80}")
    print(f"  Computation Density Statistics — {model_name}")
    print(f"  True label: {label}  |  Predicted: {pred_class}")
    print(f"{'='*80}")
    print(f"  {'Layer':<45} {'Type':<10} {'Density':>8} {'Total Ops':>14} {'Eff. Ops':>14}")
    print(f"  {'-'*91}")

    total_all = 0
    eff_all = 0
    for name, info in summary.items():
        short = name[:44]
        d = info['mean_density'] * 100
        t = info['total_ops']
        e = info['effective_ops']
        total_all += t
        eff_all += e
        print(f"  {short:<45} {info['type']:<10} {d:>7.2f}% {t:>14,} {e:>14,}")

    if total_all > 0:
        overall = eff_all / total_all * 100
        print(f"  {'-'*91}")
        print(f"  {'OVERALL':<45} {'':10} {overall:>7.2f}% {total_all:>14,} {eff_all:>14,}")

    # Group by type
    type_stats = {}
    for name, info in summary.items():
        t = info['type']
        if t not in type_stats:
            type_stats[t] = {'densities': [], 'total': 0, 'effective': 0}
        type_stats[t]['densities'].append(info['mean_density'])
        type_stats[t]['total'] += info['total_ops']
        type_stats[t]['effective'] += info['effective_ops']

    print(f"\n  Summary by Layer Type:")
    print(f"  {'Type':<15} {'Avg Density':>12} {'Total Ops':>14} {'Eff. Ops':>14} {'Layers':>8}")
    print(f"  {'-'*63}")
    for t, stats in type_stats.items():
        avg_d = np.mean(stats['densities']) * 100
        print(f"  {t:<15} {avg_d:>11.2f}% {stats['total']:>14,} "
              f"{stats['effective']:>14,} {len(stats['densities']):>8}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading model and data...")
    model, model_name, image_tensor, label, img_size, device, val_loader = \
        load_model_and_data(args)

    print(f"Model: {model_name}  |  Sample idx: {args.sample_idx}  |  Label: {label}")

    # Setup density tracking
    tracker = DensityTracker(model, img_size=img_size)
    tracker.register_hooks()

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(image_tensor)
        reset_net(model)
    pred_class = output.argmax(dim=1).item()
    tracker.finalize_sample()

    # Print statistics
    print_statistics(tracker, model_name, label, pred_class)

    # Generate visualizations
    prefix = f"{model_name}_sample{args.sample_idx}"

    print("Generating visualizations...")
    plot_layer_density_bar(
        tracker.records,
        os.path.join(args.save_dir, f"{prefix}_layer_density.png"),
        model_name)

    plot_attention_density(
        tracker.records,
        os.path.join(args.save_dir, f"{prefix}_attention_density.png"),
        model_name)

    plot_spatial_density_overlay(
        tracker, image_tensor, img_size,
        os.path.join(args.save_dir, f"{prefix}_spatial_overlay.png"),
        model_name, args.dataset)

    tracker.remove_hooks()
    print("Done!")


if __name__ == '__main__':
    main()
