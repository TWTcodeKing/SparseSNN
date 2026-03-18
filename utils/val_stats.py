"""
Validation-set SNN computation density statistical analysis.

Runs the full validation set through a trained SNN model and produces:
1. Per-layer density distribution (violin/box plots)
2. Density distribution histograms
3. Average spatial density heatmap (aggregate over all samples)
4. Per-class density comparison
5. Spatial density vs image content correlation analysis

Usage:
    # Spikformer on CIFAR-100
    uv run vis/val_stats.py \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0 --max-samples 500

    # SEW-ResNet
    uv run vis/val_stats.py \
        --model sew_resnet34 --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/sew_resnet34_cifar100_bs128_lr0.001/best.pth \
        --gpu-ids 0
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F_torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from tengine.utils import (
    set_seed, build_model, build_model_from_config, load_model_config,
    build_dataloaders, get_dataset_config,
)
from utils.density_hooks import DensityTracker, compute_overall_spatial_density


def parse_args():
    parser = argparse.ArgumentParser(description='Validation-set SNN density analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, default=None)
    group.add_argument('--config', type=str, default=None)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (use 1 for per-sample analysis)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max samples to process (0 = all)')
    parser.add_argument('--save-dir', type=str, default='./vis_output')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def load_model_and_data(args):
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(args.seed)

    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = args.img_size or ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size,
        img_size=img_size, num_workers=args.workers,
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

    return model, model_name, val_loader, img_size, num_classes, device


def collect_density_stats(model, val_loader, img_size, device, max_samples=0):
    """Run validation set and collect per-sample density statistics.

    Returns:
        layer_densities: {layer_name: [density_per_sample, ...]}
        layer_types: {layer_name: layer_type}
        spatial_accum: {layer_name: accumulated spatial maps}
        spatial_counts: {layer_name: count}
        per_sample_overall: [(overall_density, label, pred), ...]
        per_class_densities: {class_id: [overall_density, ...]}
        overall_spatial_accum: accumulated overall spatial density (H, W)
        image_accum: accumulated image intensity per pixel (H, W)
    """
    tracker = DensityTracker(model, img_size=img_size)
    tracker.register_hooks()

    layer_densities = defaultdict(list)
    layer_types = {}
    spatial_accum = defaultdict(lambda: np.zeros((img_size, img_size)))
    spatial_counts = defaultdict(int)
    per_sample_overall = []
    per_class_densities = defaultdict(list)
    overall_spatial_accum = np.zeros((img_size, img_size))
    overall_spatial_count = 0
    # Track per-pixel image intensity to correlate with density
    image_intensity_accum = np.zeros((img_size, img_size))

    total = len(val_loader.dataset)
    if max_samples > 0:
        total = min(total, max_samples)

    processed = 0
    for step, (images, targets) in enumerate(val_loader):
        if max_samples > 0 and processed >= max_samples:
            break

        bs = images.shape[0]
        images = images.to(device, non_blocking=True)

        tracker.clear()

        with torch.no_grad():
            output = model(images)
            reset_net(model)

        preds = output.argmax(dim=1).cpu()

        # Collect per-layer densities
        for name, rec in tracker.records.items():
            if rec.densities:
                layer_densities[name].extend(rec.densities)
                layer_types[name] = rec.layer_type

        # Collect spatial maps
        maps = tracker.get_spatial_maps(target_size=(img_size, img_size))
        for name, smap in maps.items():
            spatial_accum[name] += smap
            spatial_counts[name] += 1

        # Overall spatial density
        overall_map = compute_overall_spatial_density(tracker, img_size)
        if overall_map is not None:
            overall_spatial_accum += overall_map
            overall_spatial_count += 1

        # Per-sample overall density (weighted by ops)
        summary = tracker.get_summary()
        total_ops = sum(v['total_ops'] for v in summary.values())
        eff_ops = sum(v['effective_ops'] for v in summary.values())
        overall_d = eff_ops / max(total_ops, 1)

        for i in range(bs):
            label_i = targets[i].item()
            pred_i = preds[i].item()
            per_sample_overall.append((overall_d, label_i, pred_i))
            per_class_densities[label_i].append(overall_d)

        # Accumulate image intensity (grayscale, normalized)
        img_gray = images.cpu().mean(dim=1)  # (B, H, W) - average over channels
        # Handle non-matching sizes
        if img_gray.shape[-2:] != (img_size, img_size):
            img_gray = F_torch.interpolate(
                img_gray.unsqueeze(1), size=(img_size, img_size),
                mode='bilinear', align_corners=False).squeeze(1)
        image_intensity_accum += img_gray.mean(dim=0).numpy()

        processed += bs
        if (step + 1) % 50 == 0:
            print(f"  Processed {processed}/{total} samples...")

    tracker.remove_hooks()

    # Average spatial maps
    for name in spatial_accum:
        if spatial_counts[name] > 0:
            spatial_accum[name] /= spatial_counts[name]
    if overall_spatial_count > 0:
        overall_spatial_accum /= overall_spatial_count
        image_intensity_accum /= overall_spatial_count

    print(f"  Processed {processed} samples total.")

    return (dict(layer_densities), layer_types, dict(spatial_accum),
            per_sample_overall, dict(per_class_densities),
            overall_spatial_accum, image_intensity_accum)


# ---- Plotting functions ----

def plot_density_distributions(layer_densities, layer_types, save_path, model_name):
    """Violin/box plot of per-layer density distributions."""
    # Group layers
    conv_data = []
    conv_names = []
    attn_data = []
    attn_names = []
    linear_data = []
    linear_names = []

    for name, densities in layer_densities.items():
        lt = layer_types[name]
        d = [x * 100 for x in densities]
        short = name.replace('patch_embed.', 'PE/').replace('block.', 'B')\
                     .replace('.attn.', '/A/').replace('.mlp.', '/M/')
        if lt == 'conv':
            conv_data.append(d)
            conv_names.append(short[:25])
        elif lt in ('attn_kv', 'attn_qr'):
            attn_data.append(d)
            attn_names.append(short[:25])
        elif lt == 'linear':
            linear_data.append(d)
            linear_names.append(short[:25])

    n_groups = sum(1 for x in [conv_data, attn_data, linear_data] if x)
    fig, axes = plt.subplots(n_groups, 1, figsize=(max(14, max(
        len(conv_names), len(attn_names), len(linear_names), 1) * 0.5), 5 * n_groups))
    if n_groups == 1:
        axes = [axes]

    ax_idx = 0
    if conv_data:
        vp = axes[ax_idx].violinplot(conv_data, showmeans=True, showmedians=True)
        for body in vp['bodies']:
            body.set_facecolor('#4C72B0')
            body.set_alpha(0.7)
        axes[ax_idx].set_xticks(range(1, len(conv_names) + 1))
        axes[ax_idx].set_xticklabels(conv_names, rotation=90, fontsize=7)
        axes[ax_idx].set_ylabel('Density (%)')
        axes[ax_idx].set_title('Conv Layer Density Distribution')
        axes[ax_idx].set_ylim(0, 100)
        ax_idx += 1

    if attn_data:
        vp = axes[ax_idx].violinplot(attn_data, showmeans=True, showmedians=True)
        for body in vp['bodies']:
            body.set_facecolor('#55A868')
            body.set_alpha(0.7)
        axes[ax_idx].set_xticks(range(1, len(attn_names) + 1))
        axes[ax_idx].set_xticklabels(attn_names, rotation=90, fontsize=7)
        axes[ax_idx].set_ylabel('Density (%)')
        axes[ax_idx].set_title('Attention Density Distribution')
        ax_idx += 1

    if linear_data:
        vp = axes[ax_idx].violinplot(linear_data, showmeans=True, showmedians=True)
        for body in vp['bodies']:
            body.set_facecolor('#DD8452')
            body.set_alpha(0.7)
        axes[ax_idx].set_xticks(range(1, len(linear_names) + 1))
        axes[ax_idx].set_xticklabels(linear_names, rotation=90, fontsize=7)
        axes[ax_idx].set_ylabel('Density (%)')
        axes[ax_idx].set_title('Linear Layer Density Distribution')
        ax_idx += 1

    fig.suptitle(f'Computation Density Distributions — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_overall_histogram(per_sample_overall, save_path, model_name):
    """Histogram of overall density across all samples."""
    densities = [x[0] * 100 for x in per_sample_overall]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(densities, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(densities), color='red', linestyle='--',
               label=f'Mean: {np.mean(densities):.2f}%')
    ax.axvline(np.median(densities), color='orange', linestyle='--',
               label=f'Median: {np.median(densities):.2f}%')
    ax.set_xlabel('Overall Computation Density (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Overall Density Distribution — {model_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_class_density(per_class_densities, save_path, model_name, top_k=20):
    """Bar chart of per-class mean density (top-K highest and lowest)."""
    class_means = {}
    for cls, densities in per_class_densities.items():
        if densities:
            class_means[cls] = np.mean(densities) * 100

    if not class_means:
        return

    sorted_classes = sorted(class_means.items(), key=lambda x: x[1])

    if len(sorted_classes) > 2 * top_k:
        selected = sorted_classes[:top_k] + sorted_classes[-top_k:]
        labels = [f"C{c}" for c, _ in selected]
        values = [v for _, v in selected]
        colors = ['#C44E52'] * top_k + ['#55A868'] * top_k
    else:
        labels = [f"C{c}" for c, _ in sorted_classes]
        values = [v for _, v in sorted_classes]
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(values)))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.3), 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('Mean Density (%)')
    ax.set_title(f'Per-Class Computation Density — {model_name}\n'
                 f'({"Top/Bottom" if len(sorted_classes) > 2*top_k else "All"} {top_k} classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_avg_spatial_density(overall_spatial, image_intensity, save_path, model_name):
    """Average spatial density heatmap + correlation with image intensity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Average density heatmap
    im0 = axes[0].imshow(overall_spatial, cmap='hot', interpolation='bilinear')
    axes[0].set_title('Avg Computation Density\n(spatial)', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Average image intensity
    im1 = axes[1].imshow(image_intensity, cmap='gray', interpolation='bilinear')
    axes[1].set_title('Avg Image Intensity\n(grayscale)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Overlay: density on intensity
    # Normalize density to [0, 1] for overlay
    d_norm = (overall_spatial - overall_spatial.min()) / \
             max(overall_spatial.max() - overall_spatial.min(), 1e-8)
    # Create RGB overlay
    cmap = plt.cm.jet
    density_rgb = cmap(d_norm)[:, :, :3]
    intensity_rgb = np.stack([image_intensity] * 3, axis=-1)
    intensity_rgb = (intensity_rgb - intensity_rgb.min()) / \
                    max(intensity_rgb.max() - intensity_rgb.min(), 1e-8)
    overlay = 0.5 * intensity_rgb + 0.5 * density_rgb
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title('Density on Image\n(overlay)', fontsize=11)
    axes[2].axis('off')

    fig.suptitle(f'Average Spatial Density Map — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_density_correlation(overall_spatial, image_intensity, save_path, model_name):
    """Scatter plot: pixel intensity vs computation density (spatial correlation)."""
    intensity_flat = image_intensity.flatten()
    density_flat = overall_spatial.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    axes[0].scatter(intensity_flat, density_flat * 100, alpha=0.5, s=10, c='#4C72B0')
    axes[0].set_xlabel('Avg Pixel Intensity')
    axes[0].set_ylabel('Avg Computation Density (%)')
    axes[0].set_title('Intensity vs Density (per pixel)')

    # Compute correlation
    if intensity_flat.std() > 0 and density_flat.std() > 0:
        corr = np.corrcoef(intensity_flat, density_flat)[0, 1]
        axes[0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0].transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2D histogram (density plot)
    axes[1].hist2d(intensity_flat, density_flat * 100, bins=30, cmap='Blues')
    axes[1].set_xlabel('Avg Pixel Intensity')
    axes[1].set_ylabel('Avg Computation Density (%)')
    axes[1].set_title('Joint Distribution')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Count')

    fig.suptitle(f'Spatial Density Correlation — {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_type_summary(layer_densities, layer_types, save_path, model_name):
    """Box plot comparing density across layer types."""
    type_data = defaultdict(list)
    for name, densities in layer_densities.items():
        lt = layer_types[name]
        type_data[lt].extend([d * 100 for d in densities])

    if not type_data:
        return

    labels = list(type_data.keys())
    data = [type_data[k] for k in labels]
    label_names = {
        'conv': 'Conv', 'linear': 'Linear',
        'attn_kv': 'Attn K^T@V', 'attn_qr': 'Attn Q@(K^T@V)'
    }
    display_labels = [label_names.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=display_labels)
    colors = {'conv': '#4C72B0', 'linear': '#DD8452',
              'attn_kv': '#55A868', 'attn_qr': '#C44E52'}
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(label, '#999999'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Computation Density (%)')
    ax.set_title(f'Density by Layer Type — {model_name}')

    # Add mean annotations
    for i, (label, d) in enumerate(zip(labels, data)):
        mean_val = np.mean(d)
        ax.text(i + 1, mean_val, f'{mean_val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_val_statistics(layer_densities, layer_types, per_sample_overall,
                         per_class_densities, model_name):
    """Print aggregate statistics."""
    overall_densities = [x[0] * 100 for x in per_sample_overall]

    print(f"\n{'='*80}")
    print(f"  Validation Set Density Statistics — {model_name}")
    print(f"  Samples: {len(per_sample_overall)}  |  Classes: {len(per_class_densities)}")
    print(f"{'='*80}")

    print(f"\n  Overall Density: mean={np.mean(overall_densities):.2f}%, "
          f"std={np.std(overall_densities):.2f}%, "
          f"min={np.min(overall_densities):.2f}%, max={np.max(overall_densities):.2f}%")

    print(f"\n  Per-Layer Type Summary:")
    print(f"  {'Type':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Layers':>7}")
    print(f"  {'-'*57}")

    type_agg = defaultdict(list)
    for name, densities in layer_densities.items():
        lt = layer_types[name]
        type_agg[lt].append(np.mean(densities) * 100)

    for lt, means in type_agg.items():
        print(f"  {lt:<18} {np.mean(means):>7.2f}% {np.std(means):>7.2f}% "
              f"{np.min(means):>7.2f}% {np.max(means):>7.2f}% {len(means):>7}")

    # Classes with highest/lowest density
    class_means = {c: np.mean(d) * 100 for c, d in per_class_densities.items() if d}
    if class_means:
        sorted_cls = sorted(class_means.items(), key=lambda x: x[1])
        print(f"\n  Top-5 LOWEST density classes:")
        for c, d in sorted_cls[:5]:
            print(f"    Class {c}: {d:.2f}%")
        print(f"  Top-5 HIGHEST density classes:")
        for c, d in sorted_cls[-5:]:
            print(f"    Class {c}: {d:.2f}%")

    print(f"{'='*80}\n")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading model and data...")
    model, model_name, val_loader, img_size, num_classes, device = \
        load_model_and_data(args)

    print(f"Model: {model_name}  |  Dataset: {args.dataset}  |  "
          f"Max samples: {args.max_samples or 'all'}")

    print("Collecting density statistics...")
    (layer_densities, layer_types, spatial_accum,
     per_sample_overall, per_class_densities,
     overall_spatial, image_intensity) = collect_density_stats(
        model, val_loader, img_size, device, args.max_samples)

    # Print statistics
    print_val_statistics(layer_densities, layer_types, per_sample_overall,
                         per_class_densities, model_name)

    # Generate plots
    prefix = f"{model_name}_{args.dataset}_val"
    print("Generating visualizations...")

    plot_density_distributions(
        layer_densities, layer_types,
        os.path.join(args.save_dir, f"{prefix}_distributions.png"), model_name)

    plot_overall_histogram(
        per_sample_overall,
        os.path.join(args.save_dir, f"{prefix}_overall_hist.png"), model_name)

    plot_per_class_density(
        per_class_densities,
        os.path.join(args.save_dir, f"{prefix}_per_class.png"), model_name)

    plot_avg_spatial_density(
        overall_spatial, image_intensity,
        os.path.join(args.save_dir, f"{prefix}_avg_spatial.png"), model_name)

    plot_density_correlation(
        overall_spatial, image_intensity,
        os.path.join(args.save_dir, f"{prefix}_correlation.png"), model_name)

    plot_layer_type_summary(
        layer_densities, layer_types,
        os.path.join(args.save_dir, f"{prefix}_type_summary.png"), model_name)

    print("Done!")


if __name__ == '__main__':
    main()
