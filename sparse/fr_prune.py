"""Neuron-level firing-rate-aware N:M structured pruning.

Profiling utilities (NeuronFiringRateProfiler, ChannelFiringRateProfiler,
compute_effective_rates_conv, compute_enhanced_rates_linear,
profile_neuron_firing_rates) live in utils.profiling and are imported here.

Key functions defined in this module:
    prune_n_m_neuron_aware     — neuron-level N:M pruning for Conv2d and Linear
    apply_neuron_aware_pruning — model-level neuron-aware pruning
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional
import torch.nn.functional as F
from sparse.pruning import prune_n_m, verify_n_m
from sparse.utils import (
    _is_eligible_for_permutation,
    _get_in_channels,
    _get_upstream_entry,
)
from utils.profiling import (
    compute_effective_rates_conv,
    compute_enhanced_rates_linear,
    profile_neuron_firing_rates,
)


# ---------------------------------------------------------------------------
# 1. prune_n_m_neuron_aware
# ---------------------------------------------------------------------------

def _ensure_tuple(val, n=2):
    """Convert int or single-element tuple to n-tuple."""
    if isinstance(val, int):
        return (val,) * n
    if len(val) == 1:
        return val * n
    return tuple(val)


def prune_n_m_neuron_aware(
    weight: torch.Tensor,
    neuron_info: dict,
    module: nn.Module,
    n: int = 2,
    m: int = 4,
    lam: float = 0.5,
    alpha: float = 0.5,
    scoring: str = 'multiplicative',
) -> torch.Tensor:
    """Apply N:M pruning with neuron-level firing rate awareness.

    For Conv2d: uses spatial firing rates with F.unfold to get per-row
    rate vectors (different kernel positions see different firing stats).

    For Linear: uses variance-boosted token rates.

    Args:
        weight: Original weight tensor (NCHW for Conv2d, 2D for Linear).
        neuron_info: Dict with 'spatial_rates', 'channel_mean', 'channel_std', 'layout'.
        module: The nn.Conv2d or nn.Linear module (for extracting kernel params).
        n: Non-zeros to keep per group.
        m: Group size.
        lam: Firing rate influence strength.
        alpha: Variance boost for token rates.
        scoring: 'multiplicative' for Wanda-style S=|W|*(1+lam*r),
                 'additive' for S=|W|+lam*scale*r.

    Returns:
        Pruned weight tensor in original layout (NCHW for Conv2d, 2D for Linear).
    """
    is_conv = isinstance(module, nn.Conv2d) and weight.ndim == 4
    layout = neuron_info['layout']
    spatial_rates = neuron_info['spatial_rates']
    channel_mean = neuron_info['channel_mean']

    if is_conv:
        return _prune_conv_neuron_aware(
            weight, spatial_rates, channel_mean, layout, module,
            n, m, lam, scoring,
        )
    else:
        return _prune_linear_neuron_aware(
            weight, spatial_rates, channel_mean, layout,
            n, m, lam, alpha, scoring,
        )


def _prune_conv_neuron_aware(
    weight: torch.Tensor,
    spatial_rates: torch.Tensor,
    channel_mean: torch.Tensor,
    layout: str,
    module: nn.Conv2d,
    n: int, m: int, lam: float, scoring: str,
) -> torch.Tensor:
    """Conv2d neuron-aware pruning with per-kernel-position rates."""
    C_out, C_in, Kh, Kw = weight.shape
    kernel_size = _ensure_tuple(module.kernel_size)
    stride = _ensure_tuple(module.stride)
    padding = _ensure_tuple(module.padding)
    dilation = _ensure_tuple(module.dilation)

    # Build per-row rate matrix
    if layout == 'conv' and spatial_rates.ndim == 3:
        sr = spatial_rates  # (C_in, H, W)

        # Handle spatial size mismatch: resize if needed
        # Compute expected input spatial size from output of unfold
        # Just try unfold; if spatial_rates is too small, resize
        sr_H, sr_W = sr.shape[1], sr.shape[2]

        # Check if unfold would produce valid output
        out_H = (sr_H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_W = (sr_W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

        if out_H <= 0 or out_W <= 0:
            # spatial_rates too small for this kernel; resize up
            min_H = dilation[0] * (kernel_size[0] - 1) + 1
            min_W = dilation[1] * (kernel_size[1] - 1) + 1
            sr = F.adaptive_avg_pool2d(
                sr.unsqueeze(0), (max(sr_H, min_H), max(sr_W, min_W))
            ).squeeze(0)

        # r_eff: (C_in, Kh, Kw) — effective rate per kernel position
        r_eff = compute_effective_rates_conv(sr, kernel_size, stride, padding, dilation)
        r_eff = r_eff.to(weight.device)

        # Build full rate matrix: (C_out, Kh, Kw, C_in)
        # r_eff is (C_in, Kh, Kw) -> permute to (Kh, Kw, C_in)
        # expand to (C_out, Kh, Kw, C_in)
        r_expanded = r_eff.permute(1, 2, 0).unsqueeze(0).expand(C_out, Kh, Kw, C_in)
        # Flatten to (C_out*Kh*Kw, C_in) to match NHWC 2D layout
        r_2d = r_expanded.reshape(-1, C_in).contiguous()
    else:
        # Fallback to channel-level rates (broadcast across all rows)
        r_1d = channel_mean.to(weight.device)  # (C_in,)
        rows = C_out * Kh * Kw
        r_2d = r_1d.unsqueeze(0).expand(rows, C_in)

    # Weight in NHWC 2D: (C_out, C_in, Kh, Kw) -> (C_out, Kh, Kw, C_in) -> (C_out*Kh*Kw, C_in)
    w_nhwc = weight.permute(0, 2, 3, 1).contiguous()
    w_2d = w_nhwc.reshape(-1, C_in)

    # Score and prune
    w_pruned_2d = _score_and_prune(w_2d, r_2d, n, m, lam, scoring)

    # Reshape back to NCHW
    w_nhwc_pruned = w_pruned_2d.reshape(C_out, Kh, Kw, C_in)
    return w_nhwc_pruned.permute(0, 3, 1, 2).contiguous()


def _prune_linear_neuron_aware(
    weight: torch.Tensor,
    spatial_rates: torch.Tensor,
    channel_mean: torch.Tensor,
    layout: str,
    n: int, m: int, lam: float, alpha: float, scoring: str,
) -> torch.Tensor:
    """Linear neuron-aware pruning with variance-boosted rates."""
    rows, cols = weight.shape

    if layout == 'token' and spatial_rates.ndim == 2:
        # (N, C) -> enhanced (C,)
        r_1d = compute_enhanced_rates_linear(spatial_rates, alpha=alpha)
    else:
        # Fallback to channel_mean
        r_1d = channel_mean

    r_1d = r_1d.to(weight.device)
    r_2d = r_1d.unsqueeze(0).expand(rows, cols)

    return _score_and_prune(weight, r_2d, n, m, lam, scoring)


def _score_and_prune(
    w_2d: torch.Tensor,
    r_2d: torch.Tensor,
    n: int, m: int,
    lam: float,
    scoring: str,
) -> torch.Tensor:
    """Core scoring and N:M pruning on 2D weight with 2D rate matrix.

    Args:
        w_2d: (rows, cols) weight tensor.
        r_2d: (rows, cols) per-element firing rate values.
        n, m: N:M sparsity parameters.
        lam: Rate influence strength.
        scoring: 'multiplicative' or 'additive'.

    Returns:
        Pruned (rows, cols) weight tensor.
    """
    rows, cols = w_2d.shape
    num_groups = cols // m
    if num_groups == 0:
        return w_2d.clone()

    aligned_cols = num_groups * m
    w_abs = w_2d[:, :aligned_cols].abs()
    r_aligned = r_2d[:, :aligned_cols]

    # Normalize rates to [0, 1] so lam has meaningful dynamic range.
    # Without this, raw rates (typically 0.02–0.16) make the modulation
    # negligible: 1 + lam * 0.05 ≈ 1.02, i.e. magnitude still dominates.
    r_min = r_aligned.min()
    r_max = r_aligned.max()
    r_range = r_max - r_min
    if r_range > 1e-8:
        r_norm = (r_aligned - r_min) / r_range  # [0, 1]
    else:
        r_norm = torch.zeros_like(r_aligned)

    if scoring == 'multiplicative':
        # Wanda-style: S = |W| * (1 + lam * r_norm)
        # With r_norm in [0,1], lam=0.5 gives score range [|W|, 1.5*|W|]
        scores = w_abs * (1.0 + lam * r_norm)
    else:
        # Additive: S = |W| + lam * scale * r_norm
        w_mean = w_abs.mean()
        scores = w_abs + lam * w_mean * r_norm

    # Group into (rows, num_groups, m), keep top-N per group
    scores_grouped = scores.reshape(rows, num_groups, m)
    num_prune = m - n
    _, small_idx = scores_grouped.topk(num_prune, dim=2, largest=False)
    mask = torch.ones_like(scores_grouped)
    mask.scatter_(2, small_idx, 0.0)

    w_grouped = w_2d[:, :aligned_cols].reshape(rows, num_groups, m)
    result = w_2d.clone()
    result[:, :aligned_cols] = (w_grouped * mask).reshape(rows, aligned_cols)
    return result


# ---------------------------------------------------------------------------
# 5. _get_upstream_neuron_rates
# ---------------------------------------------------------------------------

def _get_upstream_neuron_rates(
    model: nn.Module,
    layer_name: str,
    neuron_rates: dict[str, dict],
) -> Optional[dict]:
    """Get neuron-level rate entry for the upstream neuron of a layer.

    Wrapper around _get_upstream_entry for spatial/token-level rate dicts
    (values are dicts with 'channel_mean', 'spatial_rates', etc.).
    """
    return _get_upstream_entry(model, layer_name, neuron_rates,
                               get_channels=lambda e: e['channel_mean'].shape[0])


# ---------------------------------------------------------------------------
# 6. apply_neuron_aware_pruning
# ---------------------------------------------------------------------------

def apply_neuron_aware_pruning(
    model: nn.Module,
    neuron_rates: dict[str, dict],
    lam: float = 0.5,
    alpha: float = 0.5,
    scoring: str = 'multiplicative',
    n: int = 2,
    m: int = 4,
    exclude_names: Optional[list[str]] = None,
) -> nn.Module:
    """Apply neuron-level firing-rate-aware N:M pruning to all eligible layers.

    Iterates over Conv2d and Linear layers, finds upstream neuron rates,
    and applies neuron-aware pruning in-place.

    Args:
        model: The SNN model (modified in-place).
        neuron_rates: {neuron_name: {spatial_rates, channel_mean, channel_std, layout}}
            from NeuronFiringRateProfiler.get_neuron_rates().
        lam: Firing rate influence strength. 0 = pure magnitude pruning.
        alpha: Variance boost for transformer token rates.
        scoring: 'multiplicative' (Wanda-style) or 'additive'.
        n: Non-zeros to keep per group.
        m: Group size.
        exclude_names: Module name prefixes to skip.

    Returns:
        The modified model (same object).
    """
    if exclude_names is None:
        exclude_names = ['head']

    pruning_stats = OrderedDict()

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv = isinstance(module, nn.Conv2d)
        if not (is_linear or is_conv):
            continue

        layer_type = 'Linear' if is_linear else 'Conv2d'

        # Check exclusion
        skip = False
        for excl in exclude_names:
            if name == excl or name.startswith(excl + '.'):
                skip = True
                break
        if skip:
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': 'excluded', 'shape': tuple(module.weight.shape),
                'granularity': '-',
            }
            continue

        # Check eligibility
        if not _is_eligible_for_permutation(module):
            in_ch = module.in_features if is_linear else module.in_channels
            reason = f'in_channels={in_ch} < 4'
            if is_conv and module.groups == module.in_channels and module.in_channels > 1:
                reason = 'depthwise conv'
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': reason, 'shape': tuple(module.weight.shape),
                'granularity': '-',
            }
            continue

        # Find upstream neuron rates
        info = _get_upstream_neuron_rates(model, name, neuron_rates)
        if info is None:
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': 'no upstream neuron rates found',
                'shape': tuple(module.weight.shape),
                'granularity': '-',
            }
            continue

        in_channels = _get_in_channels(module)
        if info['channel_mean'].shape[0] != in_channels:
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': f"rate channels {info['channel_mean'].shape[0]} != {in_channels}",
                'shape': tuple(module.weight.shape),
                'granularity': '-',
            }
            continue

        # Determine granularity used
        layout = info['layout']
        if is_conv and layout == 'conv' and info['spatial_rates'].ndim == 3:
            granularity = 'neuron-level (spatial)'
        elif is_linear and layout == 'token' and info['spatial_rates'].ndim == 2:
            granularity = 'neuron-level (token-var)'
        else:
            granularity = 'channel-level (fallback)'

        # Prune
        with torch.no_grad():
            w_pruned = prune_n_m_neuron_aware(
                module.weight.data, info, module,
                n=n, m=m, lam=lam, alpha=alpha, scoring=scoring,
            )
            module.weight.data.copy_(w_pruned)

        # Compute sparsity
        total = module.weight.numel()
        nonzero = module.weight.data.count_nonzero().item()
        sparsity = 1.0 - nonzero / max(total, 1)

        pruning_stats[name] = {
            'pruned': True, 'type': layer_type,
            'reason': 'success', 'shape': tuple(module.weight.shape),
            'sparsity': sparsity, 'granularity': granularity,
        }

    # Print summary
    pruned_count = sum(1 for s in pruning_stats.values() if s['pruned'])
    total_count = len(pruning_stats)
    print(f"\nNeuron-aware {n}:{m} pruning (lam={lam}, alpha={alpha}, "
          f"scoring={scoring}): {pruned_count}/{total_count} layers pruned")
    for name, stats in pruning_stats.items():
        if stats['pruned']:
            print(f"  {name} [{stats['type']}]: {stats['shape']} -> "
                  f"sparsity={stats['sparsity']:.3f}  [{stats['granularity']}]")
        else:
            print(f"  {name} [{stats['type']}]: {stats['shape']} -> "
                  f"SKIP ({stats['reason']})")

    return model


# ---------------------------------------------------------------------------
# 7. CLI __main__
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Neuron-level firing-rate-aware N:M structured pruning for SNNs')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. sew_resnet18)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name: cifar10, cifar100, imagenet, cifar10dvs')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for profiling and evaluation')

    # Pruning params
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Firing rate influence strength (0=pure magnitude)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Variance boost weight for transformer token rates')
    parser.add_argument('--scoring', type=str, default='multiplicative',
                        choices=['multiplicative', 'additive'],
                        help='Score function: multiplicative (Wanda-style) or additive')
    parser.add_argument('--n', type=int, default=2, help='N in N:M sparsity')
    parser.add_argument('--m', type=int, default=4, help='M in N:M sparsity')

    # Profiling params
    parser.add_argument('--profile-batches', type=int, default=50,
                        help='Number of batches for firing rate profiling')
    parser.add_argument('--max-spatial', type=int, default=16,
                        help='Max spatial dim to retain during profiling')

    # Output
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after pruning')
    parser.add_argument('--output', type=str, default=None,
                        help='Save pruned model to this path')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'],
                        help='Module name prefixes to exclude from pruning')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders,
    )
    from models.neurons import reset_net
    from sparse.permutation import measure_permutation_quality

    # Device
    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Dataset config
    ds_cfg = get_dataset_config(args.dataset)

    # Build model
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'], T=args.T)
    else:
        parser.error('Must specify either --config or --model')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    # Build dataloader
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    # Profile neuron firing rates
    print("=== Profiling neuron-level firing rates ===")
    neuron_rates = profile_neuron_firing_rates(
        model, val_loader, device,
        max_batches=args.profile_batches,
        max_spatial=args.max_spatial,
    )

    # Extract channel-level rates for quality measurement
    channel_rates = {name: info['channel_mean'] for name, info in neuron_rates.items()}

    # Measure quality before pruning
    print("\n=== Quality BEFORE pruning ===")
    quality_before = measure_permutation_quality(model, channel_rates)
    for name, q in quality_before.items():
        print(f"  {name} [{q['type']}]: alignment={q['alignment_score']:.3f}, "
              f"info_loss={q['relative_information_loss']:.4f}")

    # Apply neuron-aware pruning
    print(f"\n=== Applying neuron-aware {args.n}:{args.m} pruning ===")
    model = apply_neuron_aware_pruning(
        model, neuron_rates,
        lam=args.lam, alpha=args.alpha, scoring=args.scoring,
        n=args.n, m=args.m, exclude_names=args.exclude,
    )

    # Verify N:M pattern
    print(f"\n=== Verifying {args.n}:{args.m} sparsity pattern ===")
    all_valid = True
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.ndim == 4:
            w = module.weight.data
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()
            w_2d = w_nhwc.reshape(-1, w.shape[1])
            valid = verify_n_m(w_2d, n=args.n, m=args.m)
            if not valid:
                # Check if it was skipped (not pruned)
                skip = any(name == e or name.startswith(e + '.') for e in args.exclude)
                if not skip and _is_eligible_for_permutation(module):
                    print(f"  WARN: {name} does NOT satisfy {args.n}:{args.m} pattern")
                    all_valid = False
        elif isinstance(module, nn.Linear):
            valid = verify_n_m(module.weight.data, n=args.n, m=args.m)
            if not valid:
                skip = any(name == e or name.startswith(e + '.') for e in args.exclude)
                if not skip and _is_eligible_for_permutation(module):
                    print(f"  WARN: {name} does NOT satisfy {args.n}:{args.m} pattern")
                    all_valid = False
    if all_valid:
        print(f"  All pruned layers satisfy {args.n}:{args.m} pattern")

    # Measure quality after pruning
    print("\n=== Quality AFTER pruning ===")
    quality_after = measure_permutation_quality(model, channel_rates)
    for name, q in quality_after.items():
        print(f"  {name} [{q['type']}]: alignment={q['alignment_score']:.3f}, "
              f"info_loss={q['relative_information_loss']:.4f}, "
              f"sparsity={q['sparsity']:.3f}")

    # Save
    if args.output:
        torch.save({
            'model': model.state_dict(),
            'method': 'neuron_aware',
            'params': {
                'lam': args.lam, 'alpha': args.alpha,
                'scoring': args.scoring, 'n': args.n, 'm': args.m,
            },
            'quality_before': quality_before,
            'quality_after': quality_after,
        }, args.output)
        print(f"\nSaved pruned model to {args.output}")

    # Evaluate
    if args.evaluate:
        print("\n=== Evaluation ===")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                reset_net(model)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        print(f"Evaluation accuracy: {acc:.2f}% ({correct}/{total})")
