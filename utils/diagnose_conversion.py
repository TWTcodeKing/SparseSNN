"""Diagnose per-layer output error from N:M pruning to validate compensation theory.

For each pruned Linear/Conv2d layer, computes three quantities and checks
whether they correlate:

  rel_error  — actual relative output error using firing rates as E[x] proxy:
                 || (W_nm - W_orig) @ r ||_2  /  || W_orig @ r ||_2
               This is the ground truth for how much each layer's expected
               output was disturbed by pruning.

  bound      — triangle-inequality upper bound on rel_error:
                 Σ_{pruned (i,c)} |W_orig[i,c]| * r_c  /  Σ_{all} |W[i,c]| * r_c
               Equal to relative_information_loss from measure_permutation_quality.
               Closed-form: no data needed beyond weights + rates.

  info_loss  — relative_information_loss already stored in the pruned .pth file's
               quality_after dict. Should match bound exactly if computed the same
               way; serves as a sanity check.

Validation criterion:
  Pearson(rel_error, bound) > 0.8 → compensation will close the accuracy gap.
  Pearson(rel_error, bound) < 0.5 → error source is elsewhere (e.g. LayerNorm,
                                     accumulated errors, non-linearity mismatch).

Usage:
  uv run utils/diagnose_conversion.py \\
    --checkpoint output/.../best.pth \\
    --pruned pruned_neuron.pth \\
    --rates firing_rates.pt \\
    --config configs/spikformer/spikformer_cifar.yaml \\
    --dataset cifar100
"""

import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from sparse.utils import (
    _get_upstream_entry,
    _get_in_channels,
    _is_eligible_for_permutation,
)
try:
    from sparse.permutation import measure_permutation_quality
except ImportError:
    measure_permutation_quality = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2d(W: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Reshape weight to 2D (rows, in_channels) matching how pruning was applied.

    Linear: (out, in)         → already 2D, return as-is.
    Conv2d: (C_out, C_in, Kh, Kw) → NHWC view → (C_out*Kh*Kw, C_in).
    """
    if isinstance(module, nn.Conv2d):
        C_in = W.shape[1]
        return W.permute(0, 2, 3, 1).reshape(-1, C_in).contiguous()
    return W  # Linear: already (out, in)


def _pearson(x: list, y: list) -> float:
    """Pearson correlation ignoring NaN pairs."""
    a, b = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Per-layer diagnostic
# ---------------------------------------------------------------------------

def diagnose_layer(
    name: str,
    module: nn.Module,
    W_orig_raw: torch.Tensor,
    W_nm_raw: torch.Tensor,
    r: torch.Tensor,
    quality_entry: dict,
) -> dict:
    """Compute error metrics for one layer.

    Args:
        name: Layer name (for reporting).
        module: The nn.Linear or nn.Conv2d module (from the dense model,
                used for type dispatch only — weights come from state dicts).
        W_orig_raw: Dense weight tensor from checkpoint (NCHW for Conv2d).
        W_nm_raw:   Pruned weight tensor from .pth file (NCHW for Conv2d).
        r:          (in_channels,) firing rates from upstream neuron.
        quality_entry: Dict from quality_after[name], or None.

    Returns:
        Dict with all computed metrics.
    """
    in_ch = _get_in_channels(module)

    # Reshape to 2D view matching how N:M pruning was applied
    W_o = _to_2d(W_orig_raw, module).float()   # (rows, in_ch)
    W_n = _to_2d(W_nm_raw, module).float()     # (rows, in_ch)
    r_f = r.float().to(W_o.device)             # (in_ch,)

    if r_f.shape[0] != in_ch:
        return {'skip': f'rate shape {r_f.shape[0]} != in_ch {in_ch}'}

    rows = W_o.shape[0]

    # ------------------------------------------------------------------
    # 1. Actual output error: (W_nm - W_orig) @ r
    #    At pruned positions: W_nm[i,c] = 0 → W_diff[i,c] = -W_orig[i,c]
    #    At surviving positions: W_nm[i,c] = W_orig[i,c] → W_diff[i,c] = 0
    # ------------------------------------------------------------------
    W_diff = W_n - W_o                          # (rows, in_ch)
    err_vec = W_diff @ r_f                      # (rows,)
    expected_orig = W_o @ r_f                   # (rows,)

    norm_err  = err_vec.norm().item()
    norm_orig = expected_orig.norm().item()
    rel_error = norm_err / (norm_orig + 1e-8)

    # ------------------------------------------------------------------
    # 2. Pruned positions: zeroed in W_nm but non-zero in W_orig
    # ------------------------------------------------------------------
    pruned_mask = (W_n.abs() < 1e-8) & (W_o.abs() > 1e-8)  # (rows, in_ch)
    n_pruned  = int(pruned_mask.sum().item())
    n_total   = W_o.numel()
    sparsity  = n_pruned / max(n_total, 1)

    # ------------------------------------------------------------------
    # 3. Triangle-inequality bound
    #    Σ_{pruned (i,c)} |W_orig[i,c]| * r_c  /  Σ_{all} |W[i,c]| * r_c
    # ------------------------------------------------------------------
    r_row = r_f.unsqueeze(0).expand(rows, -1)   # (rows, in_ch)
    weighted = W_o.abs() * r_row                 # |W| * r per element
    bound_num   = weighted[pruned_mask].sum().item()
    bound_denom = weighted.sum().item()
    bound = bound_num / (bound_denom + 1e-8)

    # Alternative bound denominator: ||W_orig @ r||_2  (L2-normalised)
    bound_l2 = bound_num / (norm_orig + 1e-8)

    # ------------------------------------------------------------------
    # 4. Existing info_loss from quality_after (sanity check)
    # ------------------------------------------------------------------
    info_loss = float('nan')
    if quality_entry is not None:
        info_loss = quality_entry.get('relative_information_loss', float('nan'))

    # ------------------------------------------------------------------
    # 5. Cancellation ratio: how much do pruned contributions cancel out?
    #    bound_num / norm_err — if >> 1, positive/negative errors cancel,
    #    so compensation is less critical than the bound suggests.
    # ------------------------------------------------------------------
    cancellation = bound_num / (norm_err + 1e-8)

    return {
        'rel_error':    rel_error,
        'bound':        bound,
        'bound_l2':     bound_l2,
        'info_loss':    info_loss,
        'sparsity':     sparsity,
        'n_pruned':     n_pruned,
        'n_total':      n_total,
        'norm_err':     norm_err,
        'norm_orig':    norm_orig,
        'cancellation': cancellation,    # bound_num / norm_err
        'type':         'Conv2d' if isinstance(module, nn.Conv2d) else 'Linear',
        'in_ch':        in_ch,
        'rows':         rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose per-layer output error from N:M pruning')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Dense model checkpoint (.pth)')
    parser.add_argument('--pruned', type=str, required=True,
                        help='Pruned model file (.pth) from fr_prune or permutation')
    parser.add_argument('--rates', type=str, required=True,
                        help='Firing rates file (.pt) from firing_rate_profile.py')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (alternative to --config)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar10, cifar100, imagenet, ...)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps (default 4)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config,
    )

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Build model structure (for named_modules and upstream lookup)
    # ------------------------------------------------------------------
    ds_cfg = get_dataset_config(args.dataset)
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'], T=args.T)
    else:
        parser.error('Must specify --config or --model')

    # ------------------------------------------------------------------
    # Load dense weights
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dense_state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(dense_state)
    model.eval()
    print(f"Loaded dense model from {args.checkpoint}")

    # ------------------------------------------------------------------
    # Load pruned weights + quality metrics
    # ------------------------------------------------------------------
    pruned_data = torch.load(args.pruned, map_location='cpu', weights_only=False)
    pruned_state  = pruned_data['model']
    print(f"Loaded pruned model from {args.pruned}")
    print(f"  method={pruned_data.get('method')}, "
          f"params={pruned_data.get('params')}")

    # ------------------------------------------------------------------
    # Load firing rates
    # ------------------------------------------------------------------
    rates_data   = torch.load(args.rates, map_location='cpu', weights_only=False)
    firing_rates = rates_data['firing_rates']
    print(f"Loaded firing rates from {args.rates}: {len(firing_rates)} neuron layers")

    # Filter out duplicate '.neuron' entries (keep the shorter name)
    firing_rates = {
        k: v for k, v in firing_rates.items()
        if not k.endswith('.neuron')
    }

    # Recompute quality_after using the fixed measure_permutation_quality.
    # The stored quality_after in old .pth files has info_loss=0 due to a
    # bug (pruned positions have weight=0 so |W|*r = 0 at those positions).
    # The fixed version passes the dense reference weights so pruned
    # positions correctly reflect the energy that was removed.
    model.load_state_dict(pruned_state)
    quality_after = measure_permutation_quality(
        model, firing_rates, reference_state_dict=dense_state)
    print(f"  Recomputed quality_after: {len(quality_after)} layers (bug-fixed info_loss)")
    # Restore dense weights for per-layer weight comparison below
    model.load_state_dict(dense_state)

    # ------------------------------------------------------------------
    # Per-layer diagnostic
    # ------------------------------------------------------------------
    results = OrderedDict()
    skipped = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_permutation(module):
            continue

        w_key = name + '.weight'
        if w_key not in dense_state:
            skipped.append((name, 'not in dense_state'))
            continue
        if w_key not in pruned_state:
            skipped.append((name, 'not in pruned_state'))
            continue

        W_orig = dense_state[w_key]
        W_nm   = pruned_state[w_key]

        # Upstream firing rates via 3-strategy lookup
        r = _get_upstream_entry(
            model, name, firing_rates,
            get_channels=lambda v: v.shape[0],
        )
        if r is None:
            skipped.append((name, 'no upstream rates found'))
            continue

        quality_entry = quality_after.get(name)
        result = diagnose_layer(name, module, W_orig, W_nm, r, quality_entry)

        if 'skip' in result:
            skipped.append((name, result['skip']))
            continue

        results[name] = result

    # ------------------------------------------------------------------
    # Print skipped layers
    # ------------------------------------------------------------------
    if skipped:
        print(f"\n--- Skipped layers ({len(skipped)}) ---")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    if not results:
        print("\nNo layers diagnosed. Check checkpoint/pruned/rates files.")
        return

    # ------------------------------------------------------------------
    # Print per-layer table
    # ------------------------------------------------------------------
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rel_error'],
                            reverse=True)

    header = (f"{'Layer':<42} {'Type':6} {'Spar':5} "
              f"{'rel_err':>8} {'bound':>7} {'info_loss':>9} "
              f"{'cancel':>7} {'norm_err':>9} {'norm_orig':>9}")
    sep = '-' * len(header)

    print(f"\n--- Per-layer output error (sorted by rel_error, descending) ---")
    print(header)
    print(sep)

    for name, r in sorted_results:
        il = f"{r['info_loss']:.4f}" if not np.isnan(r['info_loss']) else '  N/A '
        print(
            f"{name:<42} {r['type']:6} {r['sparsity']:.3f} "
            f"{r['rel_error']:>8.4f} {r['bound']:>7.4f} {il:>9} "
            f"{r['cancellation']:>7.2f} {r['norm_err']:>9.4f} {r['norm_orig']:>9.4f}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    rel_errors  = [r['rel_error'] for r in results.values()]
    bounds      = [r['bound']     for r in results.values()]
    bound_l2s   = [r['bound_l2']  for r in results.values()]
    info_losses = [r['info_loss'] for r in results.values()]
    cancels     = [r['cancellation'] for r in results.values()]

    corr_bound    = _pearson(rel_errors, bounds)
    corr_bound_l2 = _pearson(rel_errors, bound_l2s)
    corr_info     = _pearson(rel_errors, info_losses)

    total_err = sum(r['norm_err']  for r in results.values())
    total_ref = sum(r['norm_orig'] for r in results.values())
    global_rel = total_err / (total_ref + 1e-8)

    print(f"\n{'='*len(sep)}")
    print(f"SUMMARY")
    print(f"{'='*len(sep)}")
    print(f"  Diagnosed layers:         {len(results)}")
    print(f"  Skipped layers:           {len(skipped)}")
    print(f"  Global rel_error (Σnorm): {global_rel:.4f}  "
          f"(total ‖err‖={total_err:.4f} / total ‖orig‖={total_ref:.4f})")
    print(f"  Mean cancellation ratio:  {np.mean(cancels):.2f}x  "
          f"(bound_num/rel_error; >1 means signs cancel)")
    print()
    print(f"  Pearson(rel_error, bound):       {corr_bound:.4f}  "
          f"← validate theory (L1-normalised bound)")
    print(f"  Pearson(rel_error, bound_l2):    {corr_bound_l2:.4f}  "
          f"← validate theory (L2-normalised bound)")
    print(f"  Pearson(rel_error, info_loss):   {corr_info:.4f}  "
          f"← sanity check vs existing metric")
    print()

    if corr_bound > 0.8:
        print("  ✓ Strong correlation: compensation should close the accuracy gap.")
    elif corr_bound > 0.5:
        print("  ~ Moderate correlation: compensation will help but other factors matter.")
    else:
        print("  ✗ Weak correlation: error source is NOT dominated by pruned column "
              "contributions. Investigate layer-norm instability, accumulated errors, "
              "or non-linearity mismatch.")

    print()
    print("  Top-5 bottleneck layers (highest rel_error):")
    for name, r in sorted_results[:5]:
        print(f"    {name:<42}  rel_error={r['rel_error']:.4f}  "
              f"bound={r['bound']:.4f}  cancel={r['cancellation']:.2f}x")

    print()
    print("  Columns: spar=sparsity, cancel=bound_num/‖err‖ "
          "(>1 means cancellation reduces actual error below bound)")


if __name__ == '__main__':
    main()
