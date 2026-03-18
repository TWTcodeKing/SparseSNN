"""Spike-to-Structured-Sparse Conversion (S2SC): element-wise compensation.

After N:M pruning, the surviving (non-zero) weights are analytically adjusted
so that the model's expected output E[y] = W_comp @ r matches the original
E[y] = W_orig @ r for every N:M group in every row.  No retraining required.

Core insight (from ANN2SNN residual membrane potential wisdom):
  Pruned weight W_orig[i,c] carries expected contribution W_orig[i,c] * r_c
  that the downstream neuron i relied on.  Instead of discarding it, we
  redistribute it to the N surviving weights in the same group using the
  least-norm solution that exactly restores the group-wise inner product:

    target[i,g] = Σ_{pruned c in group g, row i}  W_orig[i,g,c] * r[g,c]
    δ[i,g,c']   = target[i,g] * r[g,c'] / Σ_{surviving c''} r[g,c'']²

  After applying δ:
    W_comp[i,g,:] @ r[g,:] = W_orig[i,g,:] @ r[g,:]   ← exactly restored

  The N:M sparsity pattern is preserved because δ is added only to surviving
  (non-zero) positions.

Key functions:
    compensate_n_m_pruning  — per-layer element-wise compensation (2D weight)
    apply_compensation      — model-level compensation pass
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional

from sparse.pruning import verify_n_m
from sparse.utils import (
    _get_upstream_entry,
    _get_in_channels,
    _is_eligible_for_permutation,
)


# ---------------------------------------------------------------------------
# Core compensation function
# ---------------------------------------------------------------------------

def compensate_n_m_pruning(
    W_nm: torch.Tensor,
    W_orig: torch.Tensor,
    r: torch.Tensor,
    n: int = 2,
    m: int = 4,
) -> torch.Tensor:
    """Element-wise compensation for N:M pruning.

    Adjusts surviving (non-zero) weights so that the expected output
    E[y_i] = W[i,:] @ r is restored for each N:M group in each row.

    For group g in row i, the least-norm correction is:
        target[i,g] = Σ_{pruned c}    W_orig[i,g,c] * r[g,c]
        δ[i,g,c']   = target[i,g] * r[g,c'] / Σ_{surviving c''} r[g,c'']²

    This guarantees:
        W_comp[i,g,:] @ r[g,:] = W_orig[i,g,:] @ r[g,:]

    Args:
        W_nm:   (rows, in_ch) — N:M pruned weight in 2D view.
                Pruned positions must be exactly zero.
        W_orig: (rows, in_ch) — original dense weight (same 2D view).
        r:      (in_ch,)      — per-channel firing rates of upstream neuron.
        n:      Non-zeros to keep per group (default 2).
        m:      Group size (default 4).

    Returns:
        W_comp: (rows, in_ch) compensated weight.  N:M pattern preserved —
                only surviving (non-zero) positions are modified.
    """
    rows, in_ch = W_nm.shape
    num_groups = in_ch // m
    aligned_ch = num_groups * m

    W_comp = W_nm.clone()

    if num_groups == 0:
        return W_comp

    # -----------------------------------------------------------------------
    # Reshape to grouped view: (rows, num_groups, m)
    # -----------------------------------------------------------------------
    Wn_g  = W_nm[:, :aligned_ch].reshape(rows, num_groups, m)    # pruned
    Wo_g  = W_orig[:, :aligned_ch].reshape(rows, num_groups, m)  # original
    r_g   = r[:aligned_ch].reshape(num_groups, m)                 # (num_groups, m)
    r_bc  = r_g.unsqueeze(0)                                      # (1, num_groups, m)

    # -----------------------------------------------------------------------
    # Per-element pruned / surviving masks  (rows, num_groups, m)
    # -----------------------------------------------------------------------
    is_pruned    = (Wn_g == 0)   # True where weight was zeroed by N:M pruning
    is_surviving = ~is_pruned    # True where weight is non-zero

    # -----------------------------------------------------------------------
    # target[i,g] = Σ_{pruned c} W_orig[i,g,c] * r[g,c]   (rows, num_groups)
    # — expected output contribution that was lost at pruned positions
    # -----------------------------------------------------------------------
    target = (Wo_g * r_bc * is_pruned.float()).sum(dim=2)

    # -----------------------------------------------------------------------
    # denom[i,g] = Σ_{surviving c'} r[g,c']²              (rows, num_groups)
    # — used in least-norm solution; clamped for numerical safety
    # -----------------------------------------------------------------------
    denom = (r_bc ** 2 * is_surviving.float()).sum(dim=2).clamp(min=1e-8)

    # -----------------------------------------------------------------------
    # delta[i,g,c'] = target[i,g] * r[g,c'] / denom[i,g]
    # — correction to surviving weights; zero at pruned positions
    # -----------------------------------------------------------------------
    delta = (target.unsqueeze(2) * r_bc) / denom.unsqueeze(2)  # (rows, num_groups, m)
    delta = delta * is_surviving.float()

    # Write compensated weights back
    W_comp[:, :aligned_ch] = (Wn_g + delta).reshape(rows, aligned_ch)

    # Remainder columns (in_ch % m != 0): belong to no complete group → skip
    return W_comp


# ---------------------------------------------------------------------------
# Model-level compensation
# ---------------------------------------------------------------------------

def apply_compensation(
    model: nn.Module,
    original_state_dict: dict,
    firing_rates: dict,
    n: int = 2,
    m: int = 4,
    alpha: float = 0.2,
    exclude_names: Optional[list] = None,
) -> nn.Module:
    """Apply element-wise N:M compensation to all eligible layers in-place.

    For each eligible Linear / Conv2d layer that has already been N:M pruned,
    calls compensate_n_m_pruning to adjust surviving weights so that
    E[y] = W_comp @ r ≈ W_orig @ r per N:M group.

    Note on alpha (scale factor):
        alpha=1.0 achieves exact restoration of E[output] in the linear regime,
        but hurts SNN accuracy because SNN threshold neurons are non-linear:
        larger weights → more membrane potential → altered firing rates downstream.
        alpha=0.2 is empirically optimal on Spikformer-CIFAR100 (58.02% vs 57.58%
        pruned baseline vs 52.39% at alpha=1.0).

    Args:
        model:                Pruned SNN model.  Modified in-place.
        original_state_dict:  State dict of the dense (pre-pruning) model.
        firing_rates:         {neuron_name: (C,) rates} from
                              ChannelFiringRateProfiler / profile_model_firing_rates.
        n:                    Non-zeros per group (default 2).
        m:                    Group size (default 4).
        alpha:                Compensation scale in [0, 1].  0 = no compensation
                              (original pruned weights), 1 = full least-norm
                              compensation.  Default 0.2 (empirically optimal for
                              SNNs; balances error correction vs non-linearity).
        exclude_names:        Module name prefixes to skip (default ['head']).

    Returns:
        The modified model (same object).
    """
    if exclude_names is None:
        exclude_names = ['head']

    stats = OrderedDict()

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_permutation(module):
            continue

        # Exclusion check
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            stats[name] = {'status': 'excluded'}
            continue

        w_key = name + '.weight'
        if w_key not in original_state_dict:
            stats[name] = {'status': 'no reference weight'}
            continue

        # Upstream firing rates for this layer's input channels
        r = _get_upstream_entry(
            model, name, firing_rates, get_channels=lambda v: v.shape[0])
        if r is None:
            stats[name] = {'status': 'no upstream rates'}
            continue

        in_ch = _get_in_channels(module)
        if r.shape[0] != in_ch:
            stats[name] = {'status': f'rate mismatch {r.shape[0]} vs {in_ch}'}
            continue

        # Work on CPU in float32 for numerical precision
        W_nm   = module.weight.data.cpu().float()
        W_orig = original_state_dict[w_key].cpu().float()
        r_cpu  = r.cpu().float()
        device = module.weight.device
        is_conv = isinstance(module, nn.Conv2d)

        # Reshape to 2D matching the NHWC view used during pruning
        if is_conv:
            shape_nchw = W_nm.shape
            W_nm_2d   = W_nm.permute(0, 2, 3, 1).reshape(-1, in_ch).contiguous()
            W_orig_2d = W_orig.permute(0, 2, 3, 1).reshape(-1, in_ch).contiguous()
        else:
            W_nm_2d   = W_nm     # (out, in)
            W_orig_2d = W_orig   # (out, in)

        # Relative output error before compensation (E[x] ≈ r)
        err_before = ((W_nm_2d - W_orig_2d) @ r_cpu).norm().item()
        ref_norm   = (W_orig_2d @ r_cpu).norm().item()

        # Apply element-wise compensation (scaled by alpha)
        W_full_2d = compensate_n_m_pruning(W_nm_2d, W_orig_2d, r_cpu, n=n, m=m)
        W_comp_2d = W_nm_2d + alpha * (W_full_2d - W_nm_2d)

        # Relative output error after compensation
        err_after = ((W_comp_2d - W_orig_2d) @ r_cpu).norm().item()

        # Reshape back to original layout and update model weights
        if is_conv:
            C_out, C_in, Kh, Kw = shape_nchw
            W_comp = (W_comp_2d.reshape(C_out, Kh, Kw, C_in)
                                .permute(0, 3, 1, 2).contiguous())
        else:
            W_comp = W_comp_2d

        module.weight.data.copy_(W_comp.to(device))

        stats[name] = {
            'status':           'ok',
            'type':             'Conv2d' if is_conv else 'Linear',
            'rel_error_before': err_before / (ref_norm + 1e-8),
            'rel_error_after':  err_after  / (ref_norm + 1e-8),
            'reduction_factor': err_before / (err_after + 1e-8),
            'pattern_ok':       verify_n_m(W_comp_2d, n=n, m=m),
        }

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    ok = [s for s in stats.values() if s.get('status') == 'ok']
    skipped = [(n_, s) for n_, s in stats.items() if s.get('status') != 'ok']

    print(f"\nCompensation: {len(ok)}/{len(stats)} eligible layers adjusted")
    print(f"{'Layer':<42} {'Type':6} "
          f"{'err_before':>10} {'err_after':>10} {'reduction':>10} {'N:M ok':>7}")
    print('-' * 88)
    for lname, s in stats.items():
        if s['status'] != 'ok':
            continue
        print(f"  {lname:<40} {s['type']:6} "
              f"{s['rel_error_before']:>10.4f} {s['rel_error_after']:>10.4f} "
              f"{s['reduction_factor']:>10.1f}x {str(s['pattern_ok']):>7}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} layer(s):")
        for lname, s in skipped:
            print(f"    {lname}: {s['status']}")

    if ok:
        mean_b = sum(s['rel_error_before'] for s in ok) / len(ok)
        mean_a = sum(s['rel_error_after']  for s in ok) / len(ok)
        print(f"\n  Mean rel_error:  {mean_b:.4f} → {mean_a:.4f}  "
              f"({mean_b / (mean_a + 1e-8):.1f}x reduction)")
        bad_pattern = [n_ for n_, s in stats.items()
                       if s.get('status') == 'ok' and not s['pattern_ok']]
        if bad_pattern:
            print(f"\n  WARNING: N:M pattern broken in {len(bad_pattern)} layer(s): "
                  f"{bad_pattern}")
        else:
            print(f"  N:M pattern preserved in all {len(ok)} compensated layers ✓")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply element-wise N:M compensation to a pruned SNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Dense model checkpoint (.pth) — original weights')
    parser.add_argument('--pruned', type=str, required=True,
                        help='Pruned model (.pth) from fr_prune or permutation')
    parser.add_argument('--rates', type=str, required=True,
                        help='Firing rates file (.pt) from firing_rate_profile.py')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (alternative to --config)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar10, cifar100, imagenet, ...)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to dataset root (required for --evaluate)')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps (default 4)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n', type=int, default=2, help='N in N:M (default 2)')
    parser.add_argument('--m', type=int, default=4, help='M in N:M (default 4)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Compensation scale [0,1]. 1=full, 0=none. '
                             'Default 0.2 (empirically optimal for SNNs)')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'],
                        help='Layer name prefixes to skip (default: head)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after compensation')
    parser.add_argument('--output', type=str, default=None,
                        help='Save compensated model to this path')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders,
    )
    from models.neurons import reset_net

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

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

    # Load dense (original) weights
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dense_state = ckpt['model'] if 'model' in ckpt else ckpt
    print(f"Loaded dense checkpoint from {args.checkpoint}")

    # Load pruned model
    pruned_data = torch.load(args.pruned, map_location='cpu', weights_only=False)
    pruned_state = pruned_data['model']
    model.load_state_dict(pruned_state)
    model = model.to(device)
    print(f"Loaded pruned model from {args.pruned}  "
          f"(method={pruned_data.get('method')}, "
          f"params={pruned_data.get('params')})")

    # Load firing rates (channel-level)
    rates_data   = torch.load(args.rates, map_location='cpu', weights_only=False)
    firing_rates = {k: v for k, v in rates_data['firing_rates'].items()
                    if not k.endswith('.neuron')}
    print(f"Loaded firing rates from {args.rates}: {len(firing_rates)} neuron layers")

    # Apply compensation
    print(f"\n=== Applying element-wise {args.n}:{args.m} compensation ===")
    model = apply_compensation(
        model, dense_state, firing_rates,
        n=args.n, m=args.m, alpha=args.alpha, exclude_names=args.exclude,
    )

    # Save
    if args.output:
        torch.save({
            'model':   model.state_dict(),
            'method':  'compensation',
            'source':  args.pruned,
            'n': args.n, 'm': args.m,
        }, args.output)
        print(f"\nSaved compensated model to {args.output}")

    # Evaluate
    if args.evaluate:
        if args.data_root is None:
            parser.error('--data-root is required for --evaluate')

        _, val_loader = build_dataloaders(
            args.dataset, args.data_root, args.batch_size,
            img_size=ds_cfg['img_size'], num_workers=4,
        )

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                reset_net(model)
                correct += outputs.argmax(1).eq(targets).sum().item()
                total   += targets.size(0)

        acc = 100.0 * correct / total
        print(f"\nEvaluation accuracy: {acc:.2f}%  ({correct}/{total})")
