"""SparseGPT-style Optimal Brain Surgeon for N:M structured pruning of SNNs.

Training-free, calibration-only approach:
  1. Collect per-layer Hessians H = X·Xᵀ from ~128 forward passes (no labels)
  2. Use second-order OBS importance w²/[H⁻¹]_{pp} to select N:M masks
  3. Apply closed-form OBS weight updates that optimally compensate survivors
  4. Compensation propagates across M-groups via sequential column processing

Reference:
    Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately
    Pruned in One-Shot", ICML 2023.  Adapted here for SNN binary activations.
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional

from sparse.pruning import verify_n_m
from sparse.utils import _is_eligible_for_permutation, _get_in_channels


# ---------------------------------------------------------------------------
# Hessian collection
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 128,
    exclude_names: Optional[list] = None,
) -> dict[str, torch.Tensor]:
    """Collect per-layer Hessians H = X·Xᵀ / n from calibration data.

    Registers forward hooks on eligible Linear/Conv2d layers to accumulate
    the input covariance matrix incrementally (never stores full X).

    For Conv2d: H is (C_in, C_in), aggregated over spatial positions.
    For Linear: H is (in_features, in_features).

    Args:
        model:         SNN model (eval mode, on device).
        dataloader:    Calibration data (no labels needed, only forward pass).
        device:        Compute device.
        max_batches:   Number of calibration batches (default 128).
        exclude_names: Layer name prefixes to skip (default ['head']).

    Returns:
        {layer_name: H} where H is (d_in, d_in) on *device*.
    """
    from models.neurons import reset_net

    if exclude_names is None:
        exclude_names = ['head']

    # Identify eligible layers
    eligible = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_permutation(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            continue
        eligible[name] = module

    # Incremental accumulation: H += x @ x^T,  n_samples += x.shape[1]
    hessians: dict[str, torch.Tensor] = {}
    n_samples: dict[str, int] = {}
    hooks = []

    def _make_hook(name, mod):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if isinstance(mod, nn.Conv2d):
                # x: (B*T, C_in, H, W) → (C_in, B*T*H*W)
                b, c, h, w = x.shape
                x = x.permute(1, 0, 2, 3).reshape(c, -1)
            elif isinstance(mod, nn.Linear):
                # x: (..., in_features) → (in_features, n)
                x = x.reshape(-1, x.shape[-1]).T
            else:
                return
            n = x.shape[1]
            if name not in hessians:
                d = x.shape[0]
                hessians[name] = torch.zeros(d, d, device=x.device, dtype=torch.float32)
                n_samples[name] = 0
            hessians[name].addmm_(x, x.T)
            n_samples[name] += n
        return hook_fn

    for name, mod in eligible.items():
        hooks.append(mod.register_forward_hook(_make_hook(name, mod)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            model(images)
            reset_net(model)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Profiled {batch_idx + 1}/{max_batches} batches")

    for h in hooks:
        h.remove()

    # Normalize
    for name in hessians:
        if n_samples[name] > 0:
            hessians[name] /= n_samples[name]

    print(f"Collected Hessians for {len(hessians)} layers "
          f"({min(batch_idx + 1, max_batches)} batches)")
    return hessians


# ---------------------------------------------------------------------------
# Per-layer OBS N:M pruning
# ---------------------------------------------------------------------------

def obs_prune_layer(
    W_2d: torch.Tensor,
    H: torch.Tensor,
    n: int = 2,
    m: int = 4,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """SparseGPT-style OBS N:M pruning for a single layer.

    Processes columns left-to-right in M-groups:
      1. Score weights in the group: s = w² / [H⁻¹]_{pp}
      2. Keep top-N per row (prune M−N lowest)
      3. Apply OBS compensation: δw = −w_p / [H⁻¹]_{pp} · H⁻¹[p, :]
      4. Compensation propagates to all later columns

    Args:
        W_2d:     (rows, cols) weight matrix (NHWC-reshaped for Conv2d).
        H:        (cols, cols) Hessian = X·Xᵀ / n_samples.
        n, m:     N:M sparsity (default 2:4).
        percdamp: Dampening as fraction of mean(diag(H)).
                  Critical for SNNs — sparse binary activations make H
                  near-singular.  Default 0.01.

    Returns:
        (rows, cols) weight tensor with N:M pattern and OBS compensation.
    """
    rows, cols = W_2d.shape
    num_groups = cols // m
    aligned = num_groups * m

    if num_groups == 0:
        return W_2d.clone()

    W = W_2d[:, :aligned].clone().float()
    Hs = H[:aligned, :aligned].clone().float()

    # --- Dampening ---
    damp = percdamp * Hs.diag().mean()
    idx = torch.arange(aligned, device=Hs.device)
    Hs[idx, idx] += damp

    # --- H⁻¹ via Cholesky ---
    dead = Hs.diag() == 0
    Hs[dead, dead] = 1          # avoid zero-diagonal entries
    try:
        L = torch.linalg.cholesky(Hs)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        # Aggressive fallback dampening
        Hs[idx, idx] += 10 * damp
        L = torch.linalg.cholesky(Hs)
        H_inv = torch.cholesky_inverse(L)
    H_inv[dead, :] = 0
    H_inv[:, dead] = 0

    # --- Process M-groups sequentially ---
    for g in range(num_groups):
        j0, j1 = g * m, (g + 1) * m

        # Score weights in this group (second-order importance)
        group_w = W[:, j0:j1]                                     # (rows, m)
        h_diag = H_inv[j0:j1, j0:j1].diag().clamp(min=1e-10)     # (m,)
        scores = group_w ** 2 / h_diag.unsqueeze(0)                # (rows, m)

        # Keep top-n per row; prune the rest
        _, keep = scores.topk(n, dim=1)                # (rows, n)
        prune = torch.ones(rows, m, dtype=torch.bool, device=W.device)
        prune.scatter_(1, keep, False)

        # OBS compensation, column by column within the group
        for k in range(m):
            col = j0 + k
            pruned_rows = prune[:, k]          # (rows,) bool
            if not pruned_rows.any():
                continue

            q = W[pruned_rows, col].clone()
            W[pruned_rows, col] = 0.0

            # Update all later columns for pruned rows:
            # δw[:, col+1:] = −(q / H_inv[col,col]) * H_inv[col, col+1:]
            if col + 1 < aligned:
                scale = q / H_inv[col, col].clamp(min=1e-10)       # (n_pruned,)
                W[pruned_rows, col + 1:] -= (
                    scale.unsqueeze(1) * H_inv[col, col + 1:aligned].unsqueeze(0)
                )

    # Reassemble with unaligned tail (left untouched)
    W_out = W_2d.clone()
    W_out[:, :aligned] = W
    return W_out


# ---------------------------------------------------------------------------
# Model-level OBS pruning
# ---------------------------------------------------------------------------

def apply_obs_pruning(
    model: nn.Module,
    hessians: dict[str, torch.Tensor],
    n: int = 2,
    m: int = 4,
    percdamp: float = 0.01,
    exclude_names: Optional[list] = None,
) -> nn.Module:
    """Apply OBS N:M pruning to all eligible layers in-place.

    Args:
        model:         SNN model (dense weights). Modified in-place.
        hessians:      {layer_name: H} from collect_hessians.
        n, m:          N:M parameters (default 2:4).
        percdamp:      Hessian dampening fraction.
        exclude_names: Layer name prefixes to skip (default ['head']).

    Returns:
        The modified model (same object).
    """
    if exclude_names is None:
        exclude_names = ['head']

    print(f"\n=== OBS {n}:{m} pruning (percdamp={percdamp}) ===\n")
    print(f"{'Layer':<42} {'Type':>6} {'shape':>14} {'rel_err':>8}  ok")
    print("-" * 80)

    stats = OrderedDict()
    skipped = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _is_eligible_for_permutation(module):
            continue
        if any(name == e or name.startswith(e + '.') for e in exclude_names):
            skipped.append(name)
            continue
        if name not in hessians:
            skipped.append(name)
            continue

        H = hessians[name]
        in_ch = _get_in_channels(module)
        is_conv = isinstance(module, nn.Conv2d)

        W = module.weight.data.cpu().float()
        if is_conv:
            orig_shape = W.shape
            W_2d = W.permute(0, 2, 3, 1).reshape(-1, in_ch).contiguous()
        else:
            W_2d = W
            orig_shape = None

        W_orig_2d = W_2d.clone()

        # --- OBS prune ---
        W_pruned = obs_prune_layer(W_2d, H.cpu(), n=n, m=m, percdamp=percdamp)

        # --- Metrics ---
        # Relative output error using H diagonal as activation magnitude proxy
        r_proxy = H.cpu().diag().sqrt().clamp(min=1e-8)[:in_ch]
        err = ((W_pruned - W_orig_2d) @ r_proxy).norm().item()
        ref = (W_orig_2d @ r_proxy).norm().item()
        rel_err = err / (ref + 1e-8)

        pattern_ok = verify_n_m(W_pruned, n=n, m=m)
        ltype = 'Conv2d' if is_conv else 'Linear'
        r, c = W_2d.shape
        ok_str = '✓' if pattern_ok else '✗'
        print(f"  {name:<40} {ltype:>6} {f'({r},{c})':>14} "
              f"{rel_err:>8.4f}  {ok_str}")

        # --- Write back ---
        if is_conv:
            C_out, C_in, Kh, Kw = orig_shape
            W_back = (W_pruned.reshape(C_out, Kh, Kw, C_in)
                      .permute(0, 3, 1, 2).contiguous())
        else:
            W_back = W_pruned
        module.weight.data.copy_(W_back)

        stats[name] = {'rel_err': rel_err, 'pattern_ok': pattern_ok}

    # --- Summary ---
    if stats:
        mean_err = sum(s['rel_err'] for s in stats.values()) / len(stats)
        bad = [n_ for n_, s in stats.items() if not s['pattern_ok']]
        print(f"\n  Pruned {len(stats)} layers  |  mean rel_error: {mean_err:.4f}")
        if bad:
            print(f"  WARNING: N:M pattern broken in {len(bad)} layer(s): {bad}")
        else:
            print(f"  N:M pattern preserved in all {len(stats)} layers ✓")
    if skipped:
        print(f"  Skipped {len(skipped)} layer(s): {', '.join(skipped)}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OBS-based N:M structured pruning for SNNs (training-free)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Dense model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. ms_resnet34)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (cifar10, cifar100, imagenet, ...)')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps (default 4)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='Hessian dampening fraction (default 0.01). '
                             'Increase for SNNs with very low firing rates.')
    parser.add_argument('--calib-batches', type=int, default=128,
                        help='Number of calibration batches for Hessian '
                             'estimation (default 128)')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'],
                        help='Layer name prefixes to skip (default: head)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after pruning')
    parser.add_argument('--output', type=str, default=None,
                        help='Save pruned model to this path')
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
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T)
    else:
        parser.error('Must specify --config or --model')

    # Load dense checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    dense_state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(dense_state)
    model = model.to(device)
    print(f"Loaded dense checkpoint from {args.checkpoint}")

    # Build calibration dataloader (training set for Hessian estimation)
    train_loader, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=4,
    )

    # Step 1: Collect Hessians
    print(f"\n--- Collecting Hessians ({args.calib_batches} batches) ---")
    hessians = collect_hessians(
        model, train_loader, device,
        max_batches=args.calib_batches, exclude_names=args.exclude,
    )

    # Save original state for later comparison
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Step 2: Apply OBS pruning
    model = apply_obs_pruning(
        model, hessians,
        n=args.n, m=args.m, percdamp=args.percdamp,
        exclude_names=args.exclude,
    )

    # Save
    if args.output:
        torch.save({
            'model': model.state_dict(),
            'method': 'obs',
            'original_state': original_state,
            'params': {
                'n': args.n, 'm': args.m,
                'percdamp': args.percdamp,
                'calib_batches': args.calib_batches,
            },
        }, args.output)
        print(f"\nSaved OBS-pruned model to {args.output}")

    # Evaluate
    if args.evaluate:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                reset_net(model)
                correct += outputs.argmax(1).eq(targets).sum().item()
                total += targets.size(0)

        acc = 100.0 * correct / total
        print(f"\nEvaluation accuracy: {acc:.2f}%  ({correct}/{total})")
