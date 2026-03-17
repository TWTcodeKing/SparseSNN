"""Canonical N:M structured pruning primitives.

All N:M pruning operations across the codebase should import from this module.
In N:M sparsity, every group of M consecutive elements retains exactly N
non-zero values (the N largest by magnitude).  The special case N=2, M=4 is
required by NVIDIA Sparse Tensor Cores for hardware-accelerated inference.

Provides:
    prune_n_m          — core N:M magnitude pruning on arbitrary tensors
    prune_n_m_firing_aware — N:M pruning biased by upstream firing rates
    verify_n_m         — verify a tensor satisfies the N:M pattern
    prune_model_linear — apply N:M pruning to all nn.Linear in a model

Convenience aliases (default N=2, M=4):
    prune_2_4          = partial(prune_n_m, n=2, m=4)
    project_to_2_4     = prune_2_4   (backward compat with structured_training)
"""

import torch
import torch.nn as nn
from functools import partial


# ---------------------------------------------------------------------------
# Core N:M magnitude pruning
# ---------------------------------------------------------------------------

def prune_n_m(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """Apply N:M structured pruning to a weight tensor.

    For each group of M contiguous elements along dim=-1, zeros the (M-N)
    elements with the smallest magnitude, keeping the N largest.  Works on
    tensors of any rank >= 1 (flattens leading dims, then restores shape).

    Args:
        weight: Tensor with at least 1 dimension.  Pruning is applied along
            the last dimension (dim=-1).
        n: Number of non-zero elements to keep per group.
        m: Group size.  Must satisfy 0 < n < m.

    Returns:
        Pruned tensor with same shape and dtype as input.
    """
    assert 0 < n < m, f"Need 0 < n < m, got n={n}, m={m}"

    orig_shape = weight.shape
    orig_dtype = weight.dtype

    # Flatten all leading dims into one so we always work on a 2D view
    w2d = weight.reshape(-1, orig_shape[-1])
    rows, cols = w2d.shape

    # Pad to multiple of m along cols
    pad = (m - cols % m) % m
    if pad > 0:
        w2d = torch.cat(
            [w2d, torch.zeros(rows, pad, dtype=orig_dtype, device=w2d.device)],
            dim=1,
        )

    padded_cols = w2d.shape[1]
    grouped = w2d.reshape(rows, padded_cols // m, m)

    # Find the (m-n) smallest by magnitude and build a binary mask
    num_prune = m - n
    _, small_idx = grouped.abs().topk(num_prune, dim=2, largest=False)
    mask = torch.ones_like(grouped)
    mask.scatter_(2, small_idx, 0.0)

    projected = (grouped * mask).reshape(rows, padded_cols)

    # Remove padding
    if pad > 0:
        projected = projected[:, :cols]

    return projected.reshape(orig_shape)


# Convenience aliases for the 2:4 case
prune_2_4 = partial(prune_n_m, n=2, m=4)
prune_2_4.__doc__ = "Apply 2:4 structured pruning.  See `prune_n_m` for details."

# Backward-compatible alias used by structured_training.py
project_to_2_4 = prune_2_4


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_n_m(weight: torch.Tensor, n: int = 2, m: int = 4) -> bool:
    """Verify that a 2D weight tensor satisfies the N:M sparsity pattern.

    Checks that every group of M consecutive elements along dim=1 has
    exactly (M - N) zeros.

    Args:
        weight: 2D tensor to verify.
        n: Expected non-zeros per group.
        m: Group size.

    Returns:
        True if the N:M pattern holds for all groups.
    """
    if weight.ndim != 2:
        return False

    out_features, in_features = weight.shape
    check_width = (in_features // m) * m
    if check_width == 0:
        return True

    grouped = weight[:, :check_width].reshape(out_features, check_width // m, m)
    zeros_per_group = (grouped == 0).sum(dim=2)
    return (zeros_per_group == m - n).all().item()


verify_2_4 = partial(verify_n_m, n=2, m=4)
verify_2_4.__doc__ = "Verify 2:4 sparsity pattern.  See `verify_n_m` for details."


# ---------------------------------------------------------------------------
# Model-level pruning
# ---------------------------------------------------------------------------

def prune_model_linear(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
    exclude_names: list = None,
) -> dict:
    """Apply N:M pruning to all nn.Linear layers in a model (in-place).

    Args:
        model: The model whose Linear weights will be pruned in-place.
        n: Non-zeros to keep per group.
        m: Group size.
        exclude_names: List of module name prefixes/exact names to skip.

    Returns:
        Dict with per-layer stats:
            {layer_name: {'shape', 'total_params', 'zeros_before',
                          'zeros_after', 'density', 'skipped', 'reason'}}
    """
    if exclude_names is None:
        exclude_names = []

    stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_info = {
            'shape': tuple(module.weight.shape),
            'total_params': module.weight.numel(),
            'zeros_before': (module.weight == 0).sum().item(),
            'skipped': False,
            'reason': '',
        }

        # Check exclusion
        skip = False
        for excl in exclude_names:
            if name == excl or name.startswith(excl + '.'):
                skip = True
                layer_info['skipped'] = True
                layer_info['reason'] = f'excluded by name match: {excl}'
                break

        if not skip:
            out_f, in_f = module.weight.shape
            if in_f < m:
                layer_info['skipped'] = True
                layer_info['reason'] = f'in_features={in_f} < {m}, too small for {n}:{m}'
                skip = True

        if not skip:
            with torch.no_grad():
                module.weight.data = prune_n_m(module.weight.data, n=n, m=m)

        layer_info['zeros_after'] = (module.weight == 0).sum().item()
        nonzero = layer_info['total_params'] - layer_info['zeros_after']
        layer_info['density'] = nonzero / max(layer_info['total_params'], 1)

        stats[name] = layer_info

    return stats


# ---------------------------------------------------------------------------
# Firing-rate-aware N:M pruning
# ---------------------------------------------------------------------------

def prune_n_m_firing_aware(
    weight_2d: torch.Tensor,
    rates: torch.Tensor,
    n: int = 2,
    m: int = 4,
    lam: float = 0.5,
) -> torch.Tensor:
    """Apply N:M structured pruning biased by upstream firing rates.

    Instead of keeping top-N by magnitude alone, uses a composite score:
        score[i, c] = |W[i, c]| + lam * scale * r_c

    where scale = mean(|W|) / (mean(r) + eps) auto-normalizes the firing
    rate contribution to be comparable with weight magnitudes.

    High-firing channels get a score bonus, making them less likely to be
    pruned.  Low-firing channels get less bonus, making them more likely to
    be pruned — even if their weight magnitude is not the smallest.

    Args:
        weight_2d: (rows, cols) weight tensor.
        rates: (cols,) per-channel firing rates from upstream neuron.
        n: Non-zeros to keep per group.
        m: Group size.
        lam: Controls firing rate influence.  0 = pure magnitude pruning,
            larger values = stronger preference to keep high-firing channels.

    Returns:
        Pruned weight tensor of the same shape.
    """
    rows, cols = weight_2d.shape
    num_groups = cols // m
    if num_groups == 0:
        return weight_2d.clone()

    aligned_cols = num_groups * m

    # Auto-scale: make firing rate contribution comparable to weight magnitudes
    w_abs = weight_2d[:, :aligned_cols].abs()
    w_mean = w_abs.mean().item()
    r_mean = rates[:aligned_cols].mean().item()
    scale = w_mean / (r_mean + 1e-8)

    # Build score matrix: (rows, aligned_cols)
    rate_bonus = lam * scale * rates[:aligned_cols].unsqueeze(0).to(weight_2d.device)
    scores = w_abs + rate_bonus

    # Group and keep top-N by score per group
    scores_grouped = scores.reshape(rows, num_groups, m)
    _, top_idx = scores_grouped.topk(n, dim=2)
    mask = torch.zeros_like(scores_grouped)
    mask.scatter_(2, top_idx, 1.0)

    # Apply mask to original weights (not scores)
    w_grouped = weight_2d[:, :aligned_cols].reshape(rows, num_groups, m)
    result = weight_2d.clone()
    result[:, :aligned_cols] = (w_grouped * mask).reshape(rows, aligned_cols)
    return result
