"""2:4 magnitude-based structured pruning for nn.Linear weights.

In 2:4 sparsity, every group of 4 consecutive elements in a row retains exactly
2 non-zero values (the 2 largest by magnitude). This pattern is required by
NVIDIA Sparse Tensor Cores for hardware-accelerated sparse matrix multiplication.
"""

import torch
import torch.nn as nn


def prune_2_4(weight: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 structured pruning to a weight tensor.

    For each row, groups elements in chunks of 4 and zeros the 2 smallest
    by magnitude, keeping the 2 largest.

    Args:
        weight: 2D tensor of shape (out_features, in_features).

    Returns:
        Pruned weight tensor with same shape. Every group of 4 consecutive
        elements along dim=1 has exactly 2 zeros.

    Raises:
        ValueError: If weight is not 2D.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.ndim}D")

    out_features, in_features = weight.shape

    # Pad in_features to multiple of 4 if needed
    pad_amount = (4 - in_features % 4) % 4
    if pad_amount > 0:
        padded = torch.zeros(out_features, in_features + pad_amount,
                             dtype=weight.dtype, device=weight.device)
        padded[:, :in_features] = weight
    else:
        padded = weight.clone()

    padded_in = padded.shape[1]

    # Reshape into groups of 4: (out_features, num_groups, 4)
    grouped = padded.reshape(out_features, padded_in // 4, 4)

    # Find the 2 smallest by magnitude in each group and zero them
    abs_grouped = grouped.abs()
    # topk with largest=False gives the 2 smallest
    _, smallest_indices = abs_grouped.topk(2, dim=2, largest=False)

    # Create mask: start with all ones, then zero out the 2 smallest
    mask = torch.ones_like(grouped)
    mask.scatter_(2, smallest_indices, 0.0)

    pruned = grouped * mask
    pruned = pruned.reshape(out_features, padded_in)

    # Remove padding
    if pad_amount > 0:
        pruned = pruned[:, :in_features]

    return pruned


def verify_2_4(weight: torch.Tensor) -> bool:
    """Verify that a weight tensor satisfies the 2:4 sparsity pattern.

    Checks that every group of 4 consecutive elements along dim=1 has
    exactly 2 zeros.

    Args:
        weight: 2D tensor to verify.

    Returns:
        True if the 2:4 pattern holds for all groups.
    """
    if weight.ndim != 2:
        return False

    out_features, in_features = weight.shape
    # Only check the portion that divides evenly by 4
    check_width = (in_features // 4) * 4
    if check_width == 0:
        return True

    grouped = weight[:, :check_width].reshape(out_features, check_width // 4, 4)
    zeros_per_group = (grouped == 0).sum(dim=2)
    return (zeros_per_group == 2).all().item()


def prune_model_linear(model: nn.Module, exclude_names: list = None) -> dict:
    """Apply 2:4 pruning to all nn.Linear layers in a model.

    Args:
        model: The model whose Linear weights will be pruned in-place.
        exclude_names: List of module name prefixes/exact names to skip.
            For example, ['head'] to exclude the classification head.

    Returns:
        Dict with per-layer stats:
            {layer_name: {'shape': tuple, 'total_params': int,
                          'zeros_before': int, 'zeros_after': int,
                          'density': float, 'skipped': bool, 'reason': str}}
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
            # Check dimension requirements: in_features should ideally be
            # multiple of 4 for clean 2:4 grouping. We handle non-multiples
            # via padding in prune_2_4, but warn about it.
            out_f, in_f = module.weight.shape
            if in_f < 4:
                layer_info['skipped'] = True
                layer_info['reason'] = f'in_features={in_f} < 4, too small for 2:4'
                skip = True

        if not skip:
            with torch.no_grad():
                module.weight.data = prune_2_4(module.weight.data)

        layer_info['zeros_after'] = (module.weight == 0).sum().item()
        nonzero = layer_info['total_params'] - layer_info['zeros_after']
        layer_info['density'] = nonzero / max(layer_info['total_params'], 1)

        stats[name] = layer_info

    return stats
