"""
Activation-aware channel permutation for SNN N:M structured sparsity.

The core idea: profile per-channel spike firing rates, learn a channel
permutation P that groups low-firing channels together, so that when N:M
hardware prunes (M-N) of every M weight columns, it naturally hits the
already-silent channels -- causing negligible information loss.

Modules:
- compute_permutation_for_n_m: compute optimal channel ordering for N:M pruning
- ChannelPermutation: applies a fixed channel permutation via index gather
- PermutedLinear: wraps nn.Linear with input channel permutation + reordered weights
- PermutedConv2d: wraps nn.Conv2d with input channel permutation + reordered weights
- convert_model_with_permutation: replace Linear/Conv2d layers in a model
- measure_permutation_quality: measure alignment between pruned positions and low-fire channels
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional

from sparse.pruning import prune_n_m, prune_n_m_firing_aware
from sparse.utils import (
    _find_neuron_for_layer,
    _get_in_channels,
    _is_eligible_for_permutation,
    _get_upstream_entry,
)


class ChannelPermutation(nn.Module):
    """Applies a fixed channel permutation to the last dimension of a tensor.

    Stores a permutation index buffer and uses it to reorder channels.
    This module has no learnable parameters.

    Args:
        num_channels: Number of channels.
        permutation: (C,) LongTensor defining the permutation. If None,
            identity permutation is used.
    """

    def __init__(self, num_channels: int, permutation: torch.Tensor = None):
        super().__init__()
        if permutation is None:
            permutation = torch.arange(num_channels)
        assert permutation.shape == (num_channels,), \
            f"Permutation shape {permutation.shape} != ({num_channels},)"
        self.register_buffer('perm', permutation.long())
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute the last dimension: out[..., i] = x[..., perm[i]]."""
        return x[..., self.perm]

    def extra_repr(self) -> str:
        return f'num_channels={self.num_channels}'


class PermutedLinear(nn.Module):
    """Linear layer with input channel permutation for 2:4 sparsity alignment.

    Absorbs the permutation into the weight matrix so that no runtime
    permutation is needed during inference:
        y = (x @ W^T) is equivalent to (x_permuted @ W_permuted^T)
    where W_permuted[:, i] = W[:, perm[i]].

    After construction, call apply_2_4_pruning() to prune the reordered weights.

    Args:
        linear: The original nn.Linear module to wrap.
        permutation: (in_features,) LongTensor defining input channel permutation.
    """

    def __init__(self, linear: nn.Linear, permutation: torch.Tensor):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features

        assert permutation.shape == (in_features,), \
            f"Permutation shape {permutation.shape} != ({in_features},)"

        # Reorder weight columns according to permutation:
        # new_weight[:, i] = old_weight[:, perm[i]]
        # This means: if perm maps new_position -> old_channel,
        # then gathering columns by perm reorders them.
        with torch.no_grad():
            reordered_weight = linear.weight.data[:, permutation.long()]

        self.linear = nn.Linear(in_features, out_features,
                                bias=linear.bias is not None)
        self.linear.weight = nn.Parameter(reordered_weight)
        if linear.bias is not None:
            self.linear.bias = nn.Parameter(linear.bias.data.clone())

        self.register_buffer('perm', permutation.long())
        self._pruned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with implicit channel permutation.

        The permutation is absorbed into the weights, so we permute the
        input channels to match the reordered weight columns.
        """
        x_permuted = x[..., self.perm]
        return self.linear(x_permuted)

    def apply_n_m_pruning(self, n: int = 2, m: int = 4):
        """Apply N:M structured pruning to the (already permuted) weights.

        After permutation, low-firing channels are at positions N..M-1 of each
        group of M. The N:M pruning will keep the top-N by magnitude in each
        group, which should align with the high-firing positions 0..N-1.
        """
        with torch.no_grad():
            self.linear.weight.data = prune_n_m(self.linear.weight.data, n=n, m=m)
        self._pruned = True

    # Backward-compatible alias
    def apply_2_4_pruning(self):
        self.apply_n_m_pruning(n=2, m=4)

    def extra_repr(self) -> str:
        return (f'in_features={self.linear.in_features}, '
                f'out_features={self.linear.out_features}, '
                f'pruned={self._pruned}')


class PermutedConv2d(nn.Module):
    """Conv2d layer with input channel permutation for 2:4 sparsity alignment.

    Absorbs the permutation into the weight tensor so that no runtime
    permutation is needed during inference. For Conv2d with weight shape
    (C_out, C_in, Kh, Kw), column permutation acts on the C_in dimension:
        W_permuted[:, i, :, :] = W[:, perm[i], :, :]

    2:4 sparsity is applied along input channels using NHWC layout:
    weight is permuted to (C_out, Kh, Kw, C_in), pruned in groups of 4
    along the last dim, then permuted back. This matches NVIDIA Sparse
    Tensor Core layout.

    Depthwise convolutions (groups == in_channels) are skipped since each
    output channel only sees one input channel — permutation has no effect.

    Args:
        conv: The original nn.Conv2d module to wrap.
        permutation: (in_channels,) LongTensor defining input channel permutation.
    """

    def __init__(self, conv: nn.Conv2d, permutation: torch.Tensor):
        super().__init__()
        in_channels = conv.in_channels
        out_channels = conv.out_channels

        assert permutation.shape == (in_channels,), \
            f"Permutation shape {permutation.shape} != ({in_channels},)"
        assert conv.groups == 1 or conv.groups != in_channels, \
            "PermutedConv2d does not support depthwise convolutions"

        # Reorder weight along input channel dim:
        # new_weight[:, i, :, :] = old_weight[:, perm[i], :, :]
        with torch.no_grad():
            reordered_weight = conv.weight.data[:, permutation.long(), :, :]

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        self.conv.weight = nn.Parameter(reordered_weight)
        if conv.bias is not None:
            self.conv.bias = nn.Parameter(conv.bias.data.clone())

        self.register_buffer('perm', permutation.long())
        self._pruned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with implicit channel permutation.

        Input x has shape (..., C_in, H, W). Permute C_in dim to match
        the reordered weight columns.
        """
        # Channel dim is always dim -3 for Conv2d inputs: (..., C, H, W)
        # Handle both (B, C, H, W) and (T*B, C, H, W)
        x_permuted = x[:, self.perm, :, :]
        return self.conv(x_permuted)

    def apply_n_m_pruning(self, n: int = 2, m: int = 4):
        """Apply N:M structured pruning to the (already permuted) weights.

        Uses NHWC layout for pruning: (C_out, C_in, Kh, Kw) ->
        (C_out, Kh, Kw, C_in), prune in groups of M along C_in,
        then convert back to NCHW.
        """
        with torch.no_grad():
            w = self.conv.weight.data
            # (C_out, C_in, Kh, Kw) -> (C_out*Kh*Kw, C_in) via NHWC
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()
            shape_nhwc = w_nhwc.shape
            w_2d = w_nhwc.reshape(-1, w.shape[1])
            w_2d_pruned = prune_n_m(w_2d, n=n, m=m)
            w_nhwc_pruned = w_2d_pruned.reshape(shape_nhwc)
            self.conv.weight.data = w_nhwc_pruned.permute(0, 3, 1, 2).contiguous()
        self._pruned = True

    # Backward-compatible alias
    def apply_2_4_pruning(self):
        self.apply_n_m_pruning(n=2, m=4)

    def extra_repr(self) -> str:
        return (f'in_channels={self.conv.in_channels}, '
                f'out_channels={self.conv.out_channels}, '
                f'kernel_size={self.conv.kernel_size}, '
                f'pruned={self._pruned}')


def compute_permutation_for_n_m(
    rates: torch.Tensor,
    n: int = 2,
    m: int = 4,
) -> torch.Tensor:
    """Given per-channel firing rates (C,), compute optimal permutation for N:M pruning.

    Strategy: within each group of M contiguous channels, place the (M-N)
    lowest-firing channels at positions N..M-1 (which N:M pruning will zero).

    Algorithm:
    1. Sort channels by firing rate
    2. Split into high-fire (top N/M fraction) and low-fire (bottom (M-N)/M fraction)
    3. Interleave: [high*N, low*(M-N), high*N, low*(M-N), ...]

    Args:
        rates: (C,) tensor of per-channel firing rates.
        n: Non-zeros to keep per group (default 2).
        m: Group size (default 4).

    Returns:
        (C,) LongTensor - permutation indices such that
        permuted_data[..., i] = original_data[..., perm[i]]
    """
    C = rates.shape[0]
    num_prune = m - n
    sorted_idx = rates.argsort()  # (C,) low-to-high

    # Split: low-fire channels will be pruned, high-fire kept
    # We need num_groups * num_prune low-fire and num_groups * n high-fire
    num_groups = C // m
    remainder = C % m

    n_low = num_groups * num_prune
    low_fire = sorted_idx[:n_low]
    high_fire = sorted_idx[n_low:]

    perm = torch.zeros(C, dtype=torch.long)

    for g in range(num_groups):
        h_start = g * n
        l_start = g * num_prune
        # Place N high-fire channels first, then (M-N) low-fire
        for i in range(n):
            perm[g * m + i] = high_fire[h_start + i]
        for i in range(num_prune):
            perm[g * m + n + i] = low_fire[l_start + i]

    # Handle remainder channels
    if remainder > 0:
        base = num_groups * m
        h_used = num_groups * n
        l_used = num_groups * num_prune
        remaining_high = high_fire[h_used:]
        remaining_low = low_fire[l_used:]
        remaining = torch.cat([remaining_high, remaining_low])
        for r in range(remainder):
            perm[base + r] = remaining[r]

    return perm


# Backward-compatible alias
def compute_permutation_for_2_4(rates: torch.Tensor) -> torch.Tensor:
    """Compute optimal permutation for 2:4 pruning. See `compute_permutation_for_n_m`."""
    return compute_permutation_for_n_m(rates, n=2, m=4)


def _get_upstream_rates(
    model: nn.Module,
    layer_name: str,
    firing_rates: dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Get (C,) firing rates of the upstream neuron for a layer.

    Thin wrapper around _get_upstream_entry for channel-level rate dicts.
    """
    return _get_upstream_entry(model, layer_name, firing_rates,
                               get_channels=lambda r: r.shape[0])


def convert_model_with_permutation(
    model: nn.Module,
    firing_rates: dict[str, torch.Tensor],
    exclude_names: Optional[list[str]] = None,
    apply_pruning: bool = True,
) -> nn.Module:
    """Replace eligible Linear/Conv2d layers with permuted versions + optional 2:4 pruning.

    For each eligible layer:
    1. Find the upstream spiking neuron's per-channel firing rates
    2. Compute the optimal permutation (interleave high/low for 2:4)
    3. Replace with PermutedLinear or PermutedConv2d (absorbs permutation into weights)
    4. Optionally apply 2:4 pruning to the permuted weights

    Args:
        model: The SNN model to convert (modified in-place).
        firing_rates: {neuron_name: (C,) rates} from profiling.
        exclude_names: Module name prefixes to skip (e.g. ['head']).
        apply_pruning: Whether to apply 2:4 pruning after permutation.

    Returns:
        The modified model (same object).
    """
    # compute_permutation_for_n_m is defined in this module

    if exclude_names is None:
        exclude_names = ['head']

    conversion_stats = OrderedDict()
    replacements = []  # (parent_module, attr_name, new_module)

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
            conversion_stats[name] = {
                'converted': False,
                'type': layer_type,
                'reason': 'excluded',
                'shape': tuple(module.weight.shape),
            }
            continue

        # Check eligibility (min size, not depthwise)
        if not _is_eligible_for_permutation(module):
            in_ch = module.in_features if is_linear else module.in_channels
            reason = f'in_channels={in_ch} < 4'
            if is_conv and module.groups == module.in_channels and module.in_channels > 1:
                reason = 'depthwise conv (groups == in_channels)'
            conversion_stats[name] = {
                'converted': False,
                'type': layer_type,
                'reason': reason,
                'shape': tuple(module.weight.shape),
            }
            continue

        in_channels = _get_in_channels(module)

        # Find upstream firing rates
        upstream_rates = _get_upstream_rates(model, name, firing_rates)
        if upstream_rates is None:
            conversion_stats[name] = {
                'converted': False,
                'type': layer_type,
                'reason': 'no matching upstream neuron firing rates found',
                'shape': tuple(module.weight.shape),
            }
            continue

        if upstream_rates.shape[0] != in_channels:
            conversion_stats[name] = {
                'converted': False,
                'type': layer_type,
                'reason': (f'rate channels {upstream_rates.shape[0]} != '
                           f'in_channels {in_channels}'),
                'shape': tuple(module.weight.shape),
            }
            continue

        # Compute permutation
        perm = compute_permutation_for_n_m(upstream_rates)

        device = module.weight.device

        # Create permuted layer
        if is_linear:
            permuted = PermutedLinear(module, perm.to(device))
        else:
            permuted = PermutedConv2d(module, perm.to(device))

        if apply_pruning:
            permuted.apply_2_4_pruning()

        # Schedule replacement (can't modify model while iterating)
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, permuted))

        conversion_stats[name] = {
            'converted': True,
            'type': layer_type,
            'reason': 'success',
            'shape': tuple(module.weight.shape),
            'perm_channels': perm.shape[0],
            'pruned': apply_pruning,
        }

    # Apply replacements
    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    # Print summary
    converted = sum(1 for s in conversion_stats.values() if s['converted'])
    total = len(conversion_stats)
    print(f"\nChannel permutation conversion: {converted}/{total} layers converted")
    for name, stats in conversion_stats.items():
        status = "OK" if stats['converted'] else f"SKIP ({stats['reason']})"
        print(f"  {name} [{stats['type']}]: {stats['shape']} -> {status}")

    return model


def apply_firing_aware_pruning(
    model: nn.Module,
    firing_rates: dict[str, torch.Tensor],
    lam: float = 0.5,
    exclude_names: Optional[list[str]] = None,
) -> nn.Module:
    """Apply firing-rate-aware 2:4 pruning directly to model weights (in-place).

    No permutation is needed. Instead, the pruning score incorporates
    upstream firing rates so that low-firing channels are preferentially
    pruned even if their weight magnitude is not the smallest:

        score[i, c] = |W[i, c]| + lam * scale * r_c

    This directly addresses the mismatch between column-level permutation
    and row-level magnitude pruning by making pruning itself aware of
    channel activity.

    Args:
        model: The SNN model (modified in-place).
        firing_rates: {neuron_name: (C,) rates} from profiling.
        lam: Firing rate influence strength. 0 = pure magnitude pruning.
        exclude_names: Module name prefixes to skip (e.g. ['head']).

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
            }
            continue

        in_channels = _get_in_channels(module)

        # Find upstream firing rates
        upstream_rates = _get_upstream_rates(model, name, firing_rates)
        if upstream_rates is None:
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': 'no upstream rates found',
                'shape': tuple(module.weight.shape),
            }
            continue

        if upstream_rates.shape[0] != in_channels:
            pruning_stats[name] = {
                'pruned': False, 'type': layer_type,
                'reason': f'rate channels {upstream_rates.shape[0]} != in_channels {in_channels}',
                'shape': tuple(module.weight.shape),
            }
            continue

        # Apply firing-aware N:M pruning
        with torch.no_grad():
            w = module.weight.data
            if is_conv and w.ndim == 4:
                # NHWC layout for Conv2d
                w_nhwc = w.permute(0, 2, 3, 1).contiguous()
                shape_nhwc = w_nhwc.shape
                w_2d = w_nhwc.reshape(-1, w.shape[1])
                w_2d_pruned = prune_n_m_firing_aware(w_2d, upstream_rates, lam=lam)
                w_pruned = w_2d_pruned.reshape(shape_nhwc).permute(0, 3, 1, 2).contiguous()
            else:
                w_pruned = prune_n_m_firing_aware(w, upstream_rates, lam=lam)

            module.weight.data.copy_(w_pruned)

        # Compute sparsity
        total = w.numel()
        nonzero = module.weight.data.count_nonzero().item()
        sparsity = 1.0 - nonzero / max(total, 1)

        pruning_stats[name] = {
            'pruned': True, 'type': layer_type,
            'reason': 'success', 'shape': tuple(module.weight.shape),
            'sparsity': sparsity,
        }

    # Print summary
    pruned_count = sum(1 for s in pruning_stats.values() if s['pruned'])
    total_count = len(pruning_stats)
    print(f"\nFiring-aware 2:4 pruning (lam={lam}): {pruned_count}/{total_count} layers pruned")
    for name, stats in pruning_stats.items():
        if stats['pruned']:
            print(f"  {name} [{stats['type']}]: {stats['shape']} -> "
                  f"sparsity={stats['sparsity']:.3f}")
        else:
            print(f"  {name} [{stats['type']}]: {stats['shape']} -> "
                  f"SKIP ({stats['reason']})")

    return model


def _get_weight_2d_for_quality(module: nn.Module) -> tuple:
    """Extract a 2D view of weight along input channels for quality analysis.

    For Linear (out, in): use directly.
    For Conv2d (C_out, C_in, Kh, Kw): permute to NHWC -> (C_out*Kh*Kw, C_in).

    Returns:
        (weight_2d, in_channels, is_conv)
    """
    if isinstance(module, nn.Linear):
        return module.weight.data, module.in_features, False
    elif isinstance(module, nn.Conv2d):
        w = module.weight.data
        # (C_out, C_in, Kh, Kw) -> (C_out, Kh, Kw, C_in) -> (C_out*Kh*Kw, C_in)
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        w_2d = w_nhwc.reshape(-1, w.shape[1])
        return w_2d, w.shape[1], True
    raise TypeError(f"Unsupported: {type(module)}")


def measure_permutation_quality(
    model: nn.Module,
    firing_rates: dict[str, torch.Tensor],
    n: int = 2,
    m: int = 4,
) -> dict:
    """Measure how well N:M-pruned positions align with low-firing channels.

    Since N:M pruning operates per-row (each row independently keeps its
    top-N by magnitude in every group of M), alignment is measured per-row
    then averaged across all rows and groups.

    For each (row, group), check whether the (M-N) pruned positions fall on
    the (M-N) lowest-firing channels in that group. Alignment = fraction of
    pruned positions that correspond to low-firing channels.

    Also computes information loss: the weighted contribution of pruned
    channels, where contribution = firing_rate × |weight|.

    Args:
        model: Model (may or may not have Permuted* layers).
        firing_rates: {neuron_name: (C,) rates} from profiling.
        n: Non-zeros to keep per group (default 2).
        m: Group size (default 4).

    Returns:
        {layer_name: {alignment_score, expected_random, information_loss, ...}}
    """
    num_prune = m - n
    quality = OrderedDict()

    for name, module in model.named_modules():
        # Identify the inner layer and whether permutation was applied
        has_perm = False
        perm = None
        if isinstance(module, PermutedLinear):
            inner = module.linear
            has_perm = True
            perm = module.perm
            layer_type = 'Linear'
        elif isinstance(module, PermutedConv2d):
            inner = module.conv
            has_perm = True
            perm = module.perm
            layer_type = 'Conv2d'
        elif isinstance(module, nn.Linear):
            inner = module
            layer_type = 'Linear'
        elif isinstance(module, nn.Conv2d):
            inner = module
            layer_type = 'Conv2d'
        else:
            continue

        # Skip inner modules of Permuted* to avoid double counting
        # (e.g. block.0.attn.q_linear is PermutedLinear, and
        #  block.0.attn.q_linear.linear is its inner nn.Linear)
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        if parent_name:
            parent_mod = dict(model.named_modules()).get(parent_name)
            if isinstance(parent_mod, (PermutedLinear, PermutedConv2d)):
                continue

        # Find upstream rates
        upstream_rates = _get_upstream_rates(model, name, firing_rates)
        if upstream_rates is None:
            continue

        in_channels = _get_in_channels(module)
        if upstream_rates.shape[0] != in_channels:
            continue

        num_groups = in_channels // m
        if num_groups == 0:
            continue

        # Get 2D weight view: (rows, in_channels)
        weight_2d, _, _ = _get_weight_2d_for_quality(inner)

        aligned_ch = num_groups * m
        rows = weight_2d.shape[0]
        # (rows, num_groups, m)
        w_grouped = weight_2d[:, :aligned_ch].reshape(rows, num_groups, m)

        # Per-row pruning: a position is pruned if it's zero in that row
        is_pruned = (w_grouped == 0)  # (rows, num_groups, m)

        # Get the rates for the current channel ordering
        if has_perm:
            current_rates = upstream_rates[perm.cpu()]
        else:
            current_rates = upstream_rates

        # rates_grouped: (num_groups, m) — same across all rows
        rates_grouped = current_rates[:aligned_ch].reshape(num_groups, m)

        # In each group, which (M-N) positions have the lowest firing rates?
        _, low_idx = rates_grouped.topk(num_prune, dim=1, largest=False)
        low_mask = torch.zeros(num_groups, m, dtype=torch.bool)
        low_mask.scatter_(1, low_idx, True)
        # Broadcast to (rows, num_groups, 4)
        low_mask = low_mask.unsqueeze(0).expand_as(is_pruned)

        # Alignment: fraction of pruned positions that are low-firing channels
        total_pruned = is_pruned.sum().item()
        if total_pruned > 0:
            aligned = (is_pruned.cpu() & low_mask).sum().item()
            alignment = aligned / total_pruned
        else:
            alignment = float('nan')

        # Information loss: sum of rate_c * |w| at pruned positions
        # This measures actual signal energy removed by pruning
        rates_broadcast = rates_grouped.unsqueeze(0).expand(rows, -1, -1)
        weighted_contribution = rates_broadcast * w_grouped.abs().cpu()
        total_contribution = weighted_contribution.sum().item()
        pruned_contribution = weighted_contribution[is_pruned.cpu()].sum().item()
        relative_loss = pruned_contribution / max(total_contribution, 1e-8)

        # Sparsity achieved
        total_elements = rows * aligned_ch
        sparsity = total_pruned / max(total_elements, 1)

        quality[name] = {
            'alignment_score': alignment,
            'expected_random': num_prune / m,  # random chance of hitting low-fire
            'information_loss': pruned_contribution,
            'relative_information_loss': relative_loss,
            'sparsity': sparsity,
            'type': layer_type,
            'has_permutation': has_perm,
            'num_groups': num_groups,
            'in_channels': in_channels,
        }

    return quality


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply 2:4 structured pruning to an SNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--rates', type=str, required=True,
                        help='Path to firing rates file (.pt) from firing_rate_profile.py')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Path to dataset root (required for --evaluate)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU IDs')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation after conversion')
    parser.add_argument('--output', type=str, default=None,
                        help='Save converted model to this path')
    parser.add_argument('--exclude', type=str, nargs='*', default=['head'],
                        help='Module names to exclude from conversion')
    parser.add_argument('--method', type=str, default='firing_aware',
                        choices=['permutation', 'firing_aware'],
                        help='Pruning method: permutation (channel reorder + magnitude prune) '
                             'or firing_aware (magnitude + firing rate score)')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Firing rate influence for firing_aware method (0=pure magnitude)')
    parser.add_argument('--no-prune', action='store_true',
                        help='Skip 2:4 pruning (only apply permutation, permutation method only)')
    args = parser.parse_args()

    from tengine.utils import (
        load_model_config, build_model_from_config, build_model,
        get_dataset_config, build_dataloaders, accuracy,
    )
    from models.neurons import reset_net

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

    # Load firing rates
    rates_data = torch.load(args.rates, map_location='cpu', weights_only=False)
    firing_rates = rates_data['firing_rates']
    print(f"Loaded firing rates for {len(firing_rates)} neuron layers")

    # Measure quality before pruning
    print("\n--- Quality BEFORE pruning ---")
    quality_before = measure_permutation_quality(model, firing_rates)
    for name, q in quality_before.items():
        print(f"  {name} [{q['type']}]: alignment={q['alignment_score']:.3f}, "
              f"info_loss={q['relative_information_loss']:.4f}, "
              f"sparsity={q['sparsity']:.3f}")

    # Apply pruning
    if args.method == 'firing_aware':
        print(f"\n--- Applying firing-aware 2:4 pruning (lam={args.lam}) ---")
        model = apply_firing_aware_pruning(
            model, firing_rates,
            lam=args.lam,
            exclude_names=args.exclude,
        )
    else:
        print("\n--- Applying channel permutation ---")
        model = convert_model_with_permutation(
            model, firing_rates,
            exclude_names=args.exclude,
            apply_pruning=not args.no_prune,
        )

    # Measure quality after pruning
    print("\n--- Quality AFTER pruning ---")
    quality_after = measure_permutation_quality(model, firing_rates)
    for name, q in quality_after.items():
        print(f"  {name} [{q['type']}]: alignment={q['alignment_score']:.3f}, "
              f"info_loss={q['relative_information_loss']:.4f}, "
              f"sparsity={q['sparsity']:.3f}")

    # Save converted model
    if args.output:
        torch.save({
            'model': model.state_dict(),
            'method': args.method,
            'lam': args.lam if args.method == 'firing_aware' else None,
            'quality_before': quality_before,
            'quality_after': quality_after,
        }, args.output)
        print(f"\nSaved converted model to {args.output}")

    # Evaluate
    if args.evaluate:
        if args.data_root is None:
            parser.error('--data-root is required for --evaluate')

        _, val_loader = build_dataloaders(
            args.dataset, args.data_root, args.batch_size,
            img_size=ds_cfg['img_size'], num_workers=4,
        )

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
        print(f"\nEvaluation accuracy: {acc:.2f}% ({correct}/{total})")
