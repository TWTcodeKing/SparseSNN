"""Shared utilities for sparse/ submodules.

Provides upstream-neuron lookup helpers used by both permutation.py and
fr_prune.py. This module must not import from permutation.py or fr_prune.py
at module level to avoid circular imports.
"""

import re
from typing import Optional

import torch
import torch.nn as nn


def _find_neuron_for_layer(model: nn.Module, layer_name: str) -> Optional[str]:
    """Find the upstream spiking neuron whose output feeds into a given layer.

    Handles naming conventions for both Linear and Conv2d layers across
    all SNN architectures in this codebase:

    Transformer (Spikformer/QKFormer/MaxFormer) naming patterns:
        *_linear / *_conv -> upstream *_lif neuron
        proj_linear -> attn_lif (upstream)
        fc2_linear / fc2_conv -> fc1_lif
        proj_conv1 -> proj_lif (upstream of proj_conv1 is proj_lif)

    ResNet (SEW/MS) naming patterns:
        *.conv1 (in SeqToANNContainer) -> *.sn1 is downstream, not upstream
        For ResNets: *.conv_bn1 -> upstream is *.sn1 (neuron fires first)
        *.conv_bn2 -> upstream is *.sn2

    Args:
        model: The model.
        layer_name: Full name of the Linear or Conv2d module.

    Returns:
        Name of the upstream neuron, or None if not found.
    """
    modules_dict = dict(model.named_modules())

    candidates = []

    # Pattern: *_linear -> *_lif, *_conv -> *_lif
    for suffix, replacement in [('_linear', '_lif'), ('_conv', '_lif')]:
        if suffix in layer_name:
            # proj_linear -> attn_lif
            if layer_name.endswith('proj_linear') or layer_name.endswith('proj_conv'):
                candidates.append(layer_name.rsplit('proj', 1)[0] + 'attn_lif')
            # fc2_* -> fc1_lif
            if 'fc2' in layer_name:
                candidates.append(layer_name.replace('fc2' + suffix, 'fc1_lif'))
            # Generic: replace suffix with _lif
            candidates.append(layer_name.rsplit(suffix, 1)[0] + '_lif')

    # ResNet patterns: conv_bn1 -> sn1 (in MS-ResNet, neuron is upstream of conv)
    if 'conv_bn' in layer_name:
        m = re.search(r'conv_bn(\d+)', layer_name)
        if m:
            candidates.append(layer_name.rsplit('conv_bn', 1)[0] + 'sn' + m.group(1))

    # Numbered conv patterns: proj_conv1 -> proj_lif, proj_conv2 -> proj_lif1
    m = re.search(r'_conv(\d+)$', layer_name)
    if m:
        num = int(m.group(1))
        base = layer_name.rsplit('_conv', 1)[0]
        if num == 0:
            candidates.append(base + '_lif')
        else:
            # proj_conv1 -> proj_lif, proj_conv2 -> proj_lif1, proj_conv3 -> proj_lif2
            candidates.append(base + '_lif' + (str(num - 1) if num > 1 else ''))

    # Depthwise conv patterns: conv -> conv_neuron, dwconv -> dwconv_neuron
    if layer_name.endswith('conv') or layer_name.endswith('dwconv'):
        candidates.append(layer_name + '_neuron')

    for cand in candidates:
        if cand in modules_dict:
            return cand

    return None


def _get_in_channels(module: nn.Module) -> int:
    """Get the input channel count for a Linear, Conv2d, or Permuted* module.

    Uses duck typing for Permuted* wrappers to avoid circular imports with
    permutation.py (PermutedLinear wraps an nn.Linear as .linear;
    PermutedConv2d wraps an nn.Conv2d as .conv).
    """
    # Duck-type PermutedLinear: has .linear attribute that is nn.Linear
    if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear):
        return module.linear.in_features
    # Duck-type PermutedConv2d: has .conv attribute that is nn.Conv2d
    if hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d):
        return module.conv.in_channels
    if isinstance(module, nn.Linear):
        return module.in_features
    if isinstance(module, nn.Conv2d):
        return module.in_channels
    raise TypeError(f"Unsupported module type: {type(module)}")


def _is_eligible_for_permutation(module: nn.Module) -> bool:
    """Check if a module is eligible for channel permutation.

    Eligible: nn.Linear with in_features >= 4, or nn.Conv2d with
    in_channels >= 4 and groups == 1 (not depthwise).
    """
    if isinstance(module, nn.Linear):
        return module.in_features >= 4
    if isinstance(module, nn.Conv2d):
        # Skip depthwise convolutions: each output channel sees only one
        # input channel, so permutation has no effect on 2:4 pruning.
        if module.groups == module.in_channels and module.in_channels > 1:
            return False
        return module.in_channels >= 4
    return False


def _get_upstream_entry(
    model: nn.Module,
    layer_name: str,
    entries: dict,
    get_channels,
):
    """Generic upstream neuron lookup shared by permutation and fr_prune.

    Tries three strategies to find a matching entry in `entries`:
    1. Exact upstream neuron via naming patterns (_find_neuron_for_layer)
    2. Deepest neuron sharing the same parent prefix with matching channel count
    3. Any entry with matching channel count (fallback)

    Args:
        model: The model.
        layer_name: Full name of the Linear or Conv2d module.
        entries: Dict mapping neuron names to values (Tensor or dict).
        get_channels: Callable(entry) -> int that extracts channel count from
            a value in `entries`. E.g. ``lambda r: r.shape[0]`` for Tensor
            values, or ``lambda e: e['channel_mean'].shape[0]`` for dict values.

    Returns:
        Matched entry value, or None if not found.
    """
    modules_dict = dict(model.named_modules())
    in_channels = _get_in_channels(modules_dict[layer_name])

    # Strategy 1: known naming patterns
    neuron_name = _find_neuron_for_layer(model, layer_name)
    if neuron_name and neuron_name in entries:
        entry = entries[neuron_name]
        if get_channels(entry) == in_channels:
            return entry

    # Strategy 2: deepest neuron sharing the same parent prefix
    parts = layer_name.rsplit('.', 1)
    parent_prefix = parts[0] if len(parts) == 2 else ''

    best_match = None
    best_depth = -1
    for nname, entry in entries.items():
        if get_channels(entry) != in_channels:
            continue
        if parent_prefix and nname.startswith(parent_prefix):
            depth = len(nname.split('.'))
            if depth > best_depth:
                best_depth = depth
                best_match = nname

    if best_match:
        return entries[best_match]

    # Strategy 3: any entry with matching channel count (less reliable)
    for entry in entries.values():
        if get_channels(entry) == in_channels:
            return entry

    return None
