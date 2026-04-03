"""Shared utilities for sparse/ submodules.

Provides:
  - Module eligibility and weight reshaping helpers (moved from OBC/OBS)
  - Upstream-neuron lookup helpers
  - Spiking neuron parameter accessors (_get_inner_neuron, _get_vth, _set_vth, etc.)
  - Calibration data collectors (collect_firing_rates, collect_membrane_potentials)

This module must not import from sbc.py or snn_sbc.py
at module level to avoid circular imports.
"""

import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module eligibility and weight reshaping (moved from OBC.py / OBS.py)
# ---------------------------------------------------------------------------

def _is_eligible(module: nn.Module) -> bool:
    """Check if a module is eligible for pruning."""
    if isinstance(module, nn.Linear):
        return module.in_features >= 4
    if isinstance(module, nn.Conv2d):
        return module.groups == 1 and module.in_channels >= 1
    return False


def _get_weight_2d(module: nn.Module) -> tuple[torch.Tensor, Optional[tuple]]:
    """Get 2D weight view for TensorRT-compatible 2:4 pruning.

    TensorRT requires 2:4 sparsity along the C (input channel) dimension,
    independently for each output channel K and spatial position (R, S):
        for each (k, r, s): weights[k, c*4:(c+1)*4, r, s] has ≤ 2 non-zeros

    So we reshape Conv2d [K, C, R, S] → (K*R*S, C) where each row is one
    (k, r, s) combination, and the 2:4 pattern is along the C columns.

    Linear [K, C] is already correct — pruning is along C.

    Returns:
        (W_2d, orig_shape) where orig_shape is Conv2d weight shape (None for Linear).
    """
    W = module.weight.data
    if isinstance(module, nn.Conv2d):
        orig_shape = W.shape  # (K, C, R, S)
        K, C, R, S = orig_shape
        # Reshape to (K*R*S, C): each row = one (k,r,s), columns = input channels
        W_2d = W.permute(0, 2, 3, 1).reshape(K * R * S, C).contiguous()
        return W_2d, orig_shape
    return W, None


def _write_weight_back(module: nn.Module, W_2d: torch.Tensor, orig_shape: Optional[tuple]):
    """Write pruned 2D weight back to the module, reversing _get_weight_2d."""
    if orig_shape is not None:
        K, C, R, S = orig_shape
        # (K*R*S, C) → (K, R, S, C) → (K, C, R, S)
        W_back = W_2d.reshape(K, R, S, C).permute(0, 3, 1, 2).contiguous()
    else:
        W_back = W_2d
    module.weight.data.copy_(W_back.to(module.weight.device))


def _detect_neuron_fed_layers(
    model: nn.Module,
    eligible: OrderedDict,
    neuron_types: tuple,
) -> dict[str, bool]:
    """Detect which eligible layers receive input from a spiking neuron.

    Uses forward hooks to trace actual execution order. A Linear/Conv2d is
    "neuron-fed" if ANY spiking neuron has fired before it in the forward
    pass. Only the very first Linear/Conv2d layers (before any neuron has
    executed) are considered "dense-input" — these receive raw images.

    This correctly handles cross-container boundaries (e.g., LIF in
    patch_embed feeds q_linear in block.0.attn).
    """
    # Track execution order via hooks
    exec_order = []  # list of (name, is_neuron, is_eligible)
    hooks = []
    modules_dict = dict(model.named_modules())

    for name, mod in model.named_modules():
        is_neuron = isinstance(mod, neuron_types)
        is_elig = name in eligible

        if is_neuron or is_elig:
            def make_hook(n, is_n, is_e):
                def hook(module, inp, out):
                    exec_order.append((n, is_n, is_e))
                return hook
            hooks.append(mod.register_forward_hook(make_hook(name, is_neuron, is_elig)))

    # Single dummy forward pass to trace execution order
    device = next(model.parameters()).device
    # Infer input shape from first Conv2d or model config
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    in_ch = first_conv.in_channels if first_conv else 3
    # Try common image sizes
    from models.neurons import reset_net
    with torch.no_grad():
        try:
            model(torch.randn(1, in_ch, 32, 32, device=device))
        except Exception:
            try:
                model(torch.randn(1, in_ch, 224, 224, device=device))
            except Exception:
                pass
        reset_net(model)

    for h in hooks:
        h.remove()

    # Walk execution order: layers before any neuron fires are "dense-input"
    result = {}
    neuron_has_fired = False
    for name, is_neuron, is_elig in exec_order:
        if is_neuron:
            neuron_has_fired = True
        if is_elig:
            result[name] = neuron_has_fired

    # Any eligible layers not seen in exec_order → assume dense
    for name in eligible:
        if name not in result:
            result[name] = False

    return result


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


# ---------------------------------------------------------------------------
# Spiking neuron parameter accessors
# ---------------------------------------------------------------------------

def _get_neuron_types():
    """Lazy import to avoid circular dependencies.

    Returns tuple of all spiking neuron types across architectures:
      - models.neurons: MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron
      - models.msresnet: MSNeuron (MS-ResNet CIFAR arch)
    """
    from models.neurons import (
        MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron,
    )
    types = [MultiStepLIFNeuron, MultiStepIFNeuron, LIFNeuron, IFNeuron]
    try:
        from models.msresnet import MSNeuron
        types.append(MSNeuron)
    except ImportError:
        pass
    return tuple(types)


def _get_inner_neuron(module):
    """Get the inner LIFNeuron/IFNeuron from a MultiStep wrapper.

    For MSNeuron, returns the module itself (no inner wrapper).
    """
    from models.neurons import MultiStepLIFNeuron, MultiStepIFNeuron
    if isinstance(module, (MultiStepLIFNeuron, MultiStepIFNeuron)):
        return module.neuron
    return module


def _get_vth(neuron) -> float:
    """Get v_threshold/thresh as float (handles both LIF and MSNeuron)."""
    if hasattr(neuron, 'v_threshold'):
        vth = neuron.v_threshold
    elif hasattr(neuron, 'thresh'):
        vth = neuron.thresh
    else:
        raise AttributeError(f"No threshold attr on {type(neuron).__name__}")
    return vth.item() if isinstance(vth, torch.Tensor) else float(vth)


def _set_vth(neuron, value: float):
    """Set v_threshold/thresh (handles both LIF and MSNeuron)."""
    if hasattr(neuron, 'v_threshold'):
        attr = 'v_threshold'
    elif hasattr(neuron, 'thresh'):
        attr = 'thresh'
    else:
        raise AttributeError(f"No threshold attr on {type(neuron).__name__}")
    current = getattr(neuron, attr)
    if isinstance(current, nn.Parameter):
        current.data.fill_(value)
    elif isinstance(current, torch.Tensor):
        current.fill_(value)
    else:
        setattr(neuron, attr, value)


def _get_tau(neuron) -> Optional[float]:
    """Get tau as float (None if IFNeuron).

    Handles both standard LIF (has .tau) and MSNeuron (has .decay).
    For MSNeuron, converts decay to equivalent tau: tau = 1/(1-decay).
    """
    from models.neurons import IFNeuron
    if isinstance(neuron, IFNeuron):
        return None
    if hasattr(neuron, 'tau'):
        tau = neuron.tau
        return tau.item() if isinstance(tau, torch.Tensor) else float(tau)
    if hasattr(neuron, 'decay'):
        decay = neuron.decay
        d = decay.item() if isinstance(decay, torch.Tensor) else float(decay)
        return 1.0 / max(1.0 - d, 1e-6)
    return None


def _set_tau(neuron, value: float):
    """Set tau."""
    if isinstance(neuron.tau, nn.Parameter):
        neuron.tau.data.fill_(value)
    elif isinstance(neuron.tau, torch.Tensor):
        neuron.tau.fill_(value)
    else:
        neuron.tau = value


# ---------------------------------------------------------------------------
# Calibration data collectors
# ---------------------------------------------------------------------------

def collect_firing_rates(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 64,
) -> dict[str, float]:
    """Collect mean firing rate for each spiking neuron layer.

    Returns:
        {neuron_name: mean_firing_rate} across all spatial/channel dims.
    """
    from models.neurons import reset_net
    NEURON_TYPES = _get_neuron_types()

    rates = {}
    counts = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, NEURON_TYPES):
            continue
        if name.endswith('.neuron'):
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                rate = out.detach().float().mean().item()
                if n not in rates:
                    rates[n] = 0.0
                    counts[n] = 0
                rates[n] += rate
                counts[n] += 1
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            model(batch[0].to(device))
            reset_net(model)

    for h in hooks:
        h.remove()

    return {n: rates[n] / max(counts[n], 1) for n in rates}


def collect_membrane_potentials(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 32,
) -> dict[str, torch.Tensor]:
    """Collect membrane potential values (pre-threshold) for each neuron.

    Returns:
        {neuron_name: 1D tensor of sampled membrane potential values}
    """
    from models.neurons import reset_net
    NEURON_TYPES = _get_neuron_types()

    potentials = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, NEURON_TYPES):
            continue
        if name.endswith('.neuron'):
            continue

        inner = _get_inner_neuron(module)

        def make_hook(n, inner_neuron, mod_ref):
            def hook(mod, inp, out):
                # For standard LIF/IF: membrane potential stored in inner.v
                # For MSNeuron: stateless, reconstruct from input (pre-threshold)
                v = getattr(inner_neuron, 'v', None)
                if v is None:
                    # MSNeuron: input tensor IS the pre-threshold accumulation
                    # Use last timestep's input as proxy for membrane potential
                    x = inp[0]
                    if x.ndim >= 3 and x.shape[0] <= 16:  # (T, B, ...) format
                        v = x[-1]  # last timestep
                    else:
                        v = x
                if isinstance(v, torch.Tensor):
                    if n not in potentials:
                        potentials[n] = []
                    v_flat = v.detach().float().reshape(-1)
                    n_sample = max(v_flat.numel() // 10, 1000)
                    if v_flat.numel() > n_sample:
                        idx = torch.randperm(v_flat.numel(), device=v.device)[:n_sample]
                        v_flat = v_flat[idx]
                    potentials[n].append(v_flat.cpu())
            return hook

        hooks.append(module.register_forward_hook(make_hook(name, inner, module)))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            model(batch[0].to(device))
            reset_net(model)

    for h in hooks:
        h.remove()

    return {n: torch.cat(potentials[n]) for n in potentials if potentials[n]}
