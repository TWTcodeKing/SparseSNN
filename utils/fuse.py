"""Layer fusion utilities for SNN models.

Provides BN fusion: absorb BatchNorm parameters into preceding Conv2d/Linear
layers, replacing BN with Identity. This eliminates BN's dependence on
running statistics, which become stale after weight pruning.

Usage:
    from utils.fuse import fuse_bn
    model, n_fused = fuse_bn(model)
"""

import re
import torch
import torch.nn as nn


def fuse_bn(model: nn.Module) -> tuple[nn.Module, int]:
    """Fuse BatchNorm layers into preceding Conv2d/Linear layers.

    For each BN layer, finds the matching Conv/Linear by naming convention
    (*_bn → *_conv or *_linear) and absorbs BN parameters:
        W_fused = γ/√(σ²+ε) · W
        b_fused = γ/√(σ²+ε) · (b - μ) + β

    The BN layer is then replaced with nn.Identity().

    Naming conventions supported:
        proj_bn   → proj_conv / proj_linear
        q_bn      → q_linear
        fc1_bn    → fc1_linear
        proj_bn1  → proj_conv1
        conv_bn1  → conv_bn1 (ResNet: conv_bn1.module.0 matched separately)

    Args:
        model: Model to fuse (modified in-place).

    Returns:
        (model, n_fused) — the modified model and count of fused BN layers.
    """
    modules_dict = dict(model.named_modules())
    fused = 0

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue

        # Find matching Conv/Linear by naming pattern
        layer = _find_preceding_layer(name, modules_dict)
        if layer is None:
            continue

        # Fuse BN into the layer
        with torch.no_grad():
            scale = mod.weight / torch.sqrt(mod.running_var + mod.eps)

            if isinstance(layer, nn.Conv2d):
                layer.weight.data = scale.view(-1, 1, 1, 1) * layer.weight.data
            elif isinstance(layer, nn.Linear):
                layer.weight.data = scale.view(-1, 1) * layer.weight.data

            if layer.bias is not None:
                layer.bias.data = scale * (layer.bias.data - mod.running_mean) + mod.bias
            else:
                layer.bias = nn.Parameter(
                    scale * (-mod.running_mean) + mod.bias
                )

        # Replace BN with Identity
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = modules_dict[parts[0]]
            setattr(parent, parts[1], nn.Identity())
        else:
            setattr(model, name, nn.Identity())
        fused += 1

    return model, fused


def _find_preceding_layer(
    bn_name: str,
    modules_dict: dict,
) -> nn.Module:
    """Find the Conv2d/Linear that precedes a BN layer by naming convention."""
    # Extract base and optional number: proj_bn → (proj, ''), proj_bn1 → (proj, '1')
    m = re.match(r'(.+)_bn(\d*)$', bn_name)
    if m is None:
        # Try ResNet pattern: *.conv_bn1.module.1 → *.conv_bn1.module.0
        if '.module.' in bn_name:
            # BN inside SeqToANNContainer: conv_bn1.module.1 → conv_bn1.module.0
            conv_name = bn_name.rsplit('.', 1)[0] + '.0'
            if conv_name in modules_dict:
                layer = modules_dict[conv_name]
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    return layer
        return None

    base = m.group(1)
    num = m.group(2)  # '' or '1', '2', etc.

    # Try: base_conv{num}, base_linear{num}
    for suffix in ['_conv', '_linear']:
        # Handle parent prefix: block.0.attn.q_bn → block.0.attn.q_linear
        cand = base + suffix + num
        if cand in modules_dict:
            layer = modules_dict[cand]
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                return layer

    return None


def recalibrate_bn(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 64,
    momentum: float = 0.1,
) -> nn.Module:
    """Recalibrate BatchNorm running statistics after weight modification.

    Sets all BN layers to train mode (to update running stats), runs
    calibration data through the model, then restores eval mode.
    This is training-free — no gradients are computed.

    Args:
        model:       Model with potentially stale BN stats.
        dataloader:  Calibration data.
        device:      Compute device.
        max_batches: Number of calibration batches.
        momentum:    BN momentum for running stats update.

    Returns:
        The model with updated BN statistics (same object).
    """
    from models.neurons import reset_net

    # Reset and enable stats update
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.reset_running_stats()
            mod.train()
            mod.momentum = momentum

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            model(batch[0].to(device))
            reset_net(model)

    model.eval()
    return model
