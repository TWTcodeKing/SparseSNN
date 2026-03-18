"""Random sparse weight generation utilities for benchmarking."""

import torch
import torch.nn as nn


def sparsify_weight(weight, sparsity):
    """Apply random unstructured sparsity mask to a weight tensor.

    Args:
        weight: Tensor to sparsify.
        sparsity: Fraction of elements to zero out (0.0 = dense, 1.0 = all zeros).

    Returns:
        Sparsified weight tensor (same shape).
    """
    mask = torch.rand_like(weight) >= sparsity
    return weight * mask


def sparsify_model(model, sparsity, layer_types=(nn.Linear, nn.Conv2d)):
    """Apply random unstructured sparsity to all matching layers in-place.

    Args:
        model: nn.Module to sparsify.
        sparsity: Fraction of weights to zero out per layer.
        layer_types: Tuple of layer types to sparsify.

    Returns:
        Dict of per-layer stats: {name: {'shape', 'total_params', 'actual_sparsity'}}.
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, layer_types) and hasattr(module, 'weight'):
            module.weight.data = sparsify_weight(module.weight.data, sparsity)
            total = module.weight.numel()
            zeros = (module.weight.data == 0).sum().item()
            stats[name] = {
                'shape': tuple(module.weight.shape),
                'total_params': total,
                'actual_sparsity': zeros / total,
            }
    return stats


def measure_weight_sparsity(model):
    """Measure actual weight sparsity of all parameters.

    Args:
        model: nn.Module to measure.

    Returns:
        Dict with 'overall_sparsity' and 'per_layer' stats.
    """
    per_layer = {}
    total_params = 0
    total_zeros = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        zeros = (param.data == 0).sum().item()
        total_params += numel
        total_zeros += zeros
        per_layer[name] = {
            'sparsity': zeros / numel if numel > 0 else 0.0,
            'shape': tuple(param.shape),
        }
    return {
        'overall_sparsity': total_zeros / total_params if total_params > 0 else 0.0,
        'per_layer': per_layer,
    }
