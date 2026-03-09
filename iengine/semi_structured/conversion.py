"""Convert nn.Linear weights to SparseSemiStructuredTensor for hardware acceleration.

Requires NVIDIA GPUs with Sparse Tensor Core support (Ampere / Ada Lovelace / Hopper).
Weights must be in float16 and satisfy the 2:4 sparsity pattern. Both dimensions
of Linear weight matrices must be multiples of 16.
"""

import torch
import torch.nn as nn
from typing import Optional

from .pruning import prune_2_4, verify_2_4


# Attribute name used to stash dense weights for restore_dense()
_DENSE_WEIGHT_ATTR = '_dense_weight_backup'
_CONVERTED_FLAG = '_semi_structured_converted'
_FP16_HOOK_ATTR = '_fp16_input_hook'


def _dims_valid_for_semi_structured(weight: torch.Tensor) -> bool:
    """Check if both weight dimensions are multiples of 16."""
    return weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0


def _should_exclude(name: str, exclude_names: Optional[list]) -> bool:
    """Check if a module name matches any exclusion pattern."""
    if not exclude_names:
        return False
    for excl in exclude_names:
        if name == excl or name.startswith(excl + '.'):
            return True
    return False


def _make_semi_structured_forward(original_forward):
    """Factory to create a forward that flattens to 2D, casts fp16, and restores."""
    def _semi_structured_forward(x):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1]).half()
        out_2d = original_forward(x_2d)
        out_shape = orig_shape[:-1] + (out_2d.shape[-1],)
        return out_2d.reshape(out_shape).to(orig_dtype)
    return _semi_structured_forward


def convert_linear_to_semi_structured(
    model: nn.Module,
    exclude_names: Optional[list] = None,
    exclude_head: bool = True,
) -> dict:
    """Convert eligible nn.Linear layers to use SparseSemiStructuredTensor weights.

    Steps for each eligible layer:
        1. Back up the original dense fp32 weight
        2. Convert weight to float16
        3. Apply 2:4 pruning
        4. Replace weight with SparseSemiStructuredTensor.from_dense()

    Args:
        model: Model to convert (modified in-place). Should already be on CUDA.
        exclude_names: Module name prefixes to skip.
        exclude_head: If True, automatically exclude modules named 'head'
            (the classification head typically has output dim not divisible by 16).

    Returns:
        Dict of converted layer info:
            {name: {'shape': tuple, 'dtype_before': str, 'converted': bool, 'reason': str}}
    """
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

    # Enable fast path
    SparseSemiStructuredTensor._FORCE_CUTLASS = True

    if exclude_names is None:
        exclude_names = []
    if exclude_head:
        exclude_names = list(exclude_names) + ['head']

    conversion_info = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        info = {
            'shape': tuple(module.weight.shape),
            'dtype_before': str(module.weight.dtype),
            'converted': False,
            'reason': '',
        }

        # Check exclusion
        if _should_exclude(name, exclude_names):
            info['reason'] = 'excluded by name'
            conversion_info[name] = info
            continue

        # Check dimension requirements
        if not _dims_valid_for_semi_structured(module.weight):
            info['reason'] = (
                f'dimensions {tuple(module.weight.shape)} not multiples of 16'
            )
            conversion_info[name] = info
            continue

        with torch.no_grad():
            # Back up original weight (in original dtype)
            setattr(module, _DENSE_WEIGHT_ATTR, module.weight.data.clone().cpu())

            # Convert to float16
            w_fp16 = module.weight.data.half()

            # Apply 2:4 pruning
            w_pruned = prune_2_4(w_fp16)

            # Convert to SparseSemiStructuredTensor
            module.weight = nn.Parameter(
                to_sparse_semi_structured(w_pruned),
                requires_grad=False,
            )

            # Also convert bias to fp16 if present
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.half(),
                    requires_grad=False,
                )

        # Monkey-patch forward to handle:
        # 1. fp32→fp16 input cast (weights are fp16, rest of model is fp32)
        # 2. Flatten ND input to 2D (SparseSemiStructuredTensor only supports 2D)
        # 3. fp16→fp32 output cast
        _original_forward = module.forward
        module.forward = _make_semi_structured_forward(_original_forward)
        setattr(module, _FP16_HOOK_ATTR, _original_forward)

        info['converted'] = True
        info['reason'] = 'success'
        conversion_info[name] = info

    # Mark model as converted
    setattr(model, _CONVERTED_FLAG, True)

    return conversion_info


def restore_dense(model: nn.Module) -> nn.Module:
    """Restore all converted Linear layers to their original dense fp32 weights.

    Args:
        model: Model previously converted with convert_linear_to_semi_structured.

    Returns:
        The model with dense weights restored (same object, modified in-place).
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if hasattr(module, _DENSE_WEIGHT_ATTR):
            dense_w = getattr(module, _DENSE_WEIGHT_ATTR)
            device = next(module.parameters()).device if list(module.parameters()) else 'cpu'
            module.weight = nn.Parameter(dense_w.to(device), requires_grad=True)
            delattr(module, _DENSE_WEIGHT_ATTR)

            # Restore bias dtype too
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.float(),
                    requires_grad=True,
                )

            # Restore original forward method
            if hasattr(module, _FP16_HOOK_ATTR):
                module.forward = getattr(module, _FP16_HOOK_ATTR)
                delattr(module, _FP16_HOOK_ATTR)

    if hasattr(model, _CONVERTED_FLAG):
        delattr(model, _CONVERTED_FLAG)

    return model
