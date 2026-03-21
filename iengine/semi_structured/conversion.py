"""Convert nn.Linear and nn.Conv2d weights to SparseSemiStructuredTensor.

Requires NVIDIA GPUs with Sparse Tensor Core support (Ampere / Ada / Hopper).
Weights must be in float16 and satisfy the 2:4 sparsity pattern.  Both
dimensions of the 2D weight matrix must be multiples of 16.

Linear layers are converted directly (weight is already 2D).
Conv2d layers are converted via im2col: the forward pass unfolds the input
into a 2D matrix, then uses sparse GEMM with the unrolled weight
W_2d = W.reshape(C_out, C_in*Kh*Kw).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sparse.pruning import prune_2_4, verify_2_4


# Attribute name used to stash dense weights for restore_dense()
_DENSE_WEIGHT_ATTR = '_dense_weight_backup'
_CONVERTED_FLAG = '_semi_structured_converted'
_FP16_HOOK_ATTR = '_fp16_input_hook'


def _dims_valid_for_semi_structured(weight: torch.Tensor) -> bool:
    """Check if weight dimensions satisfy SparseSemiStructuredTensor requirements.

    CUTLASS sparse kernels require:
      - dim 0 (rows / M): multiple of 32
      - dim 1 (cols / K): multiple of 64
    For fp16 on Ampere/Ada/Hopper.
    """
    return weight.shape[0] % 32 == 0 and weight.shape[1] % 64 == 0


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


def _make_semi_structured_forward_fp16(original_forward):
    """Factory for fp16 models: flatten to 2D only, no dtype cast."""
    def _semi_structured_forward(x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        out_2d = original_forward(x_2d)
        out_shape = orig_shape[:-1] + (out_2d.shape[-1],)
        return out_2d.reshape(out_shape)
    return _semi_structured_forward


def _reshape_pre_hook(module, args):
    """Pre-hook: flatten ND input to 2D for SparseSemiStructuredTensor."""
    x = args[0]
    if x.ndim > 2:
        module._sparse_orig_shape = x.shape
        return (x.reshape(-1, x.shape[-1]),)
    module._sparse_orig_shape = None
    return args


def _reshape_post_hook(module, args, output):
    """Post-hook: restore ND shape after sparse matmul."""
    orig_shape = module._sparse_orig_shape
    if orig_shape is not None:
        out_shape = orig_shape[:-1] + (output.shape[-1],)
        return output.reshape(out_shape)
    return output


class SparseConv2d(nn.Module):
    """Conv2d replacement using im2col + 2:4 sparse GEMM.

    Stores the weight as a 2D SparseSemiStructuredTensor of shape
    (C_out, C_in*Kh*Kw) and computes convolution via:
        x_unfolded = F.unfold(x, ...)          # (B, C_in*Kh*Kw, L)
        output = W_sparse @ x_unfolded          # (B, C_out, L)
        output = F.fold(output, ...)            # (B, C_out, H_out, W_out)

    This routes the heavy GEMM through Sparse Tensor Cores while keeping
    the unfold/fold on standard dense paths.
    """

    def __init__(self, conv: nn.Conv2d, w_sparse_2d):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        # 2D sparse weight: (C_out, C_in*Kh*Kw)
        self.weight = nn.Parameter(w_sparse_2d, requires_grad=False)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        # x: (B, C_in, H, W)
        orig_dtype = x.dtype
        B = x.shape[0]
        # im2col: (B, C_in*Kh*Kw, L) where L = H_out * W_out
        x_unf = F.unfold(
            x, self.kernel_size,
            dilation=self.dilation, padding=self.padding, stride=self.stride,
        )
        L = x_unf.shape[2]

        # Sparse GEMM: (C_out, C_in*Kh*Kw) @ (C_in*Kh*Kw, B*L)
        # Cast to fp16 if needed (sparse weight is always fp16)
        x_2d = x_unf.permute(0, 2, 1).reshape(B * L, -1)  # (B*L, K)
        if x_2d.dtype != self.weight.dtype:
            x_2d = x_2d.to(self.weight.dtype)
        out_2d = F.linear(x_2d, self.weight)                # (B*L, C_out)
        output = out_2d.reshape(B, L, -1).permute(0, 2, 1)  # (B, C_out, L)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(0).unsqueeze(2)

        # Compute output spatial dims
        H_out = (x.shape[2] + 2 * self.padding[0] - self.dilation[0]
                 * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (x.shape[3] + 2 * self.padding[1] - self.dilation[1]
                 * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return output.reshape(B, self.out_channels, H_out, W_out)

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, sparse_2:4=True')


def _conv2d_weight_to_2d(weight: torch.Tensor) -> torch.Tensor:
    """Reshape Conv2d weight (C_out, C_in, Kh, Kw) → (C_out, C_in*Kh*Kw)."""
    return weight.reshape(weight.shape[0], -1).contiguous()


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


def convert_to_semi_structured(
    model: nn.Module,
    exclude_names: Optional[list] = None,
    exclude_head: bool = True,
    convert_conv2d: bool = True,
) -> dict:
    """Convert eligible nn.Linear AND nn.Conv2d layers to 2:4 sparse.

    Linear layers: weight stored as SparseSemiStructuredTensor, forward
        monkey-patched for ND→2D reshape + dtype cast.
    Conv2d layers: replaced with SparseConv2d (im2col + sparse GEMM).
        Only groups=1 Conv2d layers are eligible.

    Both dimensions of the 2D weight matrix must be multiples of 16:
      - Linear: (out_features, in_features)
      - Conv2d: (C_out, C_in * Kh * Kw)

    Args:
        model:         Model to convert (modified in-place, should be on CUDA).
        exclude_names: Module name prefixes to skip.
        exclude_head:  If True, exclude modules named 'head'.
        convert_conv2d: If True, also convert eligible Conv2d layers.

    Returns:
        Dict of per-layer conversion info.
    """
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    SparseSemiStructuredTensor._FORCE_CUTLASS = True

    if exclude_names is None:
        exclude_names = []
    if exclude_head:
        exclude_names = list(exclude_names) + ['head']

    conversion_info = {}

    # Collect Conv2d replacements (can't modify dict during iteration)
    conv_replacements = []

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_conv = isinstance(module, nn.Conv2d)

        if not is_linear and not is_conv:
            continue
        if is_conv and not convert_conv2d:
            continue

        info = {
            'type': 'Linear' if is_linear else 'Conv2d',
            'shape': tuple(module.weight.shape),
            'converted': False,
            'reason': '',
        }

        if _should_exclude(name, exclude_names):
            info['reason'] = 'excluded by name'
            conversion_info[name] = info
            continue

        # Conv2d: only groups=1 is supported
        if is_conv and module.groups != 1:
            info['reason'] = f'groups={module.groups} (only groups=1 supported)'
            conversion_info[name] = info
            continue

        # Get the 2D weight shape for dimension check
        if is_linear:
            w_2d_shape = module.weight.shape  # (out, in)
        else:
            w_2d = _conv2d_weight_to_2d(module.weight.data)
            w_2d_shape = w_2d.shape  # (C_out, C_in*Kh*Kw)

        if w_2d_shape[0] % 32 != 0 or w_2d_shape[1] % 64 != 0:
            info['reason'] = (
                f'2D shape {tuple(w_2d_shape)}: need rows%32==0, cols%64==0'
            )
            conversion_info[name] = info
            continue

        with torch.no_grad():
            # Back up original weight
            setattr(module, _DENSE_WEIGHT_ATTR, module.weight.data.clone().cpu())

            if is_linear:
                w_fp16 = module.weight.data.half()
                # Skip re-pruning if already 2:4 (e.g. from OBS)
                if not verify_2_4(w_fp16):
                    w_fp16 = prune_2_4(w_fp16)

                module.weight = nn.Parameter(
                    to_sparse_semi_structured(w_fp16),
                    requires_grad=False,
                )
                if module.bias is not None:
                    module.bias = nn.Parameter(
                        module.bias.data.half(), requires_grad=False,
                    )

                # Monkey-patch forward for reshape + dtype cast
                _orig_fwd = module.forward
                module.forward = _make_semi_structured_forward(_orig_fwd)
                setattr(module, _FP16_HOOK_ATTR, _orig_fwd)

            else:  # Conv2d
                w_2d_fp16 = _conv2d_weight_to_2d(module.weight.data).half()
                # Skip re-pruning if already 2:4
                if not verify_2_4(w_2d_fp16):
                    w_2d_fp16 = prune_2_4(w_2d_fp16)

                w_sparse = to_sparse_semi_structured(w_2d_fp16)

                # Prepare bias in fp16
                if module.bias is not None:
                    module.bias.data = module.bias.data.half()

                # Queue replacement (SparseConv2d replaces nn.Conv2d)
                conv_replacements.append((name, module, w_sparse))

        info['converted'] = True
        info['2d_shape'] = tuple(w_2d_shape)
        info['reason'] = 'success'
        conversion_info[name] = info

    # Apply Conv2d → SparseConv2d replacements
    for name, conv_module, w_sparse in conv_replacements:
        sparse_conv = SparseConv2d(conv_module, w_sparse)
        # Copy to same device
        device = next(conv_module.parameters()).device
        sparse_conv = sparse_conv.to(device)

        # Navigate to parent module and replace
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent = dict(model.named_modules())[parts[0]]
            child_name = parts[1]
        setattr(parent, child_name, sparse_conv)

    setattr(model, _CONVERTED_FLAG, True)
    return conversion_info
