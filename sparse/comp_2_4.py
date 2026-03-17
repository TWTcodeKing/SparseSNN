"""Dense-Sparse Weight Factorization for SNN inference.

Decomposes each Linear weight W into:
    W_24  — 2:4 structured sparse (accelerated by Sparse Tensor Cores)
    W_res — dense residual (processed via gather-accumulate on sparse activations)

such that W_24 + W_res == W_original exactly.

The 2:4 component runs through NVIDIA semi-structured sparse matmul, while the
residual leverages SNN activation sparsity: for binary spike inputs, we only need
to gather the columns of W_res corresponding to non-zero activations.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional

from sparse.pruning import prune_2_4


def factorize_weight_2_4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorize a weight matrix into 2:4 sparse + dense residual components.

    Args:
        weight: 2D tensor of shape (out_features, in_features).

    Returns:
        (W_24, W_res) where:
            W_24: 2:4 pruned weight (top-2 magnitude per group of 4)
            W_res: dense residual such that W_24 + W_res == weight exactly
    """
    W_24 = prune_2_4(weight)
    W_res = weight - W_24
    return W_24, W_res


# Attribute names for backup and state tracking
_ORIGINAL_LINEAR_ATTR = '_factorized_original_linear'
_FACTORIZED_FLAG = '_is_factorized'


class FactorizedLinear(nn.Module):
    """Linear layer factorized into 2:4 sparse + dense residual paths.

    Forward pass:
        1. Cast input to fp16
        2. main = SparseSemiStructured matmul(W_24, x)
        3. residual = gather_accumulate(W_res, x)  (exploits activation sparsity)
        4. output = main + residual + bias
        5. Cast back to original dtype

    Profiling:
        Set FactorizedLinear.PROFILE = True to collect per-path timing.
        Call FactorizedLinear.print_profile_summary() to display results.
    """

    # ── Class-level profiling state ──
    PROFILE = False
    _profile_data = {}  # {layer_id: {main_ms, residual_ms, calls, ...}}
    _profile_events = []  # deferred CUDA events: (layer_id, 'main'|'res', start, end, metadata)

    def __init__(self, original_linear: nn.Linear, name: str = ''):
        super().__init__()

        self._layer_name = name
        weight = original_linear.weight.data
        W_24, W_res = factorize_weight_2_4(weight)

        # Store residual as dense fp16 parameter (no grad for inference)
        self.W_res = nn.Parameter(W_res.half(), requires_grad=False)

        # Convert W_24 to SparseSemiStructuredTensor for hardware acceleration
        self._has_semi_structured = False
        self._semi_structured_error = None
        try:
            from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
            SparseSemiStructuredTensor._FORCE_CUTLASS = True
            # Must be on CUDA for semi-structured conversion
            w24_cuda_fp16 = W_24.half().cuda()
            self.W_24 = nn.Parameter(
                to_sparse_semi_structured(w24_cuda_fp16),
                requires_grad=False,
            )
            self._has_semi_structured = True
        except ImportError as e:
            self._semi_structured_error = f'ImportError: {e}'
            self.W_24 = nn.Parameter(W_24.half(), requires_grad=False)
        except RuntimeError as e:
            self._semi_structured_error = f'RuntimeError: {e}'
            self.W_24 = nn.Parameter(W_24.half(), requires_grad=False)
        except Exception as e:
            self._semi_structured_error = f'{type(e).__name__}: {e}'
            self.W_24 = nn.Parameter(W_24.half(), requires_grad=False)

        # Bias
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.half(), requires_grad=False
            )
        else:
            self.bias = None

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Import gather_accumulate (with fallback)
        self._gather_acc_fn = None
        try:
            from iengine.gather_accumulate import gather_accumulate
            self._gather_acc_fn = gather_accumulate
        except ImportError:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        need_cast = (orig_dtype != torch.float16)
        profiling = FactorizedLinear.PROFILE

        if profiling:
            te_start = torch.cuda.Event(enable_timing=True)
            te_end = torch.cuda.Event(enable_timing=True)
            te_start.record()

        # Flatten to 2D for matmul: (..., in_features) -> (batch, in_features)
        x_2d = x.reshape(-1, x.shape[-1])
        if need_cast:
            x_2d = x_2d.half()

        # ── Path 1: Main branch (W_24 @ x) — Sparse Tensor Core ──
        if profiling:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        main = torch.nn.functional.linear(x_2d, self.W_24)
        if profiling:
            t1.record()

        # ── Path 2: Residual branch (W_res @ x) — dense fp16 ──
        if profiling:
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t2.record()
        residual = torch.nn.functional.linear(x_2d, self.W_res)
        if profiling:
            t3.record()

        # ── Combine ──
        output = main + residual
        if self.bias is not None:
            output = output + self.bias

        # Restore shape and dtype
        out_shape = orig_shape[:-1] + (self.out_features,)
        if need_cast:
            result = output.reshape(out_shape).to(orig_dtype)
        else:
            result = output.reshape(out_shape)

        # ── Defer profiling event resolution (NO sync here!) ──
        if profiling:
            te_end.record()
            lid = self._layer_name or id(self)
            density = float((x_2d != 0).float().mean())
            FactorizedLinear._profile_events.append(
                (lid, t0, t1, t2, t3, te_start, te_end,
                 (self.out_features, self.in_features),
                 self._has_semi_structured, density)
            )

        return result

    @classmethod
    def reset_profile(cls):
        cls._profile_data = {}
        cls._profile_events = []

    @classmethod
    def resolve_events(cls):
        """Resolve all deferred CUDA events into _profile_data. Call ONCE after all forward passes."""
        if not cls._profile_events:
            return
        torch.cuda.synchronize()  # single sync to resolve all events
        for (lid, t0, t1, t2, t3, te_start, te_end, shape, has_ss, density) in cls._profile_events:
            main_ms = t0.elapsed_time(t1)
            res_ms = t2.elapsed_time(t3)
            total_layer_ms = te_start.elapsed_time(te_end)
            if lid not in cls._profile_data:
                cls._profile_data[lid] = {
                    'main_ms': 0.0, 'residual_ms': 0.0,
                    'overhead_ms': 0.0, 'total_layer_ms': 0.0,
                    'calls': 0, 'shape': shape,
                    'has_semi_structured': has_ss,
                    'density': density,
                }
            d = cls._profile_data[lid]
            d['main_ms'] += main_ms
            d['residual_ms'] += res_ms
            d['total_layer_ms'] += total_layer_ms
            d['overhead_ms'] += total_layer_ms - main_ms - res_ms
            d['calls'] += 1
            d['density'] = density
        cls._profile_events = []

    @classmethod
    def print_profile_summary(cls, total_forward_ms: float = 0.0, num_samples: int = 0):
        """Print per-layer profiling breakdown.

        Args:
            total_forward_ms: Total wall-clock time for all forward passes (from external timer).
            num_samples: Total number of samples processed.
        """
        cls.resolve_events()

        if not cls._profile_data:
            print("  No profiling data collected. Set FactorizedLinear.PROFILE = True")
            return

        # Determine number of forward passes (= calls per layer, all layers see same count)
        n_calls = max(d['calls'] for d in cls._profile_data.values())

        print(f"\n{'Layer':<45} {'Shape':<14} {'SemiStr':<8} "
              f"{'Density':<8} {'Main/call':<10} {'Res/call':<10} {'OH/call':<10} {'Total/call':<10} {'Calls':<6}")
        print('-' * 130)

        total_main = 0.0
        total_res = 0.0
        total_overhead = 0.0
        total_layer = 0.0
        for lid, d in sorted(cls._profile_data.items(), key=lambda x: str(x[0])):
            name = str(lid)[:44]
            shape_str = f"{d['shape'][0]}x{d['shape'][1]}"
            semi = 'Yes' if d['has_semi_structured'] else 'No'
            avg_main = d['main_ms'] / max(d['calls'], 1)
            avg_res = d['residual_ms'] / max(d['calls'], 1)
            avg_oh = d['overhead_ms'] / max(d['calls'], 1)
            avg_total = d['total_layer_ms'] / max(d['calls'], 1)
            total_main += d['main_ms']
            total_res += d['residual_ms']
            total_overhead += d['overhead_ms']
            total_layer += d['total_layer_ms']
            print(f"  {name:<43} {shape_str:<14} {semi:<8} "
                  f"{d['density']:<8.4f} {avg_main:<10.4f} {avg_res:<10.4f} {avg_oh:<10.4f} {avg_total:<10.4f} {d['calls']:<6}")

        divisor = max(num_samples, 1) if num_samples else max(n_calls, 1)
        unit = 'samples' if num_samples else 'calls'
        per_sample_main = total_main / divisor
        per_sample_res = total_res / divisor
        per_sample_oh = total_overhead / divisor
        per_sample_layer = total_layer / divisor

        print(f"\n  === Per-Sample Breakdown (averaged over {num_samples or n_calls} {unit}) ===")
        print(f"  Main path (Sparse TC):       {per_sample_main:.4f} ms/sample")
        print(f"  Residual path (dense fp16):  {per_sample_res:.4f} ms/sample")
        print(f"  Cast + combine overhead:     {per_sample_oh:.4f} ms/sample  (dtype cast, reshape, add, bias)")
        print(f"  FactorizedLinear total:      {per_sample_layer:.4f} ms/sample")

        if total_forward_ms > 0:
            per_sample_total = total_forward_ms / max(num_samples, 1)
            non_linear_ms = total_forward_ms - total_layer
            per_sample_nonlinear = non_linear_ms / max(num_samples, 1)
            print(f"  Non-linear (Conv/BN/LIF/..): {per_sample_nonlinear:.4f} ms/sample")
            print(f"  Total forward:               {per_sample_total:.4f} ms/sample")
            print(f"  FactorizedLinear fraction:   {total_layer / max(total_forward_ms, 1e-6) * 100:.1f}%")
            print(f"  Non-linear fraction:         {non_linear_ms / max(total_forward_ms, 1e-6) * 100:.1f}%")

    @classmethod
    def print_conversion_diagnostics(cls, model: nn.Module):
        """Print why each FactorizedLinear did or didn't get semi-structured conversion."""
        print(f"\n{'Layer':<45} {'Shape':<14} {'SemiStructured':<16} {'Error'}")
        print('-' * 100)
        for name, module in model.named_modules():
            if isinstance(module, FactorizedLinear):
                shape_str = f"{module.out_features}x{module.in_features}"
                has = 'Yes' if module._has_semi_structured else 'NO'
                err = module._semi_structured_error or ''
                print(f"  {name:<43} {shape_str:<14} {has:<16} {err}")


def convert_model_factorized(
    model: nn.Module,
    exclude_names: Optional[list] = None,
) -> dict:
    """Replace eligible Linear layers with FactorizedLinear.

    A layer is eligible if both dimensions are multiples of 16 (required by
    SparseSemiStructuredTensor) and it is not in the exclusion list.

    Args:
        model: Model to convert (modified in-place).
        exclude_names: Module name prefixes to skip. 'head' is always excluded
            (classification head typically has incompatible dimensions).

    Returns:
        Dict mapping layer names to conversion info:
            {name: {'shape': tuple, 'converted': bool, 'reason': str}}
    """
    if exclude_names is None:
        exclude_names = []
    # Always exclude head
    if 'head' not in exclude_names:
        exclude_names = list(exclude_names) + ['head']

    conversion_info = {}

    # Collect replacements first, then apply (can't modify during iteration)
    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        info = {
            'shape': tuple(module.weight.shape),
            'converted': False,
            'reason': '',
        }

        # Check exclusion
        skip = False
        for excl in exclude_names:
            if name == excl or name.startswith(excl + '.'):
                info['reason'] = f'excluded by name: {excl}'
                skip = True
                break

        if not skip:
            out_f, in_f = module.weight.shape
            if out_f % 16 != 0 or in_f % 16 != 0:
                info['reason'] = (
                    f'dimensions ({out_f}, {in_f}) not multiples of 16'
                )
                skip = True

        if not skip:
            replacements.append((name, module))
            info['converted'] = True
            info['reason'] = 'success'

        conversion_info[name] = info

    # Apply replacements
    for name, module in replacements:
        # Navigate to parent module and replace child
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent = dict(model.named_modules())[parts[0]]
            child_name = parts[1]

        factorized = FactorizedLinear(module, name=name)
        setattr(parent, child_name, factorized)

    return conversion_info


def verify_factorization(
    model_original: nn.Module,
    model_factorized: nn.Module,
    sample_input: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Verify that the factorized model produces similar outputs to the original.

    Due to fp16 precision and the two-path computation, results may differ
    slightly. We use relaxed tolerances suitable for fp16.

    Args:
        model_original: Original model (dense weights).
        model_factorized: Model with FactorizedLinear layers.
        sample_input: Input tensor for a forward pass.
        atol: Absolute tolerance for allclose check.
        rtol: Relative tolerance for allclose check.

    Returns:
        True if outputs are within tolerance.
    """
    model_original.eval()
    model_factorized.eval()

    with torch.no_grad():
        out_orig = model_original(sample_input)
        out_fact = model_factorized(sample_input)

    # Reset SNN state if applicable
    try:
        from models import reset_net
        reset_net(model_original)
        reset_net(model_factorized)
    except ImportError:
        pass

    return torch.allclose(out_orig.float(), out_fact.float(), atol=atol, rtol=rtol)
