"""Monkey-patched nn.Linear forward that exploits spike sparsity via CSR sparse matmul.

When the input activation density is below a threshold, the input is converted
to CSR sparse format and torch.sparse.mm is used instead of dense matmul.
Otherwise falls back to the original dense forward — no double computation.
"""

import torch
import torch.nn as nn

from iengine.common.density import measure_density, should_use_sparse, to_sparse_csr_2d


class SparseLinearStats:
    """Tracks per-layer statistics for sparse linear acceleration."""

    def __init__(self):
        self.total_calls = 0
        self.sparse_calls = 0
        self.total_ops = 0
        self.effective_ops = 0

    def record(self, is_sparse, input_elements, output_features, density):
        self.total_calls += 1
        ops = input_elements * output_features
        self.total_ops += ops
        if is_sparse:
            self.sparse_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.effective_ops += ops

    def reset(self):
        self.total_calls = 0
        self.sparse_calls = 0
        self.total_ops = 0
        self.effective_ops = 0

    @property
    def density(self):
        return self.effective_ops / max(self.total_ops, 1)


def make_sparse_linear_forward(module, original_forward, layer_name, stats_dict,
                                density_threshold=0.15, enabled_ref=None):
    """Create a replacement forward for nn.Linear that uses sparse matmul.

    When density is below threshold: CSR sparse matmul (skips dense entirely).
    When density is above threshold: calls original_forward (dense, no redundancy).

    Args:
        module: The nn.Linear module.
        original_forward: The original forward method to fall back to.
        layer_name: Name for stats tracking.
        stats_dict: Dict mapping layer names to SparseLinearStats.
        density_threshold: Max density for sparse path.
        enabled_ref: List with single bool element [True/False] for runtime toggle.

    Returns:
        New forward function that replaces module.forward.
    """
    if layer_name not in stats_dict:
        stats_dict[layer_name] = SparseLinearStats()

    def sparse_forward(x):
        # Disabled → use original dense forward directly
        if enabled_ref is not None and not enabled_ref[0]:
            return original_forward(x)

        if not should_use_sparse(x, threshold=density_threshold):
            stats_dict[layer_name].record(
                is_sparse=False,
                input_elements=x.shape[-1],
                output_features=module.weight.shape[0],
                density=1.0,
            )
            return original_forward(x)

        density = measure_density(x)

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        try:
            x_csr = to_sparse_csr_2d(x_2d)
            weight_t = module.weight.t()
            result = torch.sparse.mm(x_csr, weight_t)

            if module.bias is not None:
                result = result + module.bias

            out_shape = original_shape[:-1] + (module.weight.shape[0],)
            result = result.reshape(out_shape)

            stats_dict[layer_name].record(
                is_sparse=True,
                input_elements=x.shape[-1],
                output_features=module.weight.shape[0],
                density=density,
            )
            return result

        except Exception:
            # Fallback to original dense forward (not re-computing)
            stats_dict[layer_name].record(
                is_sparse=False,
                input_elements=x.shape[-1],
                output_features=module.weight.shape[0],
                density=1.0,
            )
            return original_forward(x)

    return sparse_forward
