"""TorchSparseAccelerator — CSR sparse backend using torch.sparse.mm.

Instruments nn.Linear layers with forward hooks and monkey-patches SSA
modules to use sparse matmul when spike activations are sufficiently sparse.
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from .sparse_linear import make_sparse_linear_forward, SparseLinearStats
from .sparse_attention import (
    make_sparse_ssa_forward, SparseAttentionStats
)


class TorchSparseAccelerator(SparseAccelerator):
    """Sparse inference backend using PyTorch CSR sparse tensors.

    Exploits spike sparsity in SNN activations by:
    1. Intercepting nn.Linear forward passes and using torch.sparse.mm
       when input density is below threshold.
    2. Monkey-patching SSA (Spiking Self-Attention) forward to use
       sparse matmul for q@k.T computation.

    Config options:
        density_threshold (float): Max density for sparse path. Default 0.15.
        enable_linear (bool): Enable sparse Linear hooks. Default True.
        enable_attention (bool): Enable sparse SSA patching. Default True.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._density_threshold = self.config.get('density_threshold', 0.15)
        self._enable_linear = self.config.get('enable_linear', True)
        self._enable_attention = self.config.get('enable_attention', True)

        self._linear_stats = {}
        self._attention_stats = {}
        self._original_forwards = {}  # {module_name: (module, original_forward)}
        self._enabled_ref = [True]  # mutable ref for runtime toggle

    def prepare(self, model):
        """Instrument model for sparse execution.

        Scans for nn.Linear modules and registers forward hooks.
        Finds SSA modules by class name and monkey-patches their forward.
        """
        if self._enable_linear:
            self._patch_linear_modules(model)

        if self._enable_attention:
            self._patch_ssa_modules(model)

        self.attach_profiler(model)
        return model

    def _patch_linear_modules(self, model):
        """Monkey-patch forward on all nn.Linear modules for sparse execution."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                new_forward = make_sparse_linear_forward(
                    module=module,
                    original_forward=module.forward,
                    layer_name=name,
                    stats_dict=self._linear_stats,
                    density_threshold=self._density_threshold,
                    enabled_ref=self._enabled_ref,
                )
                original = monkey_patch_forward(module, new_forward)
                self._original_forwards[name] = (module, original)

    def _patch_ssa_modules(self, model):
        """Find SSA modules by class name and monkey-patch forward."""
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SSA':
                new_forward = make_sparse_ssa_forward(
                    ssa_module=module,
                    stats_dict=self._attention_stats,
                    layer_name=name,
                    density_threshold=self._density_threshold,
                    enabled_ref=self._enabled_ref,
                )
                original = monkey_patch_forward(module, new_forward)
                self._original_forwards[name] = (module, original)

    def cleanup(self, model):
        """Restore all original forward methods (Linear + SSA)."""
        self.detach_profiler()
        for name, (module, original_forward) in self._original_forwards.items():
            module.forward = original_forward
        self._original_forwards.clear()

        return model

    def get_stats(self):
        """Return sparse execution statistics.

        Returns dict with:
            - total_ops: total multiply-accumulate operations
            - effective_ops: non-zero operations actually computed
            - density: effective_ops / total_ops
            - per_layer: {layer_name: {total_ops, effective_ops, density, type}}
        """
        total_ops = 0
        effective_ops = 0
        per_layer = {}

        # Linear layer stats
        for name, stats in self._linear_stats.items():
            total_ops += stats.total_ops
            effective_ops += stats.effective_ops
            per_layer[name] = {
                'type': 'linear',
                'total_ops': stats.total_ops,
                'effective_ops': stats.effective_ops,
                'density': stats.density,
                'total_calls': stats.total_calls,
                'sparse_calls': stats.sparse_calls,
            }

        # Attention stats
        for name, stats in self._attention_stats.items():
            total_ops += stats.total_ops
            effective_ops += stats.effective_ops
            per_layer[name] = {
                'type': 'attention',
                'total_ops': stats.total_ops,
                'effective_ops': stats.effective_ops,
                'density': stats.density,
                'total_calls': stats.total_calls,
                'sparse_qk_calls': stats.sparse_qk_calls,
                'sparse_av_calls': stats.sparse_av_calls,
            }

        density = effective_ops / max(total_ops, 1)

        return {
            'total_ops': total_ops,
            'effective_ops': effective_ops,
            'density': density,
            'per_layer': per_layer,
        }

    def enable(self):
        """Enable sparse execution."""
        super().enable()
        self._enabled_ref[0] = True

    def disable(self):
        """Disable sparse execution (fall back to dense)."""
        super().disable()
        self._enabled_ref[0] = False

    def reset_stats(self):
        """Clear all accumulated statistics."""
        for stats in self._linear_stats.values():
            stats.reset()
        for stats in self._attention_stats.values():
            stats.reset()

    @property
    def name(self):
        return 'TorchSparse'
