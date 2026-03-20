"""TorchSparseAccelerator — CSR sparse backend using torch.sparse.mm.

Instruments nn.Linear and nn.Conv2d layers with forward hooks, and
monkey-patches SSA modules to use sparse matmul when spike activations
are sufficiently sparse.
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from iengine.common.stats import LinearStats, AttentionStats
from .conversion import (
    make_sparse_linear_forward,
    make_sparse_conv2d_forward,
    make_sparse_ssa_forward,
)


class TorchSparseAccelerator(SparseAccelerator):
    """Sparse inference backend using PyTorch CSR sparse tensors.

    Exploits spike sparsity in SNN activations by:
    1. Intercepting nn.Linear forward passes and using torch.sparse.mm
       when input density is below threshold.
    2. Intercepting nn.Conv2d forward passes via im2col + CSR sparse GEMM.
    3. Monkey-patching SSA (Spiking Self-Attention) forward to use
       sparse matmul for q@k.T computation.

    Config options:
        density_threshold (float): Max density for sparse path. Default 0.15.
        enable_linear (bool):    Enable sparse Linear hooks. Default True.
        enable_conv2d (bool):    Enable sparse Conv2d hooks. Default True.
        enable_attention (bool): Enable sparse SSA patching. Default True.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._density_threshold = self.config.get('density_threshold', 0.15)
        self._enable_linear = self.config.get('enable_linear', True)
        self._enable_conv2d = self.config.get('enable_conv2d', True)
        self._enable_attention = self.config.get('enable_attention', True)

        self._linear_stats = LinearStats()
        self._conv_stats = LinearStats()
        self._attention_stats = {}  # {name: AttentionStats}
        self._original_forwards = {}
        self._enabled_ref = [True]

    def prepare(self, model):
        if self._enable_linear:
            self._patch_linear(model)
        if self._enable_conv2d:
            self._patch_conv2d(model)
        if self._enable_attention:
            self._patch_ssa(model)
        self.attach_profiler(model)
        return model

    def _patch_linear(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                new_fwd = make_sparse_linear_forward(
                    module, module.forward, name, self._linear_stats,
                    self._density_threshold, self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)

    def _patch_conv2d(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                new_fwd = make_sparse_conv2d_forward(
                    module, module.forward, name, self._conv_stats,
                    self._density_threshold, self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)

    def _patch_ssa(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SSA':
                attn_stats = AttentionStats()
                self._attention_stats[name] = attn_stats
                new_fwd = make_sparse_ssa_forward(
                    module, attn_stats, name,
                    self._density_threshold, self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)

    def cleanup(self, model):
        self.detach_profiler()
        for name, (module, original_fwd) in self._original_forwards.items():
            module.forward = original_fwd
        self._original_forwards.clear()
        return model

    def get_stats(self):
        total_ops = self._linear_stats.total_ops + self._conv_stats.total_ops
        effective_ops = self._linear_stats.effective_ops + self._conv_stats.effective_ops
        per_layer = {}

        # Linear stats
        for name, s in self._linear_stats.per_layer.items():
            per_layer[name] = {
                'type': 'linear',
                'total_ops': s['total_ops'],
                'effective_ops': s['effective_ops'],
                'density': sum(s['densities']) / len(s['densities']) if s['densities'] else 1.0,
                'total_calls': s['sparse_calls'] + s['dense_calls'],
                'sparse_calls': s['sparse_calls'],
            }

        # Conv2d stats
        for name, s in self._conv_stats.per_layer.items():
            per_layer[name] = {
                'type': 'conv2d',
                'total_ops': s['total_ops'],
                'effective_ops': s['effective_ops'],
                'density': sum(s['densities']) / len(s['densities']) if s['densities'] else 1.0,
                'total_calls': s['sparse_calls'] + s['dense_calls'],
                'sparse_calls': s['sparse_calls'],
            }

        # Attention stats
        for name, astats in self._attention_stats.items():
            total_ops += astats.total_ops
            effective_ops += astats.effective_ops
            per_layer[name] = {
                'type': 'attention',
                'total_ops': astats.total_ops,
                'effective_ops': astats.effective_ops,
                'density': astats.density,
                'total_calls': astats.total_calls,
                'sparse_qk_calls': astats.sparse_qk_calls,
            }

        return {
            'total_ops': total_ops,
            'effective_ops': effective_ops,
            'density': effective_ops / max(total_ops, 1),
            'per_layer': per_layer,
        }

    def enable(self):
        super().enable()
        self._enabled_ref[0] = True

    def disable(self):
        super().disable()
        self._enabled_ref[0] = False

    def reset_stats(self):
        self._linear_stats.reset()
        self._conv_stats.reset()
        for s in self._attention_stats.values():
            s.reset()

    @property
    def name(self):
        return 'TorchSparse'
