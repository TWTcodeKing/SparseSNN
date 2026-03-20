"""TritonSparseAccelerator — Triton GPU kernel backend for SNN sparsity.

Instruments Conv2d layers with im2col + Triton SpMM and SSA modules
with block-sparse attention, exploiting binary spike sparsity.
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from iengine.common.stats import LinearStats, AttentionStats
from .conversion import make_sparse_conv2d_forward, make_block_sparse_ssa_forward


class TritonSparseAccelerator(SparseAccelerator):
    """Triton-based sparse inference backend for SNNs.

    1. Conv2d layers: im2col + Triton SpMM skipping zero columns.
    2. SSA attention: block-sparse Q @ K^T via Triton kernel.

    Config options:
        density_threshold (float): Max density for sparse Conv2d (default 0.5).
        block_size (int):          Tile size for block-sparse attention (default 16).
        min_tensor_size (int):     Min elements to justify Triton launch (default 4096).
        min_seq_len (int):         Min sequence length N for sparse attention (default 32).
        enable_conv2d (bool):      Enable sparse Conv2d (default True).
        enable_attention (bool):   Enable block-sparse attention (default True).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._density_threshold = self.config.get('density_threshold', 0.5)
        self._block_size = self.config.get('block_size', 16)
        self._min_tensor_size = self.config.get('min_tensor_size', 4096)
        self._min_seq_len = self.config.get('min_seq_len', 32)
        self._enable_conv2d = self.config.get('enable_conv2d', True)
        self._enable_attention = self.config.get('enable_attention', True)

        self._conv_stats = LinearStats()
        self._attention_stats = {}
        self._original_forwards = {}
        self._enabled_ref = [True]

    def prepare(self, model):
        if self._enable_conv2d:
            self._patch_conv2d(model)
        if self._enable_attention:
            self._patch_ssa(model)
        self.attach_profiler(model)
        return model

    def _patch_conv2d(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                new_fwd = make_sparse_conv2d_forward(
                    module, module.forward, name, self._conv_stats,
                    self._density_threshold, self._min_tensor_size,
                    self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)

    def _patch_ssa(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SSA':
                attn_stats = AttentionStats()
                self._attention_stats[name] = attn_stats
                new_fwd = make_block_sparse_ssa_forward(
                    module, attn_stats, name,
                    self._block_size, self._min_seq_len,
                    self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)

    def cleanup(self, model):
        self.detach_profiler()
        for _, (module, original_fwd) in self._original_forwards.items():
            module.forward = original_fwd
        self._original_forwards.clear()
        return model

    def get_stats(self):
        total_ops = self._conv_stats.total_ops
        effective_ops = self._conv_stats.effective_ops
        per_layer = {}

        for name, s in self._conv_stats.per_layer.items():
            per_layer[name] = {
                'type': 'conv2d',
                'total_ops': s['total_ops'],
                'effective_ops': s['effective_ops'],
                'density': sum(s['densities']) / len(s['densities']) if s['densities'] else 1.0,
                'total_calls': s['sparse_calls'] + s['dense_calls'],
                'sparse_calls': s['sparse_calls'],
            }

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
        self._conv_stats.reset()
        for s in self._attention_stats.values():
            s.reset()

    @property
    def name(self):
        return 'TritonSparse'
