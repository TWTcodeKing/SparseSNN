"""SputnikAccelerator — Google Sputnik CUDA SpMM backend for SNNs.

Instruments nn.Linear layers and SSA attention modules with Sputnik
sparse kernels.  Gracefully degrades if torch_sputnik is not installed.
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from iengine.common.stats import LinearStats, AttentionStats
from .kernels import check_sputnik
from .conversion import make_sparse_linear_forward, make_sparse_ssa_forward


class SputnikAccelerator(SparseAccelerator):
    """Sparse acceleration using Google Research's Sputnik CUDA kernels.

    1. nn.Linear: Sputnik SpMM for sparse activations.
    2. SSA attention: Sputnik SpMM for sparse Q @ K^T.

    Config options:
        density_threshold (float): Max density for sparse path (default 0.15).
        min_elements (int):        Min tensor elements for sparse (default 4096).
        hook_linear (bool):        Hook nn.Linear layers (default True).
        hook_attention (bool):     Hook SSA modules (default True).
        exclude_layers (list):     Layer name patterns to skip.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._density_threshold = self.config.get('density_threshold', 0.15)
        self._min_elements = self.config.get('min_elements', 4096)
        self._hook_linear = self.config.get('hook_linear', True)
        self._hook_attention = self.config.get('hook_attention', True)
        self._exclude_layers = self.config.get('exclude_layers', [])

        self._linear_stats = LinearStats()
        self._attention_stats = {}  # {name: AttentionStats}
        self._original_forwards = {}
        self._enabled_ref = [True]

        self._available = check_sputnik()
        if not self._available:
            print("[SputnikAccelerator] WARNING: torch_sputnik not installed.\n"
                  "  Build: bash iengine/sputnik_sparse/build.sh")

    @property
    def available(self):
        return self._available

    def prepare(self, model):
        if not self._available:
            print("[SputnikAccelerator] Skipping — torch_sputnik not available.")
            return model

        if self._hook_linear:
            self._patch_linear(model)
        if self._hook_attention:
            self._patch_ssa(model)
        self.attach_profiler(model)
        return model

    def _should_skip(self, name):
        return any(ex in name for ex in self._exclude_layers)

    def _patch_linear(self, model):
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not self._should_skip(name):
                new_fwd = make_sparse_linear_forward(
                    module, module.forward, name, self._linear_stats,
                    self._density_threshold, self._min_elements,
                    self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)
                count += 1
        print(f"[SputnikAccelerator] Patched {count} Linear layers")

    def _patch_ssa(self, model):
        count = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'SSA' and not self._should_skip(name):
                attn_stats = AttentionStats()
                self._attention_stats[name] = attn_stats
                new_fwd = make_sparse_ssa_forward(
                    module, attn_stats, name,
                    self._density_threshold, self._min_elements,
                    self._enabled_ref)
                original = monkey_patch_forward(module, new_fwd)
                self._original_forwards[name] = (module, original)
                count += 1
        print(f"[SputnikAccelerator] Patched {count} SSA modules")

    def cleanup(self, model):
        self.detach_profiler()
        for _, (module, original_fwd) in self._original_forwards.items():
            module.forward = original_fwd
        self._original_forwards.clear()
        return model

    def get_stats(self):
        total_ops = self._linear_stats.total_ops
        effective_ops = self._linear_stats.effective_ops
        per_layer = {}

        for name, s in self._linear_stats.per_layer.items():
            per_layer[name] = {
                'type': 'linear',
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
        self._linear_stats.reset()
        for s in self._attention_stats.values():
            s.reset()

    @property
    def name(self):
        return 'Sputnik'
