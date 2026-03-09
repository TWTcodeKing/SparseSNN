"""Sputnik sparse acceleration backend for SNN models.

Instruments a model with Sputnik SpMM hooks on nn.Linear layers and
monkey-patches SSA attention modules for sparse execution.
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from .sparse_linear import (
    _check_sputnik, SputnikLinearStats, make_sparse_linear_forward,
)
from .sparse_attention import (
    SputnikAttentionStats, make_sparse_ssa_forward,
)


class SputnikAccelerator(SparseAccelerator):
    """Sparse acceleration using Google Research's Sputnik CUDA kernels.

    Uses Sputnik SpMM for:
    1. nn.Linear layers: sparse_activation @ weight.T (activation sparsity)
    2. SSA attention: sparse_Q @ K.T (spike sparsity in queries)

    Config options:
        density_threshold (float): Max density to use sparse path. Default 0.15.
        min_elements (int): Min tensor elements for sparse path. Default 4096.
        hook_linear (bool): Whether to hook nn.Linear layers. Default True.
        hook_attention (bool): Whether to patch SSA attention. Default True.
        exclude_layers (list): Layer name patterns to skip. Default [].
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._linear_stats = SputnikLinearStats()
        self._attn_stats = SputnikAttentionStats()
        self._original_forwards = {}  # module_id -> (module, original_forward)
        self._enabled_ref = [True]  # mutable reference for hooks

        # Config
        self._density_threshold = self.config.get('density_threshold', 0.15)
        self._min_elements = self.config.get('min_elements', 4096)
        self._hook_linear = self.config.get('hook_linear', True)
        self._hook_attention = self.config.get('hook_attention', True)
        self._exclude_layers = self.config.get('exclude_layers', [])

        # Check availability
        self._available = _check_sputnik()
        if not self._available:
            print(
                "[SputnikAccelerator] WARNING: torch_sputnik not installed.\n"
                "  Sparse acceleration will be disabled.\n"
                "  Build from source: bash iengine/sputnik_sparse/build.sh\n"
                "  See iengine/sputnik_sparse/INSTALL.md for details."
            )

    @property
    def available(self) -> bool:
        """Whether the Sputnik backend is available."""
        return self._available

    def prepare(self, model: nn.Module) -> nn.Module:
        """Instrument model with Sputnik sparse hooks.

        Registers forward hooks on nn.Linear layers and monkey-patches
        SSA.forward() for sparse attention computation.

        Args:
            model: The SNN model to instrument.

        Returns:
            The same model with hooks attached.
        """
        if not self._available:
            print("[SputnikAccelerator] Skipping prepare — torch_sputnik not available.")
            return model

        self._linear_stats.reset()
        self._attn_stats.reset()
        self._enabled_ref[0] = True

        # Monkey-patch nn.Linear layers
        if self._hook_linear:
            count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if any(ex in name for ex in self._exclude_layers):
                        continue
                    new_forward = make_sparse_linear_forward(
                        module=module,
                        original_forward=module.forward,
                        layer_name=name,
                        stats=self._linear_stats,
                        density_threshold=self._density_threshold,
                        min_elements=self._min_elements,
                        enabled_ref=self._enabled_ref,
                    )
                    original = monkey_patch_forward(module, new_forward)
                    self._original_forwards[id(module)] = (module, original)
                    count += 1
            print(f"[SputnikAccelerator] Patched {count} nn.Linear layers")

        # Monkey-patch SSA attention modules
        if self._hook_attention:
            count = 0
            for name, module in model.named_modules():
                if module.__class__.__name__ == 'SSA':
                    if any(ex in name for ex in self._exclude_layers):
                        continue
                    sparse_fwd = make_sparse_ssa_forward(
                        ssa_module=module,
                        module_name=name,
                        stats=self._attn_stats,
                        density_threshold=self._density_threshold,
                        min_elements=self._min_elements,
                        enabled_ref=self._enabled_ref,
                    )
                    original = monkey_patch_forward(module, sparse_fwd)
                    self._original_forwards[id(module)] = (module, original)
                    count += 1
            print(f"[SputnikAccelerator] Patched {count} SSA attention modules")

        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        """Remove all hooks and restore original forward methods.

        Args:
            model: The instrumented model.

        Returns:
            The model with all instrumentation removed.
        """
        # Restore all original forwards (Linear + SSA)
        n_restored = 0
        for module, original_forward in self._original_forwards.values():
            module.forward = original_forward
            n_restored += 1
        self._original_forwards.clear()
        print(f"[SputnikAccelerator] Restored {n_restored} forward methods")

        return model

    def get_stats(self) -> dict:
        """Return sparse execution statistics.

        Returns:
            dict with total_ops, effective_ops, density, per_layer, and
            backend-specific counters.
        """
        linear = self._linear_stats.to_dict()
        attn = self._attn_stats.to_dict()

        total_ops = linear['total_ops'] + attn['total_ops']
        effective_ops = linear['effective_ops'] + attn['effective_ops']
        density = effective_ops / total_ops if total_ops > 0 else 1.0

        # Merge per-layer stats
        per_layer = {}
        per_layer.update(linear.get('per_layer', {}))
        per_layer.update(attn.get('per_layer', {}))

        return {
            'total_ops': total_ops,
            'effective_ops': effective_ops,
            'density': density,
            'per_layer': per_layer,
            'linear': linear,
            'attention': attn,
        }

    def enable(self):
        """Enable sparse execution."""
        super().enable()
        self._enabled_ref[0] = True

    def disable(self):
        """Disable sparse execution (fall back to dense)."""
        super().disable()
        self._enabled_ref[0] = False
