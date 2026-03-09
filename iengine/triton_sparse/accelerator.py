"""TritonSparseAccelerator: Triton-based sparse acceleration backend for SNN models.

Instruments a model with:
1. Sparse Conv2d hooks (im2col + Triton SpMM skipping zero columns)
2. Block-sparse attention for SSA modules (Spikformer only)

Config options:
    density_threshold: float = 0.5   -- max im2col column density for sparse conv path
    block_size: int = 16             -- block tile size for block-sparse attention
    min_tensor_size: int = 4096      -- minimum M*K_nz elements to justify Triton launch
    min_seq_len: int = 32            -- minimum sequence length N for block-sparse attention
"""

import torch.nn as nn

from iengine.common.base import SparseAccelerator
from iengine.common.hooks import monkey_patch_forward
from .sparse_conv import make_sparse_conv2d_forward
from .block_sparse_attn import apply_block_sparse_attention


class TritonSparseAccelerator(SparseAccelerator):
    """Triton GPU kernel backend for sparse SNN inference.

    Exploits spike sparsity in two ways:
    - Conv2d layers: im2col + SpMM that skips zero columns entirely
    - SSA attention: block-sparse matmul that skips zero block tiles

    Performance note: For small CIFAR tensors (32x32), the Triton kernel launch
    overhead may negate sparsity savings. This backend is most beneficial for
    larger inputs (ImageNet 224x224) or deeper models with many Conv2d layers.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._patched_modules = []  # list of (module, original_forward) for cleanup
        self._stats = {}

        # Config with defaults
        self._density_threshold = self.config.get('density_threshold', 0.5)
        self._block_size = self.config.get('block_size', 16)
        self._min_tensor_size = self.config.get('min_tensor_size', 4096)
        self._min_seq_len = self.config.get('min_seq_len', 32)

    def prepare(self, model: nn.Module) -> nn.Module:
        """Instrument model for Triton sparse execution.

        1. Scan for nn.Conv2d modules, register sparse forward hooks.
        2. Scan for SSA modules, apply block-sparse attention monkey-patch.

        Args:
            model: The SNN model (Spikformer, SEW-ResNet, MS-ResNet).

        Returns:
            The same model, now instrumented with sparse hooks/patches.
        """
        self._stats = {
            'total_ops': 0,
            'effective_ops': 0,
            'sparse_launches': 0,
            'dense_fallback': 0,
            'skipped_zero': 0,
            'attn_total_blocks': 0,
            'attn_computed_blocks': 0,
            'attn_sparse_calls': 0,
            'attn_dense_fallback': 0,
        }

        # Monkey-patch Conv2d forwards
        num_conv = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                new_forward = make_sparse_conv2d_forward(
                    module=module,
                    original_forward=module.forward,
                    stats=self._stats,
                    density_threshold=self._density_threshold,
                    min_tensor_size=self._min_tensor_size,
                )
                original = monkey_patch_forward(module, new_forward)
                self._patched_modules.append((module, original))
                num_conv += 1

        # Apply block-sparse attention to SSA modules
        ssa_patches = apply_block_sparse_attention(
            model,
            block_size=self._block_size,
            min_seq_len=self._min_seq_len,
            stats=self._stats,
        )
        self._patched_modules.extend(ssa_patches)

        num_ssa = len(ssa_patches)
        print(f"[TritonSparseAccelerator] Prepared: "
              f"{num_conv} Conv2d hooks, {num_ssa} SSA patches")
        print(f"  Config: density_threshold={self._density_threshold}, "
              f"block_size={self._block_size}, "
              f"min_tensor_size={self._min_tensor_size}")

        self.attach_profiler(model)
        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        """Remove all hooks and restore original forward methods.

        Args:
            model: The instrumented model.

        Returns:
            The model with all instrumentation removed.
        """
        self.detach_profiler()
        # Restore all original forwards (Conv2d + SSA)
        num_restored = len(self._patched_modules)
        for module, original_forward in self._patched_modules:
            module.forward = original_forward
        self._patched_modules.clear()

        print(f"[TritonSparseAccelerator] Cleanup: {num_restored} forwards restored")

        return model

    def get_stats(self) -> dict:
        """Return sparse execution statistics.

        Returns dict with:
            - total_ops: Total MACs across all Conv2d layers
            - effective_ops: Non-zero MACs actually computed
            - density: effective_ops / total_ops
            - sparse_launches: Number of Triton kernel launches for Conv2d
            - dense_fallback: Times Conv2d fell back to dense
            - skipped_zero: Conv2d calls with entirely zero input
            - attn_total_blocks: Total attention blocks across all SSA calls
            - attn_computed_blocks: Non-zero attention blocks computed
            - attn_block_density: Fraction of attention blocks that were non-zero
            - per_layer: {} (per-layer stats not tracked in this backend)
        """
        total = self._stats.get('total_ops', 0)
        effective = self._stats.get('effective_ops', 0)
        density = effective / max(total, 1)

        attn_total = self._stats.get('attn_total_blocks', 0)
        attn_computed = self._stats.get('attn_computed_blocks', 0)
        attn_density = attn_computed / max(attn_total, 1)

        return {
            'total_ops': total,
            'effective_ops': effective,
            'density': density,
            'sparse_launches': self._stats.get('sparse_launches', 0),
            'dense_fallback': self._stats.get('dense_fallback', 0),
            'skipped_zero': self._stats.get('skipped_zero', 0),
            'attn_total_blocks': attn_total,
            'attn_computed_blocks': attn_computed,
            'attn_block_density': attn_density,
            'attn_sparse_calls': self._stats.get('attn_sparse_calls', 0),
            'attn_dense_fallback': self._stats.get('attn_dense_fallback', 0),
            'per_layer': {},
        }

    @property
    def name(self) -> str:
        return 'TritonSparse'

    def enable(self):
        """Enable sparse execution."""
        super().enable()
        # Hooks check self._enabled via the stats dict
        self._stats['_enabled'] = True

    def disable(self):
        """Disable sparse execution (hooks become no-ops)."""
        super().disable()
        self._stats['_enabled'] = False
