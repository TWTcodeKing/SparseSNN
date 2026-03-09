"""SemiStructuredAccelerator: SparseAccelerator backend using NVIDIA 2:4 Sparse Tensor Cores.

Applies 2:4 magnitude-based pruning to nn.Linear weights and converts them to
SparseSemiStructuredTensor for hardware-accelerated inference.
"""

import torch
import torch.nn as nn
from typing import Optional

import sys
import os

# Add project root to path so we can import iengine.common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from iengine.common.base import SparseAccelerator
from .pruning import prune_model_linear
from .conversion import (
    convert_linear_to_semi_structured,
    restore_dense,
    _CONVERTED_FLAG,
)


class SemiStructuredAccelerator(SparseAccelerator):
    """Backend using NVIDIA 2:4 structured sparsity on Linear weights.

    Config keys:
        exclude_head (bool): Skip classification head. Default True.
        exclude_names (list[str]): Additional module names to exclude.

    Usage:
        accel = SemiStructuredAccelerator({'exclude_head': True})
        model = accel.prepare(model)   # prune + convert to semi-structured
        output = model(input)          # uses Sparse Tensor Cores
        stats = accel.get_stats()
        model = accel.cleanup(model)   # restore original dense weights
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._pruning_stats = {}
        self._conversion_info = {}
        self._activation_density = {}
        self._prepared = False

    def prepare(self, model: nn.Module) -> nn.Module:
        """Convert model for semi-structured sparse inference.

        Steps:
            1. Convert model to float16 (keeping BN running stats in fp32)
            2. Apply 2:4 pruning to eligible Linear layers
            3. Convert pruned weights to SparseSemiStructuredTensor

        Args:
            model: Model on CUDA device.

        Returns:
            The same model, modified in-place.
        """
        if not self._enabled:
            return model

        exclude_names = list(self.config.get('exclude_names', []))
        exclude_head = self.config.get('exclude_head', True)

        # NOTE: We do NOT call model.half() globally — that would break
        # Conv2d and other layers that still receive float32 input.
        # Instead, convert_linear_to_semi_structured() converts only
        # eligible Linear layer weights (and biases) to float16 individually.
        # Those layers also get a forward hook to cast input to fp16.

        # Convert eligible Linear layers (pruning + semi-structured conversion)
        self._conversion_info = convert_linear_to_semi_structured(
            model,
            exclude_names=exclude_names,
            exclude_head=exclude_head,
        )

        # Collect pruning stats from the conversion results
        for lname, info in self._conversion_info.items():
            self._pruning_stats[lname] = {
                'shape': info['shape'],
                'converted': info['converted'],
                'reason': info['reason'],
                'weight_density': 0.5 if info['converted'] else 1.0,
            }

        self.attach_profiler(model)
        self._prepared = True
        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        """Restore model to dense fp32 weights.

        Args:
            model: Model previously prepared with prepare().

        Returns:
            The model with original dense weights restored.
        """
        self.detach_profiler()
        model = restore_dense(model)
        self._prepared = False
        return model

    def get_stats(self) -> dict:
        """Return sparse execution statistics.

        Returns:
            Dict with:
                - total_ops: estimated total MACs across converted layers
                - effective_ops: estimated MACs after 2:4 pruning (50% of total)
                - density: overall weight density (0.5 for fully converted)
                - per_layer: per-layer details including compound density
                - layers_converted: count of converted layers
                - layers_skipped: count of skipped layers
        """
        per_layer = {}
        total_ops = 0
        effective_ops = 0
        layers_converted = 0
        layers_skipped = 0

        for name, info in self._pruning_stats.items():
            shape = info['shape']
            layer_ops = shape[0] * shape[1]  # out_features * in_features MACs per token
            w_density = info['weight_density']

            # Compound density = activation density * weight density
            act_density = self._activation_density.get(name, {}).get('density', 1.0)
            compound = act_density * w_density

            per_layer[name] = {
                'total_ops': layer_ops,
                'effective_ops': int(layer_ops * compound),
                'density': compound,
                'weight_density': w_density,
                'activation_density': act_density,
                'converted': info['converted'],
                'shape': shape,
            }

            total_ops += layer_ops
            effective_ops += int(layer_ops * compound)

            if info['converted']:
                layers_converted += 1
            else:
                layers_skipped += 1

        return {
            'total_ops': total_ops,
            'effective_ops': effective_ops,
            'density': effective_ops / max(total_ops, 1),
            'per_layer': per_layer,
            'layers_converted': layers_converted,
            'layers_skipped': layers_skipped,
        }

    def set_activation_density(self, density_profile: dict):
        """Set per-layer activation density from profiling.

        Args:
            density_profile: Output of profile_layer_density(), mapping
                layer names to dicts with at least a 'density' key.
        """
        self._activation_density = density_profile

    @property
    def name(self) -> str:
        return 'SemiStructured2:4'
