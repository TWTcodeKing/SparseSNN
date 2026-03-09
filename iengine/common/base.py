"""Base class for all sparse acceleration backends."""

from abc import ABC, abstractmethod
import torch.nn as nn


class SparseAccelerator(ABC):
    """Abstract base for sparse inference backends.

    Each backend instruments a model (via hooks, wrapping, or weight replacement)
    to exploit spike sparsity without modifying model source files.

    Usage:
        accel = MyBackend(config)
        model = accel.prepare(model)      # instrument for sparse execution
        output = model(input)             # forward pass uses sparse ops
        stats = accel.get_stats()         # check ops saved
        model = accel.cleanup(model)      # restore original behavior
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._enabled = True

    @abstractmethod
    def prepare(self, model: nn.Module) -> nn.Module:
        """Instrument a model for sparse acceleration.

        Must NOT modify model source files. Uses hooks, module wrapping,
        or weight replacement only. Returns the instrumented model.
        """

    @abstractmethod
    def cleanup(self, model: nn.Module) -> nn.Module:
        """Remove all instrumentation and restore original behavior."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Return sparse execution statistics.

        Returns dict with at minimum:
            - 'total_ops': total multiply-accumulate operations
            - 'effective_ops': non-zero operations actually computed
            - 'density': effective_ops / total_ops
            - 'per_layer': {layer_name: {total_ops, effective_ops, density}}
        """

    @property
    def name(self) -> str:
        """Backend name for display/logging."""
        return self.__class__.__name__

    def enable(self):
        """Enable sparse execution."""
        self._enabled = True

    def disable(self):
        """Disable sparse execution (fall back to dense)."""
        self._enabled = False
