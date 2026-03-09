"""Base class for all sparse acceleration backends."""

from abc import ABC, abstractmethod
import torch.nn as nn

from .profiler import SparsityProfiler


class SparseAccelerator(ABC):
    """Abstract base for sparse inference backends.

    Each backend instruments a model (via hooks, wrapping, or weight replacement)
    to exploit spike sparsity without modifying model source files.

    Usage:
        accel = MyBackend(config)
        model = accel.prepare(model)      # instrument for sparse execution
        output = model(input)             # forward pass uses sparse ops
        stats = accel.get_stats()         # check ops saved
        report = accel.get_density_report()  # per-layer activation density
        model = accel.cleanup(model)      # restore original behavior
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._enabled = True
        self._profiler = SparsityProfiler()

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

    def attach_profiler(self, model: nn.Module) -> int:
        """Attach the sparsity profiler to the model.

        Called automatically by prepare() in subclasses. Can also be called
        manually for standalone profiling.

        Returns the number of hooks registered.
        """
        self._profiler.reset()
        return self._profiler.attach(model)

    def detach_profiler(self):
        """Remove profiler hooks from the model."""
        self._profiler.detach()

    def get_density_report(self) -> dict:
        """Return per-layer activation density report from the profiler."""
        return self._profiler.get_report()

    def export_density_report(self, filepath: str, model_name: str = '',
                              extra_info: str = ''):
        """Export the density report to a markdown file."""
        self._profiler.export_report(filepath, model_name=model_name,
                                     extra_info=extra_info)
