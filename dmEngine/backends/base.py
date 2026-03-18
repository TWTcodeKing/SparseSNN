from abc import ABC, abstractmethod
import torch.nn as nn


class WeightSparseBackend(ABC):
    """Abstract base class for weight-sparse backends.

    Unlike iengine's SparseAccelerator, weight sparsity here is STATIC:
    weights are sparsified once at prepare() time and the sparse format is
    reused every forward call.  There is no runtime density gating or profiler.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def prepare(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Sparsify weights and convert to backend-specific sparse format."""

    @abstractmethod
    def cleanup(self, model: nn.Module) -> nn.Module:
        """Restore original dense weights."""

    @abstractmethod
    def supported_sparsities(self) -> list:
        """Sparsity levels this backend supports. Empty = any."""

    @property
    @abstractmethod
    def name(self) -> str: ...
