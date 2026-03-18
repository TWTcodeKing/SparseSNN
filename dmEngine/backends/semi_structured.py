"""NVIDIA 2:4 structured sparsity backend for weight-sparse inference.

Uses ``torch.sparse.SparseSemiStructuredTensor`` with NVIDIA Sparse Tensor
Cores.  Applies 2:4 structured pruning to nn.Linear weights — every group of
4 consecutive elements keeps exactly 2 non-zero values, giving a fixed 50%
sparsity ratio enforced by hardware.

Only nn.Linear layers are converted; Conv2d weight dimensions rarely satisfy
the 2:4 alignment requirements cleanly.  Weights are cast to fp16 for Sparse
Tensor Core support, with input/output cast hooks when the model runs in fp32.

Requires PyTorch >= 2.1.
"""

import torch
import torch.nn as nn

from dmEngine.backends.base import WeightSparseBackend
from sparse.pruning import prune_2_4

try:
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    _HAS_SEMI_STRUCTURED = True
except ImportError:
    _HAS_SEMI_STRUCTURED = False


class SemiStructuredBackend(WeightSparseBackend):
    """2:4 structured-sparsity backend using NVIDIA Sparse Tensor Cores.

    Configuration keys (passed via *config* dict):
        exclude_names : list[str]
            Module name prefixes to skip (e.g. ``['head']``).
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._original_state = {}  # {name: (module, orig_forward, orig_weight, orig_bias)}
        self._exclude_names = config.get('exclude_names', []) if config else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module, sparsity: float = 0.5) -> nn.Module:
        """Apply 2:4 pruning and convert Linear weights to semi-structured.

        *sparsity* is ignored — always 50 % due to the 2:4 hardware
        constraint.
        """
        if not _HAS_SEMI_STRUCTURED:
            raise RuntimeError(
                "torch.sparse.SparseSemiStructuredTensor is not available. "
                "Upgrade to PyTorch >= 2.1 to use the SemiStructured backend."
            )

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(name == e or name.startswith(e + '.') for e in self._exclude_names):
                continue

            out_f, in_f = module.weight.shape
            if out_f % 16 != 0 or in_f % 16 != 0:
                continue  # both dims must be multiples of 16

            self._prepare_linear(name, module)

        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        """Restore original dense fp32 weights and forward methods."""
        for _name, (module, orig_forward, orig_weight, orig_bias) in self._original_state.items():
            module.forward = orig_forward
            module.weight = nn.Parameter(orig_weight)
            if orig_bias is not None:
                module.bias = nn.Parameter(orig_bias)
            # No extra attributes to clean up — weight was replaced in-place
        self._original_state.clear()
        return model

    def supported_sparsities(self) -> list:
        return [0.5]  # only 2:4 = 50 %

    @property
    def name(self) -> str:
        return 'SemiStructured'

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_linear(self, name: str, module: nn.Linear):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        orig_bias = module.bias.data.clone() if module.bias is not None else None
        self._original_state[name] = (module, orig_forward, orig_weight, orig_bias)

        # Enable CUTLASS fast path
        SparseSemiStructuredTensor._FORCE_CUTLASS = True

        with torch.no_grad():
            # Apply 2:4 magnitude pruning, convert to fp16, then to semi-structured
            w_fp16 = module.weight.data.half()
            w_pruned = prune_2_4(w_fp16)

            module.weight = nn.Parameter(
                to_sparse_semi_structured(w_pruned),
                requires_grad=False,
            )

            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.half(),
                    requires_grad=False,
                )

        # Monkey-patch forward: fp32→fp16 input, flatten to 2D, call original
        # (which now uses the sparse weight internally), then fp16→fp32 output
        _original_forward = module.forward

        def _forward_linear(x, *, _fwd=_original_forward):
            input_dtype = x.dtype
            leading = x.shape[:-1]
            x_2d = x.reshape(-1, x.shape[-1]).half()
            out_2d = _fwd(x_2d)
            out = out_2d.reshape(*leading, -1)
            if input_dtype != torch.float16:
                out = out.to(input_dtype)
            return out

        module.forward = _forward_linear
