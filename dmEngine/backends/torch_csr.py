import torch
import torch.nn as nn
import torch.nn.functional as F

from dmEngine.backends.base import WeightSparseBackend
from dmModels.sparse_weights import sparsify_weight


class TorchCSRBackend(WeightSparseBackend):
    """PyTorch CSR weight-sparse backend.

    Core idea: y = x @ W.T where W is sparse.
    Computed as y = torch.sparse.mm(W_csr, x_2d.T).T so the sparse tensor is
    always the left operand.

    The CSR weight is precomputed ONCE in prepare() and stored on the module.
    This is the key difference from iengine where CSR conversion happens per
    forward call.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._original_state = {}  # {name: (module, original_forward, original_weight)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module, sparsity: float) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._prepare_linear(name, module, sparsity)
            elif isinstance(module, nn.Conv2d):
                if module.groups != 1 or module.dilation != (1, 1):
                    continue  # unsupported configuration
                self._prepare_conv2d(name, module, sparsity)
        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        for _name, (module, orig_forward, orig_weight) in self._original_state.items():
            module.forward = orig_forward
            module.weight = nn.Parameter(orig_weight)
            if hasattr(module, '_wsparse_csr'):
                del module._wsparse_csr
        self._original_state.clear()
        return model

    def supported_sparsities(self) -> list:
        return []  # any sparsity

    @property
    def name(self) -> str:
        return 'TorchCSR'

    # ------------------------------------------------------------------
    # Linear
    # ------------------------------------------------------------------

    def _prepare_linear(self, name: str, module: nn.Linear, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        # Sparsify and convert to CSR — shape (out_features, in_features)
        sparse_w = sparsify_weight(module.weight.data, sparsity)
        w_csr = sparse_w.to_sparse_csr()
        module._wsparse_csr = w_csr
        module.weight = nn.Parameter(sparse_w)

        def _forward_linear(x, *, _mod=module):
            w_csr = _mod._wsparse_csr
            leading = x.shape[:-1]
            in_f = x.shape[-1]
            x_2d = x.reshape(-1, in_f)  # (M, in_f)
            out = torch.sparse.mm(w_csr, x_2d.t()).t()  # (M, out_f)
            if _mod.bias is not None:
                out = out + _mod.bias
            return out.reshape(*leading, -1)

        module.forward = _forward_linear

    # ------------------------------------------------------------------
    # Conv2d  (im2col + sparse matmul)
    # ------------------------------------------------------------------

    def _prepare_conv2d(self, name: str, module: nn.Conv2d, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        # Reshape weight: (C_out, C_in, kH, kW) -> (C_out, C_in*kH*kW)
        C_out = module.weight.shape[0]
        w_2d = module.weight.data.reshape(C_out, -1)
        sparse_w_2d = sparsify_weight(w_2d, sparsity)
        w_csr = sparse_w_2d.to_sparse_csr()
        module._wsparse_csr = w_csr
        # Update the dense weight too (for consistency)
        module.weight = nn.Parameter(sparse_w_2d.reshape(module.weight.shape))

        def _forward_conv2d(x, *, _mod=module):
            w_csr = _mod._wsparse_csr
            B = x.shape[0]
            kernel_size = (_mod.kernel_size if isinstance(_mod.kernel_size, tuple)
                          else (_mod.kernel_size, _mod.kernel_size))
            padding = (_mod.padding if isinstance(_mod.padding, tuple)
                      else (_mod.padding, _mod.padding))
            stride = (_mod.stride if isinstance(_mod.stride, tuple)
                     else (_mod.stride, _mod.stride))

            # im2col: (B, C_in*kH*kW, L)
            x_unfold = F.unfold(x, kernel_size=kernel_size,
                                padding=padding, stride=stride)
            K = x_unfold.shape[1]  # C_in * kH * kW
            L = x_unfold.shape[2]  # output spatial locations

            H_out = (x.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            W_out = (x.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            C_out = _mod._wsparse_csr.shape[0]

            # Sparse matmul per batch element
            outputs = []
            for b in range(B):
                # x_col: (K, L),  W_csr: (C_out, K)
                x_col = x_unfold[b]
                out_b = torch.sparse.mm(w_csr, x_col)  # (C_out, L)
                outputs.append(out_b)
            out = torch.stack(outputs, dim=0)  # (B, C_out, L)

            if _mod.bias is not None:
                out = out + _mod.bias.unsqueeze(0).unsqueeze(-1)

            return out.reshape(B, C_out, H_out, W_out)

        module.forward = _forward_conv2d
