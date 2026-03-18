import torch
import torch.nn as nn
import torch.nn.functional as F

from dmEngine.backends.base import WeightSparseBackend
from dmModels.sparse_weights import sparsify_weight

try:
    import torch_sputnik as ts
    _HAS_SPUTNIK = True
except ImportError:
    _HAS_SPUTNIK = False


def _to_sputnik_csr(weight_2d):
    """Convert 2D sparse weight to Sputnik CSR format.

    Sputnik CSR requires row_indices sorted by nnz per row in DESCENDING
    order for load balancing.

    Returns:
        (values, row_indices, row_offsets, col_indices)
    """
    csr = weight_2d.to_sparse_csr()
    row_offsets = csr.crow_indices().to(torch.int32)
    col_indices = csr.col_indices().to(torch.int32)
    values = csr.values()
    # Sort rows by nnz descending for Sputnik load balancing
    nnz_per_row = row_offsets[1:] - row_offsets[:-1]
    row_indices = torch.argsort(nnz_per_row, descending=True).to(torch.int32)
    return values, row_indices, row_offsets, col_indices


class SputnikWSparseBackend(WeightSparseBackend):
    """Sputnik weight-sparse backend.

    Same matmul strategy as TorchCSR (y = W_sparse @ x.T then transpose)
    but uses Sputnik SpMM for the sparse-dense multiply.

    Sputnik CSR components are precomputed once at prepare() time and stored
    on each module for reuse every forward call.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._original_state = {}  # {name: (module, original_forward, original_weight)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module, sparsity: float) -> nn.Module:
        if not _HAS_SPUTNIK:
            raise RuntimeError(
                "torch_sputnik is not installed. Install it to use the "
                "SputnikWSparseBackend."
            )

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._prepare_linear(name, module, sparsity)
            elif isinstance(module, nn.Conv2d):
                if module.groups != 1 or module.dilation != (1, 1):
                    continue
                self._prepare_conv2d(name, module, sparsity)
        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        for _name, (module, orig_forward, orig_weight) in self._original_state.items():
            module.forward = orig_forward
            module.weight = nn.Parameter(orig_weight)
            for attr in ('_sputnik_values', '_sputnik_row_indices',
                         '_sputnik_row_offsets', '_sputnik_col_indices',
                         '_sputnik_rows', '_sputnik_cols_a', '_sputnik_nnz'):
                if hasattr(module, attr):
                    delattr(module, attr)
        self._original_state.clear()
        return model

    def supported_sparsities(self) -> list:
        return []  # any sparsity

    @property
    def name(self) -> str:
        return 'SputnikWSparse'

    # ------------------------------------------------------------------
    # Linear
    # ------------------------------------------------------------------

    def _prepare_linear(self, name: str, module: nn.Linear, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        # Sparsify weight — shape (out_features, in_features)
        sparse_w = sparsify_weight(module.weight.data, sparsity)
        module.weight = nn.Parameter(sparse_w)

        # Convert to Sputnik CSR and store on module
        values, row_indices, row_offsets, col_indices = _to_sputnik_csr(sparse_w)
        module._sputnik_values = values
        module._sputnik_row_indices = row_indices
        module._sputnik_row_offsets = row_offsets
        module._sputnik_col_indices = col_indices
        module._sputnik_rows = sparse_w.shape[0]
        module._sputnik_cols_a = sparse_w.shape[1]
        module._sputnik_nnz = values.numel()

        def _forward_linear(x, *, _mod=module):
            leading = x.shape[:-1]
            in_f = x.shape[-1]
            x_2d = x.reshape(-1, in_f)  # (M, in_f)
            cols_b = x_2d.shape[0]

            # ts.spmm(rows, cols_a, cols_b, nnz,
            #          row_indices, values, row_offsets, col_indices, dense_rhs)
            # dense_rhs shape: (in_f, M) i.e. x.T
            out = ts.spmm(
                _mod._sputnik_rows, _mod._sputnik_cols_a, cols_b,
                _mod._sputnik_nnz,
                _mod._sputnik_row_indices, _mod._sputnik_values,
                _mod._sputnik_row_offsets, _mod._sputnik_col_indices,
                x_2d.t(),
            ).t()  # (M, out_f)

            if _mod.bias is not None:
                out = out + _mod.bias
            return out.reshape(*leading, -1)

        module.forward = _forward_linear

    # ------------------------------------------------------------------
    # Conv2d  (im2col + Sputnik SpMM)
    # ------------------------------------------------------------------

    def _prepare_conv2d(self, name: str, module: nn.Conv2d, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        C_out = module.weight.shape[0]
        w_2d = module.weight.data.reshape(C_out, -1)
        sparse_w_2d = sparsify_weight(w_2d, sparsity)
        module.weight = nn.Parameter(sparse_w_2d.reshape(module.weight.shape))

        # Convert to Sputnik CSR and store on module
        values, row_indices, row_offsets, col_indices = _to_sputnik_csr(sparse_w_2d)
        module._sputnik_values = values
        module._sputnik_row_indices = row_indices
        module._sputnik_row_offsets = row_offsets
        module._sputnik_col_indices = col_indices
        module._sputnik_rows = sparse_w_2d.shape[0]
        module._sputnik_cols_a = sparse_w_2d.shape[1]
        module._sputnik_nnz = values.numel()

        def _forward_conv2d(x, *, _mod=module):
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
            L = x_unfold.shape[2]

            H_out = (x.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            W_out = (x.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            C_out = _mod._sputnik_rows

            outputs = []
            for b in range(B):
                # x_col: (K, L),  W: (C_out, K)
                x_col = x_unfold[b]  # (K, L)
                out_b = ts.spmm(
                    _mod._sputnik_rows, _mod._sputnik_cols_a, L,
                    _mod._sputnik_nnz,
                    _mod._sputnik_row_indices, _mod._sputnik_values,
                    _mod._sputnik_row_offsets, _mod._sputnik_col_indices,
                    x_col,
                )  # (C_out, L)
                outputs.append(out_b)
            out = torch.stack(outputs, dim=0)  # (B, C_out, L)

            if _mod.bias is not None:
                out = out + _mod.bias.unsqueeze(0).unsqueeze(-1)

            return out.reshape(B, C_out, H_out, W_out)

        module.forward = _forward_conv2d
