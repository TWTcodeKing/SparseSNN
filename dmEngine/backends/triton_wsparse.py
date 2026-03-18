"""Custom Triton kernel backend for weight-sparse matmul.

Uses a CSR-based SpMM kernel: weight W is stored in CSR format (precomputed
once at prepare time), and each Triton program processes a tile of output rows,
iterating only over non-zero elements per row.

This correctly scales with element-level sparsity — unlike column-pruning which
is ineffective for random unstructured sparsity (nearly all columns have at
least one non-zero entry when the matrix has many rows).

Supports both nn.Linear and nn.Conv2d (via im2col).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmEngine.backends.base import WeightSparseBackend
from dmModels.sparse_weights import sparsify_weight

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ======================================================================
# Triton CSR SpMM kernel
# ======================================================================

if _HAS_TRITON:

    @triton.jit
    def csr_spmm_kernel(
        # CSR weight pointers
        row_offsets_ptr, col_indices_ptr, values_ptr,
        # Dense input and output
        X_ptr, Out_ptr,
        # Dimensions
        N_rows, N_cols_out,  # N_rows = out_features, N_cols_out = M (batch*spatial)
        stride_xk, stride_xm,  # X is (K_full, M) i.e. input transposed
        stride_om, stride_on,  # Out is (M, N_rows)
        # Block sizes
        BLOCK_M: tl.constexpr,
    ):
        """CSR SpMM:  Out = (W_csr @ X_t).T

        W_csr    : (N_rows, K_full) in CSR format — sparse weight
        X_t      : (K_full, M)      dense input transposed
        Out      : (M, N_rows)

        Each program handles one output row (one row of W_csr).
        It iterates over non-zero elements in that row, gathers the
        corresponding input row from X_t, and accumulates.
        """
        row_id = tl.program_id(0)  # which row of W to process
        if row_id >= N_rows:
            return

        # Get the range of non-zero elements for this row
        row_start = tl.load(row_offsets_ptr + row_id)
        row_end = tl.load(row_offsets_ptr + row_id + 1)

        # Process output columns in blocks of BLOCK_M
        col_block_id = tl.program_id(1)
        offs_m = col_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N_cols_out

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Iterate over non-zero elements in this row
        for nz_idx in range(row_start, row_end):
            col = tl.load(col_indices_ptr + nz_idx)
            val = tl.load(values_ptr + nz_idx)

            # Gather input: X_t[col, offs_m]
            x_ptrs = X_ptr + col * stride_xk + offs_m * stride_xm
            x_vals = tl.load(x_ptrs, mask=m_mask, other=0.0)

            acc += val * x_vals

        # Store: Out[offs_m, row_id]
        out_ptrs = Out_ptr + offs_m * stride_om + row_id * stride_on
        tl.store(out_ptrs, acc, mask=m_mask)


# ======================================================================
# Helpers
# ======================================================================

def _csr_spmm(row_offsets, col_indices, values, x_2d, N_rows, bias=None):
    """Launch the Triton CSR SpMM kernel.

    Computes Out = (W_csr @ x_2d.T).T  where W_csr is (N_rows, K) sparse.

    Args:
        row_offsets: (N_rows+1,) int32
        col_indices: (nnz,) int32
        values:      (nnz,) float32
        x_2d:        (M, K) dense input
        N_rows:      number of output features (rows of W)
        bias:        (N_rows,) or None

    Returns:
        (M, N_rows) output tensor
    """
    M, K = x_2d.shape
    # Transpose input for coalesced access: X_t is (K, M)
    x_t = x_2d.t().contiguous()

    out = torch.zeros((M, N_rows), device=x_2d.device, dtype=torch.float32)

    BLOCK_M = 128
    grid = (N_rows, triton.cdiv(M, BLOCK_M))

    csr_spmm_kernel[grid](
        row_offsets, col_indices, values,
        x_t, out,
        N_rows, M,
        x_t.stride(0), x_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
    )

    if bias is not None:
        out = out + bias.unsqueeze(0)

    return out


# ======================================================================
# Backend class
# ======================================================================

class TritonWSparseBackend(WeightSparseBackend):
    """Weight-sparse backend using a custom Triton CSR SpMM kernel.

    Each weight matrix is stored in CSR format (precomputed once in prepare).
    The Triton kernel iterates only over non-zero weight elements per row,
    so computation scales linearly with actual sparsity.

    Configuration keys (passed via *config* dict):
        min_tensor_size : int
            Minimum ``out_features * in_features`` to apply sparsification
            (default 4096).  Smaller layers fall back to dense computation.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._original_state = {}  # {name: (module, orig_forward, orig_weight)}
        self._min_tensor_size = config.get('min_tensor_size', 4096) if config else 4096

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module, sparsity: float) -> nn.Module:
        if not _HAS_TRITON:
            raise RuntimeError(
                "Triton is not installed.  Install with: pip install triton"
            )

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.numel() < self._min_tensor_size:
                    continue
                self._prepare_linear(name, module, sparsity)
            elif isinstance(module, nn.Conv2d):
                if module.groups != 1 or module.dilation != (1, 1):
                    continue
                if module.weight.numel() < self._min_tensor_size:
                    continue
                self._prepare_conv2d(name, module, sparsity)

        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        for _name, (module, orig_forward, orig_weight) in self._original_state.items():
            module.forward = orig_forward
            module.weight = nn.Parameter(orig_weight)
            for attr in ('_wcsr_row_offsets', '_wcsr_col_indices',
                         '_wcsr_values', '_wcsr_nrows'):
                if hasattr(module, attr):
                    delattr(module, attr)
        self._original_state.clear()
        return model

    def supported_sparsities(self) -> list:
        return []  # any sparsity

    @property
    def name(self) -> str:
        return 'TritonWSparse'

    # ------------------------------------------------------------------
    # Internal: convert weight to CSR and store on module
    # ------------------------------------------------------------------

    @staticmethod
    def _weight_to_csr(w_2d):
        """Convert 2D weight to CSR components stored as (row_offsets, col_indices, values)."""
        csr = w_2d.to_sparse_csr()
        row_offsets = csr.crow_indices().to(torch.int32).contiguous()
        col_indices = csr.col_indices().to(torch.int32).contiguous()
        values = csr.values().to(torch.float32).contiguous()
        return row_offsets, col_indices, values

    # ------------------------------------------------------------------
    # Linear
    # ------------------------------------------------------------------

    def _prepare_linear(self, name: str, module: nn.Linear, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        with torch.no_grad():
            sparse_w = sparsify_weight(module.weight.data, sparsity)
            module.weight = nn.Parameter(sparse_w)

            row_offsets, col_indices, values = self._weight_to_csr(sparse_w)
            module._wcsr_row_offsets = row_offsets
            module._wcsr_col_indices = col_indices
            module._wcsr_values = values
            module._wcsr_nrows = sparse_w.shape[0]

        def _forward_linear(x, *, _mod=module):
            leading = x.shape[:-1]
            in_f = x.shape[-1]
            x_2d = x.reshape(-1, in_f).contiguous().float()

            out = _csr_spmm(
                _mod._wcsr_row_offsets, _mod._wcsr_col_indices,
                _mod._wcsr_values, x_2d, _mod._wcsr_nrows,
                bias=_mod.bias,
            )
            return out.to(x.dtype).reshape(*leading, -1)

        module.forward = _forward_linear

    # ------------------------------------------------------------------
    # Conv2d (im2col + Triton CSR SpMM)
    # ------------------------------------------------------------------

    def _prepare_conv2d(self, name: str, module: nn.Conv2d, sparsity: float):
        orig_forward = module.forward
        orig_weight = module.weight.data.clone()
        self._original_state[name] = (module, orig_forward, orig_weight)

        C_out = module.weight.shape[0]

        with torch.no_grad():
            w_2d = module.weight.data.reshape(C_out, -1)
            sparse_w_2d = sparsify_weight(w_2d, sparsity)
            module.weight = nn.Parameter(sparse_w_2d.reshape(module.weight.shape))

            row_offsets, col_indices, values = self._weight_to_csr(sparse_w_2d)
            module._wcsr_row_offsets = row_offsets
            module._wcsr_col_indices = col_indices
            module._wcsr_values = values
            module._wcsr_nrows = C_out

        def _forward_conv2d(x, *, _mod=module):
            B = x.shape[0]
            kernel_size = (_mod.kernel_size if isinstance(_mod.kernel_size, tuple)
                          else (_mod.kernel_size, _mod.kernel_size))
            padding = (_mod.padding if isinstance(_mod.padding, tuple)
                      else (_mod.padding, _mod.padding))
            stride = (_mod.stride if isinstance(_mod.stride, tuple)
                     else (_mod.stride, _mod.stride))

            # im2col: (B, K, L) where K = C_in*kH*kW, L = H_out*W_out
            x_unfold = F.unfold(x, kernel_size=kernel_size,
                                padding=padding, stride=stride)
            K = x_unfold.shape[1]
            L = x_unfold.shape[2]

            H_out = (x.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            W_out = (x.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            C_out = _mod._wcsr_nrows

            # Reshape: (B, K, L) → (B*L, K) for batched SpMM
            x_col = x_unfold.permute(0, 2, 1).reshape(B * L, K).contiguous().float()

            out_flat = _csr_spmm(
                _mod._wcsr_row_offsets, _mod._wcsr_col_indices,
                _mod._wcsr_values, x_col, C_out,
                bias=_mod.bias,
            )  # (B*L, C_out)

            out = out_flat.to(x.dtype).reshape(B, L, C_out).permute(0, 2, 1)
            return out.reshape(B, C_out, H_out, W_out)

        module.forward = _forward_conv2d
