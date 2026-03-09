"""Sparse Linear layer acceleration using Sputnik SpMM.

Intercepts nn.Linear forward passes via hooks. When the input activation
(spike tensor) is sufficiently sparse, converts it to Sputnik's sparse
format and uses SpMM instead of dense matmul.

Sputnik SpMM operates on CSR-format sparse matrices and performs:
    output = sparse_input @ weight.T + bias
which replaces the standard nn.Linear: output = input @ weight.T + bias

For SNN models, input activations after LIF neurons are binary spike tensors
with ~6-8% density, making SpMM significantly faster than dense matmul.
"""

import os
import ctypes

import torch
import torch.nn as nn
from typing import Optional

from iengine.common.density import measure_density, should_use_sparse

# Lazy import for torch_sputnik — may not be installed
_torch_sputnik = None
_sputnik_available = None

# Paths for Sputnik runtime libraries
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SPUTNIK_LIB = os.path.join(_SCRIPT_DIR, "third_party", "sputnik", "build", "lib")
_LOCAL_LIB = os.path.join(os.path.expanduser("~"), ".local", "lib")


def _ensure_ld_paths():
    """Add Sputnik and dependency library paths to the dynamic linker search.

    This must happen BEFORE `import torch_sputnik` so that the shared
    libraries (libsputnik.so, libglog.so, libgflags.so) can be found.
    We also preload PyTorch's libc10.so which torch_sputnik depends on.
    """
    # 1. Update LD_LIBRARY_PATH for any child processes
    extra_dirs = [_SPUTNIK_LIB, _LOCAL_LIB]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        extra_dirs.insert(0, torch_lib)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    for d in extra_dirs:
        if d not in current:
            current = d + ":" + current if current else d
    os.environ["LD_LIBRARY_PATH"] = current

    # 2. Pre-load critical shared libs via ctypes so the current process
    #    can resolve them (LD_LIBRARY_PATH changes don't affect dlopen
    #    retroactively in the same process).
    for lib_dir, lib_name in [
        (torch_lib, "libc10.so"),
        (_LOCAL_LIB, "libgflags.so"),
        (_LOCAL_LIB, "libglog.so"),
        (_SPUTNIK_LIB, "libsputnik.so"),
    ]:
        path = os.path.join(lib_dir, lib_name)
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # best-effort; import will fail with a clear error later


def _check_sputnik():
    """Check if torch_sputnik is available. Caches result."""
    global _torch_sputnik, _sputnik_available
    if _sputnik_available is not None:
        return _sputnik_available
    _ensure_ld_paths()
    try:
        import torch_sputnik
        _torch_sputnik = torch_sputnik
        _sputnik_available = True
    except ImportError:
        _sputnik_available = False
    return _sputnik_available


def _require_sputnik():
    """Raise ImportError with build instructions if torch_sputnik is missing."""
    if not _check_sputnik():
        raise ImportError(
            "torch_sputnik is not installed. Build from source:\n"
            "  bash iengine/sputnik_sparse/build.sh\n"
            "See iengine/sputnik_sparse/INSTALL.md for details."
        )
    return _torch_sputnik


def _to_sputnik_csr(dense_2d: torch.Tensor):
    """Convert a dense 2D tensor to Sputnik-compatible CSR format.

    Sputnik expects CSR with:
        - row_offsets: int32, length (rows + 1)
        - col_indices: int32, length nnz
        - values: float16/float32, length nnz
        - row_indices: int32, sorted by nnz per row (descending)

    Returns:
        (values, row_indices, row_offsets, col_indices, nnz)
    """
    assert dense_2d.ndim == 2, f"Expected 2D tensor, got {dense_2d.ndim}D"
    csr = dense_2d.to_sparse_csr()
    row_offsets = csr.crow_indices().to(torch.int32)
    col_indices = csr.col_indices().to(torch.int32)
    values = csr.values()
    nnz = values.shape[0]

    # Sputnik requires row_indices sorted by nnz per row (descending).
    # Compute nnz per row from row_offsets.
    nnz_per_row = row_offsets[1:] - row_offsets[:-1]
    row_indices = torch.argsort(nnz_per_row, descending=True).to(torch.int32)

    return values, row_indices, row_offsets, col_indices, nnz


def sputnik_spmm(sparse_input: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Perform sparse_input @ weight.T using Sputnik SpMM.

    Args:
        sparse_input: (M, K) sparse activation tensor (mostly zeros).
        weight: (N, K) dense weight matrix from nn.Linear.
        bias: Optional (N,) bias vector.

    Returns:
        (M, N) output tensor.
    """
    ts = _require_sputnik()
    M, K = sparse_input.shape
    N = weight.shape[0]

    values, row_indices, row_offsets, col_indices, nnz = _to_sputnik_csr(
        sparse_input)

    # Sputnik spmm: sparse(M, K) @ dense(K, N) = dense(M, N)
    # Weight is (N, K), we need (K, N) = weight.T
    weight_t = weight.t().contiguous()

    output = ts.spmm(
        M, K, N, nnz,
        row_indices, values, row_offsets, col_indices,
        weight_t
    )

    if bias is not None:
        output = output + bias.unsqueeze(0)

    return output


class SputnikLinearStats:
    """Tracks statistics for Sputnik sparse Linear execution."""

    def __init__(self):
        self.total_calls = 0
        self.sparse_calls = 0
        self.dense_calls = 0
        self.total_ops = 0
        self.effective_ops = 0
        self.per_layer = {}

    def record(self, layer_name: str, m: int, k: int, n: int,
               density: float, used_sparse: bool):
        ops = m * k * n
        self.total_calls += 1
        self.total_ops += ops
        if used_sparse:
            self.sparse_calls += 1
            self.effective_ops += int(ops * density)
        else:
            self.dense_calls += 1
            self.effective_ops += ops

        if layer_name not in self.per_layer:
            self.per_layer[layer_name] = {
                'total_ops': 0, 'effective_ops': 0,
                'sparse_calls': 0, 'dense_calls': 0,
                'densities': [],
            }
        stats = self.per_layer[layer_name]
        stats['total_ops'] += ops
        stats['sparse_calls' if used_sparse else 'dense_calls'] += 1
        stats['densities'].append(density)
        if used_sparse:
            stats['effective_ops'] += int(ops * density)
        else:
            stats['effective_ops'] += ops

    def to_dict(self) -> dict:
        per_layer = {}
        for name, s in self.per_layer.items():
            densities = s['densities']
            per_layer[name] = {
                'total_ops': s['total_ops'],
                'effective_ops': s['effective_ops'],
                'density': sum(densities) / len(densities) if densities else 1.0,
                'sparse_calls': s['sparse_calls'],
                'dense_calls': s['dense_calls'],
            }
        overall_density = (self.effective_ops / self.total_ops
                           if self.total_ops > 0 else 1.0)
        return {
            'total_ops': self.total_ops,
            'effective_ops': self.effective_ops,
            'density': overall_density,
            'per_layer': per_layer,
            'total_calls': self.total_calls,
            'sparse_calls': self.sparse_calls,
            'dense_calls': self.dense_calls,
        }

    def reset(self):
        self.__init__()


def make_sparse_linear_forward(module: nn.Module, original_forward,
                                layer_name: str, stats: SputnikLinearStats,
                                density_threshold: float = 0.15,
                                min_elements: int = 4096,
                                enabled_ref: list = None):
    """Create a replacement forward for nn.Linear using Sputnik SpMM.

    When density is below threshold: Sputnik SpMM (skips dense entirely).
    When density is above threshold: calls original_forward (no redundancy).

    Args:
        module: The nn.Linear module.
        original_forward: The original forward method to fall back to.
        layer_name: Name for statistics tracking.
        stats: SputnikLinearStats instance for recording metrics.
        density_threshold: Maximum density to use sparse path.
        min_elements: Minimum elements to justify sparse overhead.
        enabled_ref: Mutable list [bool] — checks enabled_ref[0].

    Returns:
        New forward function that replaces module.forward.
    """
    if not _check_sputnik():
        raise ImportError(
            "torch_sputnik is not installed. Build from source:\n"
            "  bash iengine/sputnik_sparse/build.sh"
        )

    def sparse_forward(x):
        if enabled_ref and not enabled_ref[0]:
            return original_forward(x)

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, K = x_2d.shape
        N = module.weight.shape[0]

        density = measure_density(x_2d)

        if not should_use_sparse(x_2d, threshold=density_threshold,
                                 min_elements=min_elements):
            stats.record(layer_name, M, K, N, density, used_sparse=False)
            return original_forward(x)

        try:
            result = sputnik_spmm(x_2d, module.weight, module.bias)
            out_shape = orig_shape[:-1] + (N,)
            result = result.reshape(out_shape)
            stats.record(layer_name, M, K, N, density, used_sparse=True)
            return result
        except Exception:
            stats.record(layer_name, M, K, N, density, used_sparse=False)
            return original_forward(x)

    return sparse_forward
