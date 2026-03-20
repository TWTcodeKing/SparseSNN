"""Sputnik CUDA library loading and CSR format helpers.

Handles lazy import of torch_sputnik, LD_LIBRARY_PATH setup, and
conversion from dense tensors to Sputnik's CSR format (with row_indices
sorted by nnz-per-row descending, as required by Sputnik SpMM).
"""

import os
import ctypes

import torch

# Lazy import for torch_sputnik — may not be installed
_torch_sputnik = None
_sputnik_available = None

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SPUTNIK_LIB = os.path.join(_SCRIPT_DIR, "third_party", "sputnik", "build", "lib")
_LOCAL_LIB = os.path.join(os.path.expanduser("~"), ".local", "lib")


def _ensure_ld_paths():
    """Add Sputnik library paths to the dynamic linker search."""
    extra_dirs = [_SPUTNIK_LIB, _LOCAL_LIB]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        extra_dirs.insert(0, torch_lib)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    for d in extra_dirs:
        if d not in current:
            current = d + ":" + current if current else d
    os.environ["LD_LIBRARY_PATH"] = current

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
                pass


def check_sputnik():
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


def require_sputnik():
    """Return torch_sputnik module, or raise ImportError with build instructions."""
    if not check_sputnik():
        raise ImportError(
            "torch_sputnik is not installed. Build from source:\n"
            "  bash iengine/sputnik_sparse/build.sh\n"
            "See iengine/sputnik_sparse/INSTALL.md for details."
        )
    return _torch_sputnik


def to_sputnik_csr(dense_2d: torch.Tensor):
    """Convert a dense 2D tensor to Sputnik-compatible CSR format.

    Sputnik requires row_indices sorted by nnz-per-row descending.

    Returns:
        (values, row_indices, row_offsets, col_indices, nnz)
    """
    assert dense_2d.ndim == 2
    csr = dense_2d.to_sparse_csr()
    row_offsets = csr.crow_indices().to(torch.int32)
    col_indices = csr.col_indices().to(torch.int32)
    values = csr.values()
    nnz = values.shape[0]

    nnz_per_row = row_offsets[1:] - row_offsets[:-1]
    row_indices = torch.argsort(nnz_per_row, descending=True).to(torch.int32)

    return values, row_indices, row_offsets, col_indices, nnz


def sputnik_spmm(sparse_input, weight, bias=None):
    """Sparse_input @ weight.T using Sputnik SpMM.

    Args:
        sparse_input: (M, K) sparse activation tensor.
        weight: (N, K) dense weight matrix.
        bias: Optional (N,) bias.

    Returns:
        (M, N) output tensor.
    """
    ts = require_sputnik()
    M, K = sparse_input.shape
    N = weight.shape[0]

    values, row_indices, row_offsets, col_indices, nnz = to_sputnik_csr(sparse_input)
    weight_t = weight.t().contiguous()

    output = ts.spmm(M, K, N, nnz, row_indices, values, row_offsets,
                      col_indices, weight_t)
    if bias is not None:
        output = output + bias.unsqueeze(0)
    return output
