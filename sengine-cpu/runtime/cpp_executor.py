"""ctypes wrapper for the C CPU executor (libsengine_cpu.so).

Matches the API in csrc/cpu_executor.h. No Python in the hot loop —
all kernel dispatch happens in C.
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from sengine_cpu.logger import log


def _find_lib() -> str:
    """Locate libsengine_cpu.so."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'csrc', 'libsengine_cpu.so'),
        os.path.join(os.path.dirname(__file__), 'libsengine_cpu.so'),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "libsengine_cpu.so not found. Build it with: cd sengine-cpu/csrc && make")


def _preload_blas() -> str | None:
    """Load scipy's bundled OpenBLAS with RTLD_GLOBAL so the C runtime can
    dlsym('scipy_cblas_sgemm') for its conv / classifier GEMMs."""
    import glob
    try:
        import scipy
    except ImportError:
        return None
    libs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(scipy.__file__))), 'scipy.libs')
    for path in sorted(glob.glob(os.path.join(libs_dir, 'libscipy_openblas-*.so'))):
        if '64_' in os.path.basename(path):
            continue  # ILP64 build: incompatible int width
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return path
        except OSError:
            continue
    return None


class CPUCppExecutor:
    """Python wrapper around the C CPU executor."""

    def __init__(self, n_threads: int | None = None):
        lib_path = _find_lib()
        blas = _preload_blas()
        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

        n_threads = n_threads or os.cpu_count() or 1
        self._handle = self._lib.sengine_cpu_create(n_threads)
        if not self._handle:
            raise RuntimeError("sengine_cpu_create returned NULL")
        self._lib.sengine_blas_set_threads(n_threads)
        log.info("BLAS: %s", blas or "none (naive GEMM fallback)")
        self._membranes: list[np.ndarray] = []
        log.info("C executor: loaded %s, %d threads", lib_path, n_threads)

    def _setup_signatures(self):
        lib = self._lib
        c_int = ctypes.c_int
        c_float = ctypes.c_float
        c_double = ctypes.c_double
        c_char_p = ctypes.c_char_p
        c_void_p = ctypes.c_void_p
        float_p = ctypes.POINTER(ctypes.c_float)
        int_p = ctypes.POINTER(ctypes.c_int)
        void_pp = ctypes.POINTER(c_void_p)

        lib.sengine_cpu_create.argtypes = [c_int]
        lib.sengine_cpu_create.restype = c_void_p

        lib.sengine_cpu_destroy.argtypes = [c_void_p]
        lib.sengine_cpu_destroy.restype = None

        lib.sengine_cpu_load_tvm.argtypes = [c_void_p, c_char_p]
        lib.sengine_cpu_load_tvm.restype = c_int

        lib.sengine_cpu_set_schedule.argtypes = [c_void_p, int_p, c_int]
        lib.sengine_cpu_set_schedule.restype = None

        lib.sengine_cpu_alloc_nodes.argtypes = [c_void_p, c_int]
        lib.sengine_cpu_alloc_nodes.restype = None

        lib.sengine_cpu_execute.argtypes = [c_void_p]
        lib.sengine_cpu_execute.restype = None

        lib.sengine_cpu_benchmark.argtypes = [c_void_p, c_int, c_int]
        lib.sengine_cpu_benchmark.restype = c_double

        lib.sengine_cpu_reset_membranes.argtypes = [c_void_p]
        lib.sengine_cpu_reset_membranes.restype = None

        lib.sengine_cpu_register_membrane.argtypes = [c_void_p, float_p, c_int]
        lib.sengine_cpu_register_membrane.restype = None

        # TVM node
        lib.sengine_cpu_set_tvm_node.argtypes = [
            c_void_p, c_int, c_int, void_pp, c_int]
        lib.sengine_cpu_set_tvm_node.restype = None

        # IF node
        lib.sengine_cpu_set_if_node.argtypes = [
            c_void_p, c_int, float_p, float_p, float_p, c_int, c_int, c_float]
        lib.sengine_cpu_set_if_node.restype = None

        # LIF node
        lib.sengine_cpu_set_lif_node.argtypes = [
            c_void_p, c_int, float_p, float_p, float_p,
            c_int, c_int, c_float, c_float]
        lib.sengine_cpu_set_lif_node.restype = None

        # Add node
        lib.sengine_cpu_set_add_node.argtypes = [
            c_void_p, c_int, float_p, float_p, float_p, c_int]
        lib.sengine_cpu_set_add_node.restype = None

        # MaxPool node
        lib.sengine_cpu_set_maxpool_node.argtypes = [
            c_void_p, c_int, float_p, float_p,
            c_int, c_int, c_int, c_int, c_int, c_int,
            c_int, c_int, c_int, c_int, c_int, c_int]
        lib.sengine_cpu_set_maxpool_node.restype = None

        # GlobalAvgPool node
        lib.sengine_cpu_set_gavg_node.argtypes = [
            c_void_p, c_int, float_p, float_p, c_int, c_int, c_int, c_int]
        lib.sengine_cpu_set_gavg_node.restype = None

        # TemporalMean node
        lib.sengine_cpu_set_tmean_node.argtypes = [
            c_void_p, c_int, float_p, float_p, c_int, c_int]
        lib.sengine_cpu_set_tmean_node.restype = None

        # GEMM node
        lib.sengine_cpu_set_gemm_node.argtypes = [
            c_void_p, c_int, float_p, float_p, float_p, c_int, c_int, c_int]
        lib.sengine_cpu_set_gemm_node.restype = None

        # Softmax node
        lib.sengine_cpu_set_softmax_node.argtypes = [
            c_void_p, c_int, float_p, float_p, c_int, c_int]
        lib.sengine_cpu_set_softmax_node.restype = None

        # Skip node
        lib.sengine_cpu_set_skip_node.argtypes = [c_void_p, c_int]
        lib.sengine_cpu_set_skip_node.restype = None

        # Alias node
        lib.sengine_cpu_set_alias_node.argtypes = [c_void_p, c_int, float_p, float_p, c_int]
        lib.sengine_cpu_set_alias_node.restype = None
        # Native fused conv node
        lib.sengine_cpu_set_conv_node.argtypes = (
            [c_void_p, c_int] + [float_p] * 6 + [c_int] * 11 + [c_float] * 3)
        lib.sengine_cpu_set_conv_node.restype = None
        # GEMM bias
        lib.sengine_cpu_set_gemm_bias.argtypes = [c_void_p, c_int, float_p]
        lib.sengine_cpu_set_gemm_bias.restype = None
        lib.sengine_blas_set_threads.argtypes = [c_int]
        lib.sengine_blas_set_threads.restype = None

    # ─── Public API ───────────────────────────────────────────

    def alloc_nodes(self, max_id: int):
        self._lib.sengine_cpu_alloc_nodes(self._handle, max_id)

    def set_schedule(self, order: list[int]):
        arr = (ctypes.c_int * len(order))(*order)
        self._lib.sengine_cpu_set_schedule(self._handle, arr, len(order))

    def load_tvm(self, so_path: str) -> int:
        idx = self._lib.sengine_cpu_load_tvm(
            self._handle, so_path.encode('utf-8'))
        if idx < 0:
            raise RuntimeError(f"Failed to load TVM kernel: {so_path}")
        return idx

    def register_membrane(self, arr: np.ndarray):
        assert arr.dtype == np.float32
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.sengine_cpu_register_membrane(self._handle, ptr, arr.size)
        self._membranes.append(arr)

    def reset_membranes(self):
        self._lib.sengine_cpu_reset_membranes(self._handle)

    def execute(self):
        self._lib.sengine_cpu_execute(self._handle)

    def benchmark(self, warmup: int = 100, iters: int = 500) -> float:
        return self._lib.sengine_cpu_benchmark(self._handle, warmup, iters)

    # ─── Node registration helpers ────────────────────────────

    @staticmethod
    def _fp(arr: np.ndarray):
        """Get float pointer from numpy array."""
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def set_tvm_node(self, nid: int, tvm_idx: int, args: list[np.ndarray]):
        n_args = len(args)
        arr_type = ctypes.c_void_p * n_args
        ptrs = arr_type(*[a.ctypes.data for a in args])
        self._lib.sengine_cpu_set_tvm_node(
            self._handle, nid, tvm_idx, ptrs, n_args)

    def set_if_node(self, nid: int, inp: np.ndarray, out: np.ndarray,
                    membrane: np.ndarray, total: int, spatial: int, thresh: float):
        self._lib.sengine_cpu_set_if_node(
            self._handle, nid, self._fp(inp), self._fp(out),
            self._fp(membrane), total, spatial, thresh)

    def set_lif_node(self, nid: int, inp: np.ndarray, out: np.ndarray,
                     membrane: np.ndarray, total: int, spatial: int,
                     thresh: float, recip_tau: float):
        self._lib.sengine_cpu_set_lif_node(
            self._handle, nid, self._fp(inp), self._fp(out),
            self._fp(membrane), total, spatial, thresh, recip_tau)

    def set_add_node(self, nid: int, a: np.ndarray, b: np.ndarray,
                     out: np.ndarray, n: int):
        self._lib.sengine_cpu_set_add_node(
            self._handle, nid, self._fp(a), self._fp(b), self._fp(out), n)

    def set_skip_node(self, nid: int):
        self._lib.sengine_cpu_set_skip_node(self._handle, nid)

    def set_gemm_node(self, nid: int, a: np.ndarray, b: np.ndarray,
                      c: np.ndarray, M: int, K: int, N: int):
        self._lib.sengine_cpu_set_gemm_node(
            self._handle, nid, self._fp(a), self._fp(b), self._fp(c), M, K, N)

    def set_conv_node(self, nid: int, inp, weight, scale, bias, membrane, out,
                      B, H, W, C_in, F, T, KH, KW, pad, stride,
                      neuron: int, v_threshold: float, v_reset: float, recip_tau: float):
        """Native NHWC Conv+BN(+neuron). neuron: 0 none, 1 IF, 2 LIF."""
        self._lib.sengine_cpu_set_conv_node(
            self._handle, nid, self._fp(inp), self._fp(weight), self._fp(scale),
            self._fp(bias), self._fp(membrane), self._fp(out),
            int(B), int(H), int(W), int(C_in), int(F), int(T), int(KH), int(KW),
            int(pad), int(stride), int(neuron),
            float(v_threshold), float(v_reset), float(recip_tau))

    def set_gemm_bias(self, nid: int, bias):
        self._lib.sengine_cpu_set_gemm_bias(self._handle, nid, self._fp(bias))

    def destroy(self):
        if self._handle:
            self._lib.sengine_cpu_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()
