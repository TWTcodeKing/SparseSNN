"""ctypes wrapper for the standalone C++ CUDA executor (libsengine_exec.so).

Provides a Pythonic interface over the C API. No PyTorch dependency in
the C++ library — all tensor data is passed as raw GPU pointers.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

_LIB_PATH = os.path.join(os.path.dirname(__file__), '..', 'csrc', 'libsengine_exec.so')
_lib: Optional[ctypes.CDLL] = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(
            f"libsengine_exec.so not found at {_LIB_PATH}. "
            "Compile with: nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC "
            "-gencode=arch=compute_89,code=sm_89 "
            "-o sengine/csrc/libsengine_exec.so sengine/csrc/cpp_executor.cu "
            "-lcudart -ldl -lcublas"
        )
    _lib = ctypes.CDLL(_LIB_PATH)
    _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib):
    """Declare C function signatures for type safety."""
    VP = ctypes.c_void_p
    CI = ctypes.c_int
    CF = ctypes.c_float
    IP = ctypes.POINTER(ctypes.c_int)

    lib.sengine_create.restype = VP
    lib.sengine_destroy.argtypes = [VP]

    lib.sengine_load_tilelang.argtypes = [VP, ctypes.c_char_p]
    lib.sengine_load_tilelang.restype = CI

    lib.sengine_set_schedule.argtypes = [VP, IP, CI]
    lib.sengine_alloc_nodes.argtypes = [VP, CI]

    lib.sengine_set_tilelang_node_5.argtypes = [VP, CI, CI, VP, VP, VP, VP, VP]
    lib.sengine_set_tilelang_node_6.argtypes = [VP, CI, CI, VP, VP, VP, VP, VP, VP]
    lib.sengine_set_if_node.argtypes = [VP, CI, VP, VP, VP, CI, CI, CF]
    lib.sengine_set_lif_node.argtypes = [VP, CI, VP, VP, VP, CI, CI, CF, CF]
    lib.sengine_set_add_node.argtypes = [VP, CI, VP, VP, VP, CI]
    lib.sengine_set_maxpool_node.argtypes = [VP, CI, VP, VP, CI, CI, CI, CI, CI, CI, CI, CI, CI]
    lib.sengine_set_global_avgpool_node.argtypes = [VP, CI, VP, VP, CI, CI, CI, CI]
    lib.sengine_set_temporal_mean_node.argtypes = [VP, CI, VP, VP, CI, CI]
    lib.sengine_set_gemm_node.argtypes = [VP, CI, VP, VP, VP, CI, CI, CI]
    lib.sengine_set_skip_node.argtypes = [VP, CI]
    lib.sengine_set_alias_node.argtypes = [VP, CI, VP, VP, CI]
    lib.sengine_set_tilelang_node_3.argtypes = [VP, CI, CI, VP, VP, VP]
    lib.sengine_set_naive_conv_node.argtypes = [VP, CI, VP, VP, VP, VP, VP,
                                                 CI, CI, CI, CI, CI, CI, CI, CI, CI, CI, CI, CI]
    lib.sengine_set_layout_transpose_node.argtypes = [VP, CI, VP, VP, CI, CI, CI, CI, CI]
    lib.sengine_set_fused_attn_node.argtypes = [
        VP, CI,           # handle, nid
        CI, CI, CI,       # variant, gemm1_idx, gemm2_idx
        VP, VP, VP, VP,   # q, k, v, out
        VP, VP,           # workspace, membrane
        CI, CI, CI, CI, CI, CI,  # TB, heads, hd, N, H, W
        CI, CI,           # lif_total, lif_spatial
        ctypes.c_float, ctypes.c_float,  # v_thresh, recip_tau
        CI,               # needs_permute
        VP, VP,           # scale1_ptr, scale2_ptr
        CI, CI, CI, CI,   # ws_gemm1_out, ws_perm_q, ws_perm_k, ws_perm_v
    ]
    # Detection model ops
    lib.sengine_set_resize_node.argtypes = [VP, CI, VP, VP, CI, CI, CI, CI, CI, CI, CI, CI]
    lib.sengine_set_concat_node.argtypes = [VP, CI, VP, VP, VP, CI, CI, CI]
    lib.sengine_set_ilif_node.argtypes = [VP, CI, VP, VP, VP, CI, CI, CF, CF]
    lib.sengine_set_softmax_node.argtypes = [VP, CI, VP, VP, CI, CI]
    lib.sengine_add_membrane.argtypes = [VP, VP, CI]

    lib.sengine_execute.argtypes = [VP]
    lib.sengine_set_fp32.argtypes = [VP, CI]
    lib.sengine_reset_membranes.argtypes = [VP]
    lib.sengine_capture_graph.argtypes = [VP]
    lib.sengine_replay.argtypes = [VP]
    lib.sengine_execute_sequential_debug.argtypes = [VP]
    lib.sengine_sync.argtypes = [VP]

    lib.sengine_benchmark.argtypes = [VP, CI, CI]
    lib.sengine_benchmark.restype = CF


class CppExecutor:
    """Python wrapper around the C++ CUDA executor.

    Usage:
        exe = CppExecutor()
        exe.load_tilelang("/path/to/kernel.so")
        exe.set_schedule([1, 3, 5, 2, 4])
        exe.set_if_node(2, input_ptr, output_ptr, membrane_ptr, ...)
        ...
        exe.capture_graph()
        latency_ms = exe.benchmark(warmup=200, iters=1000)
        exe.destroy()
    """

    def __init__(self):
        lib = _load_lib()
        self._lib = lib
        self._handle = lib.sengine_create()
        self._tl_idx_cache: dict[str, int] = {}  # so_path → tl_idx

    def set_fp32(self, fp32: bool):
        self._lib.sengine_set_fp32(self._handle, 1 if fp32 else 0)

    def destroy(self):
        if self._handle:
            self._lib.sengine_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()

    # ─── Setup ───

    def alloc_nodes(self, max_id: int):
        self._lib.sengine_alloc_nodes(self._handle, max_id)

    def set_schedule(self, schedule: list[int]):
        arr = (ctypes.c_int * len(schedule))(*schedule)
        self._lib.sengine_set_schedule(self._handle, arr, len(schedule))

    def load_tilelang(self, so_path: str) -> int:
        """Load a standalone TileLang .so. Returns kernel index."""
        if so_path in self._tl_idx_cache:
            return self._tl_idx_cache[so_path]
        idx = self._lib.sengine_load_tilelang(self._handle, so_path.encode())
        if idx < 0:
            raise RuntimeError(f"Failed to load TileLang kernel: {so_path}")
        self._tl_idx_cache[so_path] = idx
        return idx

    def add_membrane(self, ptr: int, size: int):
        self._lib.sengine_add_membrane(self._handle, ptr, size)

    # ─── Node registration ───

    def set_tilelang_3(self, nid: int, tl_idx: int, *ptrs):
        self._lib.sengine_set_tilelang_node_3(self._handle, nid, tl_idx, *ptrs)

    def set_tilelang_5(self, nid: int, tl_idx: int, *ptrs):
        self._lib.sengine_set_tilelang_node_5(self._handle, nid, tl_idx, *ptrs)

    def set_tilelang_6(self, nid: int, tl_idx: int, *ptrs):
        self._lib.sengine_set_tilelang_node_6(self._handle, nid, tl_idx, *ptrs)

    def set_if_node(self, nid, input_ptr, output_ptr, mem_ptr,
                    total, spatial, v_thresh):
        self._lib.sengine_set_if_node(self._handle, nid,
            input_ptr, output_ptr, mem_ptr, total, spatial, v_thresh)

    def set_lif_node(self, nid, input_ptr, output_ptr, mem_ptr,
                     total, spatial, v_thresh, recip_tau):
        self._lib.sengine_set_lif_node(self._handle, nid,
            input_ptr, output_ptr, mem_ptr, total, spatial, v_thresh, recip_tau)

    def set_add_node(self, nid, a_ptr, b_ptr, out_ptr, n):
        self._lib.sengine_set_add_node(self._handle, nid, a_ptr, b_ptr, out_ptr, n)

    def set_maxpool_node(self, nid, in_ptr, out_ptr,
                         N, H, W, C, OH, OW, ks, stride, pad):
        self._lib.sengine_set_maxpool_node(self._handle, nid,
            in_ptr, out_ptr, N, H, W, C, OH, OW, ks, stride, pad)

    def set_global_avgpool_node(self, nid, in_ptr, out_ptr, N, H, W, C):
        self._lib.sengine_set_global_avgpool_node(self._handle, nid,
            in_ptr, out_ptr, N, H, W, C)

    def set_temporal_mean_node(self, nid, in_ptr, out_ptr, T, spatial):
        self._lib.sengine_set_temporal_mean_node(self._handle, nid,
            in_ptr, out_ptr, T, spatial)

    def set_gemm_node(self, nid, in_ptr, w_ptr, out_ptr, M, K, N):
        self._lib.sengine_set_gemm_node(self._handle, nid,
            in_ptr, w_ptr, out_ptr, M, K, N)

    def set_skip_node(self, nid):
        self._lib.sengine_set_skip_node(self._handle, nid)

    def set_alias_node(self, nid, src_ptr, dst_ptr, n_elems):
        self._lib.sengine_set_alias_node(self._handle, nid,
            src_ptr, dst_ptr, n_elems)

    def set_naive_conv_node(self, nid, in_ptr, w_ptr, sc_ptr, bi_ptr, out_ptr,
                            N, H, W, Cin, Cout, KH, KW, stride, pad, OH, OW, groups=1):
        self._lib.sengine_set_naive_conv_node(self._handle, nid,
            in_ptr, w_ptr, sc_ptr, bi_ptr, out_ptr,
            N, H, W, Cin, Cout, KH, KW, stride, pad, OH, OW, groups)

    def set_layout_transpose_node(self, nid, in_ptr, out_ptr, N, H, W, C, direction):
        """Set a layout transpose node (NHWC↔NCHW).

        direction: 0=NHWC→NCHW, 1=NCHW→NHWC
        """
        self._lib.sengine_set_layout_transpose_node(self._handle, nid,
            in_ptr, out_ptr, N, H, W, C, direction)

    def set_fused_attn_node(self, nid, variant, gemm1_idx, gemm2_idx,
                             q_ptr, k_ptr, v_ptr, out_ptr,
                             workspace_ptr, mem_ptr,
                             TB, heads, hd, N, H, W,
                             lif_total, lif_spatial,
                             v_thresh, recip_tau,
                             needs_permute,
                             scale1_ptr=0, scale2_ptr=0,
                             ws_gemm1_out=0, ws_perm_q=0,
                             ws_perm_k=0, ws_perm_v=0):
        """Set a fused attention node (TileLang batched GEMM + LIF).

        variant: 0=spikformer, 1=maxformer, 2=dssa, 3=token_qk
        gemm1_idx, gemm2_idx: TileLang .so indices for the two GEMM kernels
        """
        import ctypes
        self._lib.sengine_set_fused_attn_node(
            self._handle, nid, variant, gemm1_idx, gemm2_idx,
            q_ptr, k_ptr, v_ptr, out_ptr,
            workspace_ptr, mem_ptr,
            TB, heads, hd, N, H, W,
            lif_total, lif_spatial,
            ctypes.c_float(v_thresh), ctypes.c_float(recip_tau),
            needs_permute,
            scale1_ptr, scale2_ptr,
            ws_gemm1_out, ws_perm_q, ws_perm_k, ws_perm_v)

    # ─── Detection model ops ───

    def set_resize_node(self, nid, in_ptr, out_ptr, N, H, W, C, OH, OW, scale_h, scale_w):
        self._lib.sengine_set_resize_node(self._handle, nid,
            ctypes.c_void_p(in_ptr), ctypes.c_void_p(out_ptr),
            N, H, W, C, OH, OW, scale_h, scale_w)

    def set_concat_node(self, nid, a_ptr, b_ptr, out_ptr, NHW, Ca, Cb):
        self._lib.sengine_set_concat_node(self._handle, nid,
            ctypes.c_void_p(a_ptr), ctypes.c_void_p(b_ptr),
            ctypes.c_void_p(out_ptr), NHW, Ca, Cb)

    def set_ilif_node(self, nid, in_ptr, out_ptr, mem_ptr, total, spatial, decay, max_level):
        self._lib.sengine_set_ilif_node(self._handle, nid,
            ctypes.c_void_p(in_ptr), ctypes.c_void_p(out_ptr),
            ctypes.c_void_p(mem_ptr),
            total, spatial, ctypes.c_float(decay), ctypes.c_float(max_level))

    def set_softmax_node(self, nid, in_ptr, out_ptr, outer, inner):
        self._lib.sengine_set_softmax_node(self._handle, nid,
            ctypes.c_void_p(in_ptr), ctypes.c_void_p(out_ptr), outer, inner)

    # ─── Execution ───

    def capture_graph(self):
        self._lib.sengine_capture_graph(self._handle)

    def replay(self):
        self._lib.sengine_replay(self._handle)

    def execute_sequential_debug(self):
        """Run nodes one-by-one with sync after each, stop on first CUDA error."""
        self._lib.sengine_execute_sequential_debug(self._handle)

    def sync(self):
        self._lib.sengine_sync(self._handle)

    def reset_membranes(self):
        self._lib.sengine_reset_membranes(self._handle)

    def benchmark(self, warmup: int = 200, iters: int = 1000) -> float:
        """Return mean inference latency in milliseconds."""
        return self._lib.sengine_benchmark(self._handle, warmup, iters)
