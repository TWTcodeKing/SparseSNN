"""SEngine: High-performance SNN inference engine.

TensorRT-like API for building, saving, loading, and running SNN models.

    # Build from ONNX
    engine = sengine.build("model.onnx", T=4, batch_size=1)
    engine.save("model.sengine")

    # Load and run
    engine = sengine.load("model.sengine")
    output = engine.infer(input_numpy)     # numpy in → numpy out
    ms = engine.benchmark()                # latency measurement
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from sengine.ir import OpType, KernelVariant, EngineIR
from sengine.build.engine_builder import EngineBuilder
from sengine.build.export_standalone import export_all_kernels
from sengine.runtime.cpp_executor import CppExecutor
from sengine.logger import logger


TILELANG_VARIANTS = frozenset({
    KernelVariant.TileLangConvBN,
    KernelVariant.TileLangConv1x1BN,
    KernelVariant.TileLangStemConvBN,
    KernelVariant.TileLangFusedConvBNIF,
    KernelVariant.TileLangFusedConv1x1BNIF,
    KernelVariant.TileLangLinearBN,
    KernelVariant.TileLangLinearBNLIF,
})


def _detect_arch() -> str:
    """Detect current GPU SM architecture."""
    if not torch.cuda.is_available():
        return 'sm_80'
    props = torch.cuda.get_device_properties(0)
    return f'sm_{props.major}{props.minor}'


def _detect_nvcc() -> str:
    """Find nvcc binary."""
    for path in ['/usr/local/cuda-12.8/bin/nvcc', '/usr/local/cuda/bin/nvcc']:
        if os.path.exists(path):
            return path
    return 'nvcc'


class SEngine:
    """SNN inference engine with C++ CUDA Graph execution.

    Wraps the full pipeline: ONNX → IR → TileLang → schedule → C++ executor.
    All kernel dispatch happens in C++ — zero Python in the inference path.
    """

    def __init__(self):
        self._py_engine = None      # CUDAGraphEngine (Python, for buffer management)
        self._cpp_exec: Optional[CppExecutor] = None
        self._ir: Optional[EngineIR] = None
        self._schedule: list[int] = []
        self._kernels: dict = {}
        self._kernel_so_map: dict[int, str] = {}
        self.T: int = 4
        self.batch_size: int = 1
        self._graph_captured: bool = False

    # ─── Build API ───

    @staticmethod
    def build(onnx_path: str, T: int = 4, batch_size: int = 1,
              autotune: bool = False, build_dir: str | None = None) -> SEngine:
        """Build an engine from an ONNX file.

        Args:
            onnx_path: Path to plugin-mode ONNX file.
            T: Number of temporal steps.
            batch_size: Inference batch size.
            autotune: Run autotuning sweep for tile configs.
            build_dir: Directory for compiled .so files (default: /tmp/sengine_B{batch}).

        Returns:
            Ready-to-run SEngine instance.
        """
        eng = SEngine()
        eng.T = T
        eng.batch_size = batch_size

        t0 = time.time()
        arch = _detect_arch()
        nvcc = _detect_nvcc()
        logger.phase("BUILD", "Target: %s, T=%d, B=%d", arch, T, batch_size)

        # 1. Build Python engine (parse → optimize → compile → schedule)
        builder = EngineBuilder(onnx_path, T=T, batch_size=batch_size)
        eng._py_engine = builder.build(autotune=autotune, capture_graph=False)
        eng._ir = builder._ir
        eng._schedule = builder._schedule
        eng._kernels = eng._py_engine.kernels

        # 2. Export TileLang kernels as standalone .so
        if build_dir is None:
            build_dir = f'/tmp/sengine_B{batch_size}'
        eng._kernel_so_map = export_all_kernels(
            eng._kernels, eng._ir, build_dir, nvcc=nvcc, arch=arch)

        # 3. Wire up C++ executor
        eng._setup_cpp_executor()

        # 4. Capture CUDA Graph
        eng._cpp_exec.capture_graph()
        eng._cpp_exec.sync()
        # Drain any deferred CUDA errors from graph capture
        torch.cuda.synchronize()
        eng._graph_captured = True

        elapsed = time.time() - t0
        logger.phase("BUILD", "Ready in %.1fs (%d ops, C++ CUDA Graph captured)", elapsed, len(eng._schedule))
        return eng

    # ─── Save / Load ───

    def save(self, path: str):
        """Save engine to .sengine file.

        The .sengine file stores IR, schedule, tile configs, and weights.
        TileLang kernels are recompiled on load() for the target GPU.
        """
        from sengine.build.sengine_io import save_sengine
        if self._ir is None or not self._schedule:
            raise RuntimeError("Nothing to save — call build() first")
        save_sengine(path, self._ir, self._schedule, self.T, self.batch_size)
        logger.phase("SAVE", "Saved to %s (%.1f MB)", path, os.path.getsize(path) / 1e6)

    @staticmethod
    def load(path: str, build_dir: str | None = None) -> SEngine:
        """Load engine from .sengine file.

        Recompiles TileLang kernels for the current GPU (fast, uses cached tile configs).

        Args:
            path: Path to .sengine file.
            build_dir: Directory for compiled .so files.

        Returns:
            Ready-to-run SEngine instance.
        """
        from sengine.build.sengine_io import load_sengine
        from sengine.build.tilelang_compiler import TileLangCompiler
        from sengine.cuda_graph_runtime import CUDAGraphEngine

        eng = SEngine()
        t0 = time.time()
        arch = _detect_arch()
        nvcc = _detect_nvcc()

        # 1. Load IR + schedule from file
        ir, schedule, T, batch_size = load_sengine(path)
        eng._ir = ir
        eng._schedule = schedule
        eng.T = T
        eng.batch_size = batch_size

        # 2. Recompile TileLang kernels (uses cached tile configs → fast)
        compiler = TileLangCompiler(ir, T=T, batch_size=batch_size, autotune=False)
        eng._kernels = compiler.compile_all()

        # 3. Build Python engine (for buffer allocation)
        eng._py_engine = CUDAGraphEngine()
        eng._py_engine.build(ir, eng._kernels, schedule, T=T, batch_size=batch_size)

        # 4. Export standalone .so + C++ executor
        if build_dir is None:
            build_dir = f'/tmp/sengine_B{batch_size}'
        eng._kernel_so_map = export_all_kernels(
            eng._kernels, ir, build_dir, nvcc=nvcc, arch=arch)
        eng._setup_cpp_executor()
        eng._cpp_exec.capture_graph()
        eng._graph_captured = True

        elapsed = time.time() - t0
        logger.phase("LOAD", "Loaded from %s in %.1fs (%s)", path, elapsed, arch)
        return eng

    # ─── Inference ───

    def infer(self, x: np.ndarray) -> np.ndarray:
        """Run inference on a numpy array.

        Args:
            x: Input image(s), shape (B, C, H, W), float32 or float16.

        Returns:
            Output numpy array (e.g., class logits).
        """
        import ctypes as ct

        graph_input = self._py_engine._graph_input
        if graph_input is None:
            return np.zeros(1)

        # Prepare NHWC FP16 input via raw CUDA memcpy (no PyTorch kernel launches)
        # Convert NCHW → NHWC on CPU, then copy to GPU
        if x.ndim == 4:
            # (B, C, H, W) → (B, H, W, C)
            x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
            if x_nhwc.shape[0] == self.batch_size:
                x_nhwc = np.tile(x_nhwc, (self.T, 1, 1, 1))
        else:
            x_nhwc = np.ascontiguousarray(x)

        x_fp16 = x_nhwc.astype(np.float16)

        # Copy to graph input buffer via cudaMemcpy
        cuda_rt = ct.CDLL('libcudart.so')
        src_ptr = x_fp16.ctypes.data_as(ct.c_void_p)
        dst_ptr = ct.c_void_p(graph_input.data_ptr())
        nbytes = x_fp16.nbytes
        cuda_rt.cudaMemcpy(dst_ptr, src_ptr, ct.c_size_t(nbytes), ct.c_int(1))  # H2D

        # Reset membranes + replay graph (all on C++ stream)
        self._cpp_exec.reset_membranes()
        self._cpp_exec.replay()
        self._cpp_exec.sync()

        # Read output via cudaMemcpy (D2H)
        out_nid = self._py_engine._graph_output_nid
        out_buf = self._py_engine.activations.get(out_nid)
        if out_buf is None:
            return np.zeros(1)

        out_np = np.empty(out_buf.shape, dtype=np.float16)
        cuda_rt.cudaMemcpy(
            out_np.ctypes.data_as(ct.c_void_p),
            ct.c_void_p(out_buf.data_ptr()),
            ct.c_size_t(out_np.nbytes),
            ct.c_int(2))  # D2H
        return out_np.astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for infer()."""
        return self.infer(x)

    # ─── Benchmark ───

    def benchmark(self, warmup: int = 200, iters: int = 1000) -> float:
        """Return mean inference latency in milliseconds."""
        if not self._graph_captured:
            raise RuntimeError("Engine not built — call build() or load() first")
        return self._cpp_exec.benchmark(warmup, iters)

    # ─── Cleanup ───

    def destroy(self):
        """Release GPU resources."""
        if self._cpp_exec:
            self._cpp_exec.destroy()
            self._cpp_exec = None
        self._py_engine = None

    def __del__(self):
        self.destroy()

    # ─── Internals ───

    def _setup_cpp_executor(self):
        """Wire the C++ executor to the Python engine's GPU buffers."""
        exe = CppExecutor()
        ir = self._ir
        engine = self._py_engine
        schedule = self._schedule
        T = self.T

        max_nid = max(ir.nodes.keys())
        exe.alloc_nodes(max_nid)
        exe.set_schedule(schedule)

        # Load TileLang .so files
        nid_tl_idx = {}
        for nid, so_path in self._kernel_so_map.items():
            tl_idx = exe.load_tilelang(so_path)
            nid_tl_idx[nid] = tl_idx

        # Register membranes
        for nid, mem in engine.membranes.items():
            exe.add_membrane(mem.data_ptr(), mem.numel())

        # Configure each node
        act = engine.activations
        for nid in schedule:
            node = ir.nodes.get(nid)
            if node is None:
                exe.set_skip_node(nid)
                continue

            kv = node.assigned_kernel
            preds = ir.predecessors(nid)
            input_nid = preds[0] if preds else nid
            input_buf = act.get(input_nid)
            output_buf = act.get(nid)
            p = lambda t: t.data_ptr()  # noqa: E731

            if kv in TILELANG_VARIANTS and nid in nid_tl_idx:
                tl_idx = nid_tl_idx[nid]
                w = engine.weights.get(nid)
                if w is None:
                    w = engine.weights_1x1.get(nid)
                sc = engine.bn_scales.get(nid)
                bi = engine.bn_biases.get(nid)

                if input_buf is None or w is None or output_buf is None:
                    exe.set_skip_node(nid)
                    continue

                if kv in (KernelVariant.TileLangFusedConvBNIF,
                          KernelVariant.TileLangFusedConv1x1BNIF):
                    mem = None
                    for s in ir.successors(nid):
                        if s in engine.membranes:
                            mem = engine.membranes[s]
                            break
                    if mem is not None and sc is not None and bi is not None:
                        exe.set_tilelang_6(nid, tl_idx,
                            p(input_buf), p(w), p(mem), p(sc), p(bi), p(output_buf))
                    else:
                        exe.set_skip_node(nid)
                else:
                    if sc is not None and bi is not None:
                        exe.set_tilelang_5(nid, tl_idx,
                            p(input_buf), p(w), p(sc), p(bi), p(output_buf))
                    else:
                        exe.set_skip_node(nid)

            elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                mem = engine.membranes.get(nid)
                if input_buf is None or output_buf is None or mem is None:
                    exe.set_skip_node(nid)
                    continue
                v_thresh = node.neuron_params.v_threshold if node.neuron_params else 1.0
                total = input_buf.numel()
                spatial = total // T
                if kv == KernelVariant.CUDAVec4LIF:
                    recip_tau = 1.0 / node.neuron_params.tau if (node.neuron_params and node.neuron_params.tau > 0) else 0.5
                    exe.set_lif_node(nid, p(input_buf), p(output_buf), p(mem),
                                     total, spatial, v_thresh, recip_tau)
                else:
                    exe.set_if_node(nid, p(input_buf), p(output_buf), p(mem),
                                    total, spatial, v_thresh)

            elif kv == KernelVariant.Elementwise:
                if node.op_type == OpType.Add and len(preds) >= 2:
                    a = act.get(preds[0])
                    b = act.get(preds[1])
                    if a is not None and b is not None and output_buf is not None:
                        exe.set_add_node(nid, p(a), p(b), p(output_buf),
                                        min(a.numel(), b.numel()))
                    else:
                        exe.set_skip_node(nid)
                elif input_buf is not None and output_buf is not None:
                    exe.set_alias_node(nid, p(input_buf), p(output_buf), input_buf.numel())
                else:
                    exe.set_skip_node(nid)

            elif kv == KernelVariant.CuDNNPool:
                if node.op_type == OpType.MaxPool and node.pool_params and input_buf is not None and output_buf is not None:
                    pp = node.pool_params
                    s = input_buf.shape
                    ks = pp.get('kernel_size', 3)
                    st = pp.get('stride', 2)
                    pad = pp.get('padding', 1)
                    OH = (s[1] + 2*pad - ks) // st + 1
                    OW = (s[2] + 2*pad - ks) // st + 1
                    exe.set_maxpool_node(nid, p(input_buf), p(output_buf),
                                        s[0], s[1], s[2], s[3], OH, OW, ks, st, pad)
                elif node.op_type == OpType.GlobalAvgPool and input_buf is not None and output_buf is not None:
                    s = input_buf.shape
                    exe.set_global_avgpool_node(nid, p(input_buf), p(output_buf),
                                               s[0], s[1], s[2], s[3])
                else:
                    exe.set_skip_node(nid)

            elif kv == KernelVariant.TemporalMean:
                if input_buf is not None and output_buf is not None:
                    exe.set_temporal_mean_node(nid, p(input_buf), p(output_buf),
                                              T, input_buf.numel() // T)
                else:
                    exe.set_skip_node(nid)

            elif kv == KernelVariant.CuBLASGemm:
                w = engine.weights.get(nid)
                if input_buf is not None and w is not None and output_buf is not None:
                    M = input_buf.shape[0]
                    K = input_buf.shape[-1]
                    N = w.shape[0]
                    exe.set_gemm_node(nid, p(input_buf), p(w), p(output_buf), M, K, N)
                else:
                    exe.set_skip_node(nid)

            elif kv == KernelVariant.ZeroCost:
                if input_buf is not None and output_buf is not None and input_buf.data_ptr() != output_buf.data_ptr():
                    exe.set_alias_node(nid, p(input_buf), p(output_buf),
                                      min(input_buf.numel(), output_buf.numel()))
                else:
                    exe.set_skip_node(nid)

            else:
                exe.set_skip_node(nid)

        self._cpp_exec = exe


# ─── Module-level convenience functions ───

def build(onnx_path: str, T: int = 4, batch_size: int = 1, **kwargs) -> SEngine:
    """Build an SEngine from an ONNX file.

    Args:
        onnx_path: Path to plugin-mode ONNX file.
        T: Number of temporal steps.
        batch_size: Inference batch size.

    Returns:
        Ready-to-run SEngine instance.

    Example:
        engine = sengine.build("model.onnx", T=4, batch_size=1)
        engine.save("model.sengine")
    """
    return SEngine.build(onnx_path, T=T, batch_size=batch_size, **kwargs)


def load(path: str, **kwargs) -> SEngine:
    """Load an SEngine from a .sengine file.

    Recompiles kernels for the current GPU automatically.

    Args:
        path: Path to .sengine file.

    Returns:
        Ready-to-run SEngine instance.

    Example:
        engine = sengine.load("model.sengine")
        output = engine.infer(input_array)
    """
    return SEngine.load(path, **kwargs)
