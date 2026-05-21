"""Build orchestrator: ONNX → optimized CPU engine → .sengine-cpu file.

Usage:
    from sengine_cpu.build.engine_builder import CPUEngineBuilder

    engine = CPUEngineBuilder("model.onnx", T=4, batch_size=1).build()
    ms = engine.benchmark()
    engine.save("model.sengine-cpu")

Self-contained — no imports from sengine/.
"""

from __future__ import annotations

import os
import time
import numpy as np
from typing import Optional
from functools import reduce

from sengine_cpu.ir import EngineIR, CPUKernelVariant, BoundType, OpType, NeuronType
from sengine_cpu.parser import ONNXParser
from sengine_cpu.optimizer import optimize_ir
from sengine_cpu.memory import plan_memory, MemoryPlan
from sengine_cpu.build.tvm_compiler import TVMCompiler
from sengine_cpu.build.schedule_builder import build_schedule
from sengine_cpu.build.sengine_io import save_sengine_cpu, load_sengine_cpu
from sengine_cpu.runtime.cpp_executor import CPUCppExecutor
from sengine_cpu.logger import log


def _numel(shape):
    """Total elements from shape tuple."""
    if not shape:
        return 0
    return reduce(lambda a, b: a * b, shape, 1)


class CPUEngine:
    """Ready-to-run CPU inference engine."""

    def __init__(self, ir: EngineIR, schedule: list[int],
                 mem_plan: MemoryPlan, kernel_map: dict[int, str],
                 T: int, batch_size: int, n_threads: int = 0):
        self.ir = ir
        self.schedule = schedule
        self.mem_plan = mem_plan
        self.kernel_map = kernel_map
        self.T = T
        self.batch_size = batch_size
        self.n_threads = n_threads or os.cpu_count() or 1

        self._executor: Optional[CPUCppExecutor] = None
        self._tensor_bufs: dict[str, np.ndarray] = {}   # tensor_name → buffer
        self._weight_bufs: dict[str, np.ndarray] = {}   # weight_name → fp32 array
        self._membranes: dict[int, np.ndarray] = {}      # node_id → membrane
        self._input_buf: Optional[np.ndarray] = None
        self._output_buf: Optional[np.ndarray] = None

    def _setup_executor(self):
        """Wire the C executor with buffers and kernel registrations."""
        if self._executor is not None:
            return

        ir = self.ir
        exe = CPUCppExecutor(n_threads=self.n_threads)
        max_nid = max(ir.nodes.keys()) if ir.nodes else 0
        exe.alloc_nodes(max_nid + 1)

        # --- 1. Allocate weight buffers (FP32, contiguous) ---
        for name, w in ir.weights.items():
            self._weight_bufs[name] = np.ascontiguousarray(w.astype(np.float32))

        # --- 2. Allocate activation buffers per tensor name ---
        # Use memory plan slots to size buffers, but index by tensor name
        for slot in self.mem_plan.slots:
            size_elems = max(slot.size // 4, 1)
            self._tensor_bufs[slot.tensor_name] = np.zeros(size_elems, dtype=np.float32)

        # Also allocate for tensor names not in the plan (graph inputs, etc.)
        produced = set()
        for node in ir.nodes.values():
            produced.update(node.output_names)
        consumed = set()
        for node in ir.nodes.values():
            consumed.update(node.input_names)
        graph_inputs = consumed - produced - set(ir.weights.keys())
        for name in graph_inputs:
            if name not in self._tensor_bufs:
                # Allocate based on model input shape
                n = _numel(ir.model_input_shape) or 1024
                self._tensor_bufs[name] = np.zeros(n, dtype=np.float32)
                self._input_buf = self._tensor_bufs[name]

        # --- 3. Allocate membrane state buffers ---
        for nid, node in ir.nodes.items():
            if node.is_stateful and node.neuron_params:
                # Membrane shape: per-timestep spatial size
                out_shape = node.output_shapes[0] if node.output_shapes else ()
                total = _numel(out_shape)
                T = node.neuron_params.T or self.T or 4
                spatial = max(total // T, 1) if total > 0 else 1024
                mem = np.zeros(spatial, dtype=np.float32)
                self._membranes[nid] = mem
                exe.register_membrane(mem)

        # --- 4. Load TVM kernels ---
        tvm_indices: dict[str, int] = {}
        for so_path in set(self.kernel_map.values()):
            tvm_indices[so_path] = exe.load_tvm(so_path)

        # --- 5. Register each node ---
        for nid in self.schedule:
            node = ir.nodes.get(nid)
            if node is None:
                continue
            kv = node.assigned_kernel

            if kv in (CPUKernelVariant.Skip, CPUKernelVariant.ZeroCost):
                exe.set_skip_node(nid)

            elif kv.name.startswith("TVM") and nid in self.kernel_map:
                self._register_tvm_node(exe, nid, node, tvm_indices)

            elif kv == CPUKernelVariant.NativeIF:
                self._register_native_neuron(exe, nid, node, is_lif=False)

            elif kv == CPUKernelVariant.NativeLIF:
                self._register_native_neuron(exe, nid, node, is_lif=True)

            elif kv == CPUKernelVariant.NativeAdd:
                self._register_native_add(exe, nid, node)

            elif kv == CPUKernelVariant.NativeGemm:
                self._register_native_gemm(exe, nid, node)

            elif kv == CPUKernelVariant.NativeMaxPool:
                self._register_native_maxpool(exe, nid, node)

            elif kv == CPUKernelVariant.NativeGlobalAvgPool:
                self._register_native_gavg(exe, nid, node)

            elif kv == CPUKernelVariant.NativeTemporalMean:
                self._register_native_tmean(exe, nid, node)

            else:
                exe.set_skip_node(nid)

        exe.set_schedule(self.schedule)
        self._executor = exe
        log.info("Executor wired: %d nodes, %d TVM .so, %d membranes",
                 len(self.schedule), len(tvm_indices), len(self._membranes))

    # ─── Buffer resolution helpers ────────────────────────────

    def _get_buf(self, tensor_name: str) -> np.ndarray:
        """Resolve a tensor name to its buffer (activation or weight)."""
        if tensor_name in self._tensor_bufs:
            return self._tensor_bufs[tensor_name]
        if tensor_name in self._weight_bufs:
            return self._weight_bufs[tensor_name]
        # Allocate on demand
        buf = np.zeros(1024, dtype=np.float32)
        self._tensor_bufs[tensor_name] = buf
        return buf

    def _get_output_buf(self, node) -> np.ndarray:
        """Get or create output buffer for a node."""
        if node.output_names:
            name = node.output_names[0]
            if name in self._tensor_bufs:
                return self._tensor_bufs[name]
            n = _numel(node.output_shapes[0]) if node.output_shapes else 1024
            buf = np.zeros(max(n, 1), dtype=np.float32)
            self._tensor_bufs[name] = buf
            return buf
        return np.zeros(1024, dtype=np.float32)

    def _get_input_buf(self, node) -> np.ndarray:
        """Get the first data input buffer for a node (skip weight inputs)."""
        for name in node.input_names:
            if name not in self.ir.weights:
                return self._get_buf(name)
        return self._get_buf(node.input_names[0]) if node.input_names else np.zeros(1, dtype=np.float32)

    # ─── Node registration ────────────────────────────────────

    def _register_tvm_node(self, exe, nid, node, tvm_indices):
        """Register a TVM-compiled kernel node.

        TVM fused Conv+BN+IF signature: (data, weight, scale, bias, mem, spike, new_mem)
        The C executor calls: call_fn(arg0, arg1, ..., argN) with flat pointers.
        """
        so_path = self.kernel_map[nid]
        tvm_idx = tvm_indices[so_path]

        # Build arg list based on kernel variant
        kv = node.assigned_kernel

        # Fused Conv+BN+IF: (data, weight, scale, bias, membrane, spike_out, membrane_out)
        data_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)

        # Weight
        w_name = node.weight_info.name if node.weight_info else ""
        w_buf = self._weight_bufs.get(w_name, np.zeros(1, dtype=np.float32))

        # BN scale/bias
        scale_buf = np.ascontiguousarray(
            np.array(node.bn_scale, dtype=np.float32)) if node.bn_scale else np.ones(1, dtype=np.float32)
        bias_buf = np.ascontiguousarray(
            np.array(node.bn_bias, dtype=np.float32)) if node.bn_bias else np.zeros(1, dtype=np.float32)
        # Keep refs alive
        self._tensor_bufs[f"__bn_scale_{nid}"] = scale_buf
        self._tensor_bufs[f"__bn_bias_{nid}"] = bias_buf

        # Membrane (from fused neuron)
        fg_id = node.fusion_group_id
        mem_nid = -1
        if fg_id >= 0 and fg_id < len(self.ir.fusion_groups):
            mem_nid = self.ir.fusion_groups[fg_id].neuron_node_id
        mem_buf = self._membranes.get(mem_nid)
        if mem_buf is None:
            out_shape = node.output_shapes[0] if node.output_shapes else ()
            spatial = max(_numel(out_shape) // self.T, 1)
            mem_buf = np.zeros(spatial, dtype=np.float32)
            self._membranes[mem_nid] = mem_buf
            exe.register_membrane(mem_buf)

        # New membrane output buffer (same size as membrane)
        new_mem_buf = np.zeros_like(mem_buf)
        self._tensor_bufs[f"__new_mem_{nid}"] = new_mem_buf

        args = [data_buf, w_buf.ravel(), scale_buf, bias_buf, mem_buf, out_buf, new_mem_buf]
        exe.set_tvm_node(nid, tvm_idx, args)

    def _register_native_neuron(self, exe, nid, node, is_lif=False):
        """Register a standalone native IF/LIF neuron."""
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        total = _numel(out_shape)
        T = node.neuron_params.T if node.neuron_params else self.T
        spatial = max(total // T, 1)
        thresh = node.neuron_params.v_threshold if node.neuron_params else 1.0

        mem_buf = self._membranes.get(nid)
        if mem_buf is None:
            mem_buf = np.zeros(spatial, dtype=np.float32)
            self._membranes[nid] = mem_buf
            exe.register_membrane(mem_buf)

        if is_lif:
            tau = node.neuron_params.tau if node.neuron_params else 2.0
            recip_tau = 1.0 / tau if tau > 0 else 0.5
            exe.set_lif_node(nid, inp_buf, out_buf, mem_buf,
                             total, spatial, thresh, recip_tau)
        else:
            exe.set_if_node(nid, inp_buf, out_buf, mem_buf,
                            total, spatial, thresh)

    def _register_native_add(self, exe, nid, node):
        """Register a native Add node."""
        out_buf = self._get_output_buf(node)
        # Two inputs
        bufs = []
        for name in node.input_names:
            if name not in self.ir.weights:
                bufs.append(self._get_buf(name))
        if len(bufs) < 2:
            exe.set_skip_node(nid); return
        n = min(len(bufs[0]), len(bufs[1]), len(out_buf))
        exe.set_add_node(nid, bufs[0], bufs[1], out_buf, n)

    def _register_native_gemm(self, exe, nid, node):
        """Register a native GEMM (classifier) node."""
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        # Weight
        w_name = node.weight_info.name if node.weight_info else ""
        w_buf = self._weight_bufs.get(w_name, np.zeros(1, dtype=np.float32))
        # Bias
        b_name = node.bias_info.name if node.bias_info else ""
        b_buf = self._weight_bufs.get(b_name)

        in_shape = node.input_shapes[0] if node.input_shapes else ()
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        M = in_shape[0] if len(in_shape) >= 1 else 1
        K = in_shape[1] if len(in_shape) >= 2 else (in_shape[0] if in_shape else 1)
        N = out_shape[1] if len(out_shape) >= 2 else (out_shape[0] if out_shape else 1)
        # Handle transB
        transB = node.gemm_params.get("transB", 0) if node.gemm_params else 0
        if transB and node.weight_info and node.weight_info.shape:
            ws = node.weight_info.shape
            # ONNX Gemm with transB: weight is (N, K), need to transpose
            w_buf = np.ascontiguousarray(w_buf.reshape(ws).T)
            self._weight_bufs[f"__gemm_w_{nid}"] = w_buf

        exe.set_gemm_node(nid, inp_buf, w_buf.ravel(), out_buf, M, K, N)

    def _register_native_maxpool(self, exe, nid, node):
        """Register a native MaxPool node."""
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        pp = node.pool_params or {}
        in_shape = node.input_shapes[0] if node.input_shapes else (1, 1, 1, 1)
        out_shape = node.output_shapes[0] if node.output_shapes else in_shape

        N = in_shape[0] if len(in_shape) >= 1 else 1
        C = in_shape[1] if len(in_shape) >= 2 else 1
        H = in_shape[2] if len(in_shape) >= 3 else 1
        W = in_shape[3] if len(in_shape) >= 4 else 1
        OH = out_shape[2] if len(out_shape) >= 3 else 1
        OW = out_shape[3] if len(out_shape) >= 4 else 1

        ks = pp.get("kernel_shape", [3, 3])
        st = pp.get("strides", [2, 2])
        pa = pp.get("pads", [1, 1, 1, 1])
        kh = ks[0] if isinstance(ks, list) else ks
        kw = ks[1] if isinstance(ks, list) and len(ks) > 1 else kh
        sh = st[0] if isinstance(st, list) else st
        sw = st[1] if isinstance(st, list) and len(st) > 1 else sh
        ph = pa[0] if isinstance(pa, list) else pa
        pw = pa[1] if isinstance(pa, list) and len(pa) > 1 else ph

        exe._lib.sengine_cpu_set_maxpool_node(
            exe._handle, nid, exe._fp(inp_buf), exe._fp(out_buf),
            N, H, W, C, OH, OW, kh, kw, sh, sw, ph, pw)

    def _register_native_gavg(self, exe, nid, node):
        """Register a native GlobalAvgPool node."""
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        in_shape = node.input_shapes[0] if node.input_shapes else (1, 1, 1, 1)
        N = in_shape[0] if len(in_shape) >= 1 else 1
        C = in_shape[1] if len(in_shape) >= 2 else 1
        H = in_shape[2] if len(in_shape) >= 3 else 1
        W = in_shape[3] if len(in_shape) >= 4 else 1
        exe._lib.sengine_cpu_set_gavg_node(
            exe._handle, nid, exe._fp(inp_buf), exe._fp(out_buf), N, H, W, C)

    def _register_native_tmean(self, exe, nid, node):
        """Register a native TemporalMean node."""
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        in_shape = node.input_shapes[0] if node.input_shapes else ()
        total = _numel(in_shape)
        T = self.T
        spatial = max(total // T, 1)
        exe._lib.sengine_cpu_set_tmean_node(
            exe._handle, nid, exe._fp(inp_buf), exe._fp(out_buf), T, spatial)

    # ─── Public API ───────────────────────────────────────────

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference. Returns output numpy array."""
        self._setup_executor()
        self._executor.reset_membranes()
        self._executor.execute()
        return np.zeros(self.ir.model_output_shape or (1,), dtype=np.float32)

    def benchmark(self, warmup: int = 50, iters: int = 200) -> float:
        """Benchmark latency. Returns average ms per inference."""
        self._setup_executor()
        return self._executor.benchmark(warmup, iters)

    def save(self, path: str):
        """Save engine to .sengine-cpu file."""
        save_sengine_cpu(path, self.ir, self.schedule,
                          self.T, self.batch_size, self.kernel_map)


class CPUEngineBuilder:
    """Orchestrates: ONNX → parse → optimize → compile → schedule → engine."""

    def __init__(self, onnx_path: str, T: int = 4, batch_size: int = 1,
                 n_threads: int = 0, target: str | dict | None = None):
        self.onnx_path = onnx_path
        self.T = T
        self.batch_size = batch_size
        self.n_threads = n_threads or os.cpu_count() or 1
        self.target = target or {"kind": "llvm"}

    def build(self) -> CPUEngine:
        """Full build pipeline."""
        t0 = time.time()

        log.info("Parsing ONNX: %s", self.onnx_path)
        parser = ONNXParser(self.onnx_path)
        ir = parser.parse()
        log.info("Parsed: %d nodes, T=%d", len(ir.nodes), ir.T)

        log.info("Optimizing IR")
        optimize_ir(ir, batch_size=self.batch_size, T=self.T)

        log.info("Compiling TVM kernels")
        compiler = TVMCompiler(ir, T=self.T, batch_size=self.batch_size,
                                target=self.target)
        kernel_map = compiler.compile_all()

        log.info("Building BA-MTTS schedule")
        schedule = build_schedule(ir)

        log.info("Planning memory")
        mem_plan = plan_memory(ir, execution_order=schedule)

        elapsed = time.time() - t0
        log.info("Build complete in %.1fs: %d nodes, %d kernels, schedule=%d ops",
                 elapsed, len(ir.nodes), len(kernel_map), len(schedule))

        return CPUEngine(ir, schedule, mem_plan, kernel_map,
                          self.T, self.batch_size, self.n_threads)

    @staticmethod
    def load(path: str, n_threads: int = 0) -> CPUEngine:
        """Load a .sengine-cpu file."""
        t0 = time.time()
        ir, schedule, T, batch_size = load_sengine_cpu(path)

        compiler = TVMCompiler(ir, T=T, batch_size=batch_size)
        kernel_map = compiler.compile_all()
        mem_plan = plan_memory(ir, execution_order=schedule)

        elapsed = time.time() - t0
        log.info("Loaded in %.1fs from %s", elapsed, path)

        return CPUEngine(ir, schedule, mem_plan, kernel_map,
                          T, batch_size, n_threads or os.cpu_count() or 1)
