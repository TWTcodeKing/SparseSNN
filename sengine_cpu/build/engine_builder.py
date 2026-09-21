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

        # --- 2b. Alias zero-cost / absorbed nodes onto their data input ---
        # Reshape/Flatten/Identity/absorbed-neuron outputs are the same bytes as
        # their input; downstream nodes resolve buffers by tensor name, so point
        # the output names at the producer's buffer. The Tile node is the
        # engine input (T-replicated NHWC) and keeps its own buffer.
        self._input_name = None
        for nid in ir.topo_order:
            node = ir.nodes.get(nid)
            if node is None or not node.output_names:
                continue
            if node.op_type == OpType.Tile:
                # Not in the BA-MTTS schedule (zero-cost), so the memory plan
                # has no slot for it: allocate the T-replicated input here.
                self._input_name = node.output_names[0]
                if self._input_name not in self._tensor_bufs:
                    n = _numel(node.output_shapes[0]) if node.output_shapes else 0
                    if n <= 0:
                        n = _numel(ir.model_input_shape) * self.T
                    self._tensor_bufs[self._input_name] = np.zeros(max(n, 1), dtype=np.float32)
                continue
            kv = node.assigned_kernel
            if not (kv in (CPUKernelVariant.ZeroCost, CPUKernelVariant.Skip)
                    or node.op_type == OpType.Identity):
                continue
            src = next((n for n in node.input_names
                        if n not in ir.weights and n in self._tensor_bufs), None)
            if src is None:
                continue
            for out in node.output_names:
                self._tensor_bufs[out] = self._tensor_bufs[src]
        if self._input_name is None:
            for name in graph_inputs:
                self._input_name = name
                break
        self._input_buf = self._tensor_bufs.get(self._input_name)
        # Graph output: a produced tensor nobody consumes (prefer the last one in topo order)
        self._output_name = None
        for nid in reversed(ir.topo_order):
            node = ir.nodes.get(nid)
            if node is None:
                continue
            for out in node.output_names:
                if out not in consumed and out in self._tensor_bufs:
                    self._output_name = out
                    break
            if self._output_name is not None:
                break

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

            elif kv in (CPUKernelVariant.NativeConvBNIF, CPUKernelVariant.NativeConvBNLIF,
                        CPUKernelVariant.NativeConvBN):
                self._register_native_conv(exe, nid, node)

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

    def _register_native_conv(self, exe, nid, node):
        """Register the native NHWC Conv+BN(+neuron) kernel for a Conv2d node."""
        cp = node.conv_params
        inp_buf = self._get_input_buf(node)
        out_buf = self._get_output_buf(node)
        in_shape = node.input_shapes[0] if node.input_shapes else ()
        if len(in_shape) != 4:
            log.warning("Native conv #%d: missing 4D input shape, skipping", nid)
            exe.set_skip_node(nid); return
        TB, C_in, H, W = in_shape
        T = self.T
        B = max(TB // T, 1)
        F = cp.out_channels
        if cp.groups != 1 or cp.kernel_h != cp.kernel_w or cp.stride_h != cp.stride_w \
                or cp.pad_h != cp.pad_w or cp.dilation_h != 1:
            raise NotImplementedError(f"native conv #{nid}: unsupported conv params {cp}")
        # Weight (F, C_in, KH, KW) -> (KH, KW, C_in, F) == (K, F) row-major
        w_name = node.weight_info.name if node.weight_info else ""
        w = np.asarray(self.ir.weights[w_name], dtype=np.float32).reshape(F, C_in, cp.kernel_h, cp.kernel_w)
        w_kf = np.ascontiguousarray(w.transpose(2, 3, 1, 0)).reshape(-1)
        self._weight_bufs[f"__conv_w_{nid}"] = w_kf
        scale = np.ascontiguousarray(np.array(node.bn_scale, dtype=np.float32)) if node.bn_scale \
            else np.ones(F, dtype=np.float32)
        bias = np.ascontiguousarray(np.array(node.bn_bias, dtype=np.float32)) if node.bn_bias \
            else np.zeros(F, dtype=np.float32)
        if node.bias_info is not None and node.bias_info.name in self.ir.weights:
            # conv bias without BN: y = scale*(conv + b) + bias
            b = np.asarray(self.ir.weights[node.bias_info.name], dtype=np.float32).reshape(-1)
            bias = bias + scale * b
        self._weight_bufs[f"__conv_scale_{nid}"] = scale
        self._weight_bufs[f"__conv_bias_{nid}"] = bias
        # Neuron (fused) + membrane (M, F)
        neuron, v_th, v_reset, recip_tau = 0, 1.0, 0.0, 1.0
        mem_buf = np.zeros(1, dtype=np.float32)
        fg_id = node.fusion_group_id
        if fg_id >= 0 and fg_id < len(self.ir.fusion_groups):
            nn = self.ir.nodes.get(self.ir.fusion_groups[fg_id].neuron_node_id)
            npar = nn.neuron_params if nn is not None else None
            if npar is not None:
                neuron = 2 if npar.neuron_type == NeuronType.LIF else 1
                v_th = float(npar.v_threshold)
                v_reset = float(npar.v_reset if npar.hard_reset else 0.0)
                recip_tau = (1.0 / float(npar.tau)) if (neuron == 2 and npar.tau > 0) else 1.0
            OH = (H + 2 * cp.pad_h - cp.kernel_h) // cp.stride_h + 1
            OW = (W + 2 * cp.pad_w - cp.kernel_w) // cp.stride_w + 1
            need = B * OH * OW * F
            mem_buf = self._membranes.get(nn.id) if nn is not None else None
            if mem_buf is None or mem_buf.size < need:
                mem_buf = np.zeros(need, dtype=np.float32)
                if nn is not None:
                    self._membranes[nn.id] = mem_buf
                exe.register_membrane(mem_buf)
        self._weight_bufs[f"__conv_mem_{nid}"] = mem_buf
        exe.set_conv_node(nid, inp_buf, w_kf, scale, bias, mem_buf, out_buf,
                          B, H, W, C_in, F, T, cp.kernel_h, cp.kernel_w, cp.pad_h, cp.stride_h,
                          neuron, v_th, v_reset, recip_tau)

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
        # The C kernel computes C = A @ B^T with B stored (N, K) row-major,
        # which is exactly ONNX Gemm(transB=1). For transB=0 the weight is
        # (K, N) and must be transposed once at build time.
        transB = node.gemm_params.get("transB", 0) if node.gemm_params else 0
        if not transB and node.weight_info and node.weight_info.shape:
            ws = node.weight_info.shape
            w_buf = np.ascontiguousarray(w_buf.reshape(ws).T)
            self._weight_bufs[f"__gemm_w_{nid}"] = w_buf

        exe.set_gemm_node(nid, inp_buf, w_buf.ravel(), out_buf, M, K, N)
        if b_buf is not None:
            b32 = np.ascontiguousarray(np.asarray(b_buf, dtype=np.float32).reshape(-1))
            self._weight_bufs[f"__gemm_b_{nid}"] = b32
            exe.set_gemm_bias(nid, b32)

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
        """Run inference on (B, C, H, W) NCHW float32 input. Returns (B, ...) output.

        The input is converted to NHWC and replicated T times (direct
        encoding) into the Tile node's buffer, exactly like the GPU engine.
        """
        self._setup_executor()
        x = np.asarray(input_data, dtype=np.float32)
        if x.ndim == 4:
            x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        x = np.ascontiguousarray(x).reshape(-1)
        buf = self._input_buf
        if buf is None:
            raise RuntimeError("engine input buffer not found")
        if buf.size >= x.size * self.T:
            buf[:x.size * self.T] = np.tile(x, self.T)
        else:
            buf[:min(buf.size, x.size)] = x[:buf.size]
        self._executor.reset_membranes()
        self._executor.execute()
        out = self._tensor_bufs.get(self._output_name)
        if out is None:
            return np.zeros(self.ir.model_output_shape or (1,), dtype=np.float32)
        shape = [int(d) for d in (self.ir.model_output_shape or ())]
        if shape and shape[0] == 0:
            shape[0] = self.batch_size
        n = int(np.prod(shape)) if shape else out.size
        if n <= 0 or n > out.size:
            return out.copy()
        return out[:n].reshape(shape).copy()

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

        if any(n.assigned_kernel.name.startswith("TVM") for n in ir.nodes.values()):
            log.info("Compiling TVM kernels")
            compiler = TVMCompiler(ir, T=self.T, batch_size=self.batch_size,
                                    target=self.target)
            kernel_map = compiler.compile_all()
        else:
            log.info("No TVM kernels needed (native conv backend)")
            kernel_map = {}

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

        if any(n.assigned_kernel.name.startswith("TVM") for n in ir.nodes.values()):
            compiler = TVMCompiler(ir, T=T, batch_size=batch_size)
            kernel_map = compiler.compile_all()
        else:
            kernel_map = {}
        mem_plan = plan_memory(ir, execution_order=schedule)

        elapsed = time.time() - t0
        log.info("Loaded in %.1fs from %s", elapsed, path)

        return CPUEngine(ir, schedule, mem_plan, kernel_map,
                          T, batch_size, n_threads or os.cpu_count() or 1)
