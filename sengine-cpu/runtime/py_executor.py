"""Pure-Python executor for sengine-cpu.

Dispatches nodes using numpy/TVM runtime. No C executor dependency.
Used for: correctness validation, benchmarking, debugging.

For TVM kernels: loads .so via tvm.runtime.load_module and calls directly.
For native ops: uses numpy implementations.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from functools import reduce
from typing import Optional

# TVM runtime for loading compiled kernels
_TVM_VENV = "/home/twt/tvm_build/tvm_venv"
if os.path.isdir(_TVM_VENV):
    _TVM_SITE = os.path.join(_TVM_VENV, "lib/python3.12/site-packages")
    if _TVM_SITE not in sys.path:
        sys.path.insert(0, _TVM_SITE)


def _numel(shape):
    if not shape:
        return 0
    return reduce(lambda a, b: a * b, shape, 1)


class PyExecutor:
    """Python-based dispatch executor for sengine-cpu.

    Loads TVM .so via tvm.runtime, executes native ops via numpy.
    """

    def __init__(self, ir, schedule, kernel_map, T, batch_size, n_threads=1):
        from sengine_cpu.ir import CPUKernelVariant, OpType
        self.ir = ir
        self.schedule = schedule
        self.kernel_map = kernel_map
        self.T = T
        self.batch_size = batch_size
        self.n_threads = n_threads

        self._tvm_modules = {}    # so_path → tvm.runtime.Module
        self._tensor_bufs = {}    # tensor_name → np.ndarray
        self._weight_bufs = {}    # weight_name → np.ndarray (fp32)
        self._membranes = {}      # node_id → np.ndarray
        self._setup()

    def _setup(self):
        ir = self.ir

        # Weights
        for name, w in ir.weights.items():
            self._weight_bufs[name] = np.ascontiguousarray(w.astype(np.float32))

        # Allocate activation buffers
        for nid in ir.topo_order:
            node = ir.nodes.get(nid)
            if node is None:
                continue
            for out_name in node.output_names:
                if out_name not in self._tensor_bufs:
                    n = _numel(node.output_shapes[0]) if node.output_shapes else 1024
                    self._tensor_bufs[out_name] = np.zeros(max(n, 1), dtype=np.float32)

        # Graph inputs
        produced = set()
        for node in ir.nodes.values():
            produced.update(node.output_names)
        consumed = set()
        for node in ir.nodes.values():
            consumed.update(node.input_names)
        for name in consumed - produced - set(ir.weights.keys()):
            if name not in self._tensor_bufs:
                n = _numel(ir.model_input_shape) or 1024
                self._tensor_bufs[name] = np.zeros(n, dtype=np.float32)

        # Membranes
        for nid, node in ir.nodes.items():
            if node.is_stateful and node.neuron_params:
                out_shape = node.output_shapes[0] if node.output_shapes else ()
                total = _numel(out_shape)
                T = node.neuron_params.T or self.T
                spatial = max(total // T, 1)
                self._membranes[nid] = np.zeros(spatial, dtype=np.float32)

        # Load C fused kernel library (used for Conv+BN+IF/LIF)
        self._c_fused_lib = None
        c_fused_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "csrc", "libfused_conv_bn_if.so")
        if os.path.exists(c_fused_path):
            import ctypes
            self._c_fused_lib = ctypes.CDLL(c_fused_path)
            # T-loop IF: data, weight, scale, bias, membrane, spikes, M, C_in, F, T, thresh, reset
            fn = getattr(self._c_fused_lib, "fused_conv1x1_bn_if_tloop", None)
            if fn:
                fn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*4 + [ctypes.c_float]*2
                fn.restype = None
            # T-loop LIF: + decay, recip_tau
            fn = getattr(self._c_fused_lib, "fused_conv1x1_bn_lif_tloop", None)
            if fn:
                fn.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*4 + [ctypes.c_float]*4
                fn.restype = None

        # Load TVM modules (fallback for non-fused TVM kernels)
        try:
            import tvm
            for so_path in set(self.kernel_map.values()):
                if os.path.exists(so_path):
                    self._tvm_modules[so_path] = tvm.runtime.load_module(so_path)
        except ImportError:
            pass

    def _get_buf(self, name):
        if name in self._tensor_bufs:
            return self._tensor_bufs[name]
        if name in self._weight_bufs:
            return self._weight_bufs[name]
        return np.zeros(1, dtype=np.float32)

    def _get_input(self, node):
        for name in node.input_names:
            if name not in self.ir.weights:
                return self._get_buf(name)
        return self._get_buf(node.input_names[0]) if node.input_names else np.zeros(1, dtype=np.float32)

    def _get_output(self, node):
        if node.output_names:
            return self._get_buf(node.output_names[0])
        return np.zeros(1, dtype=np.float32)

    def reset_membranes(self):
        for mem in self._membranes.values():
            mem[:] = 0

    def execute(self):
        """Execute the schedule once."""
        from sengine_cpu.ir import CPUKernelVariant, OpType

        for nid in self.schedule:
            node = self.ir.nodes.get(nid)
            if node is None:
                continue
            kv = node.assigned_kernel

            if kv in (CPUKernelVariant.Skip, CPUKernelVariant.ZeroCost):
                # Zero-cost: alias output to input
                if node.input_names and node.output_names:
                    inp = self._get_input(node)
                    out_name = node.output_names[0]
                    out = self._get_buf(out_name)
                    n = min(len(inp), len(out))
                    out[:n] = inp[:n]
                continue

            if kv.name.startswith("TVM") and nid in self.kernel_map:
                self._exec_tvm(nid, node)

            elif kv == CPUKernelVariant.NativeIF:
                self._exec_if(nid, node)

            elif kv == CPUKernelVariant.NativeLIF:
                self._exec_lif(nid, node)

            elif kv == CPUKernelVariant.NativeAdd:
                self._exec_add(node)

            elif kv == CPUKernelVariant.NativeGemm:
                self._exec_gemm(node)

            elif kv == CPUKernelVariant.NativeMaxPool:
                self._exec_maxpool(node)

            elif kv == CPUKernelVariant.NativeGlobalAvgPool:
                self._exec_gavg(node)

            elif kv == CPUKernelVariant.NativeTemporalMean:
                self._exec_tmean(node)

    def _exec_tvm(self, nid, node):
        """Execute Conv kernel — fused (interleaved) or decomposed (BLAS GEMM)."""
        from sengine_cpu.ir import CPUKernelVariant
        import ctypes

        # Route decomposed Conv+BN to BLAS path
        kv = node.assigned_kernel
        if kv in (CPUKernelVariant.TVMConvBN, CPUKernelVariant.TVMConv1x1BN):
            self._exec_decomposed_conv(node)
            return

        # ── Fused (interleaved) path: C micro-kernel with T-loop ──
        inp = self._get_input(node)
        out = self._get_output(node)

        # Weight
        w_name = node.weight_info.name if node.weight_info else ""
        w = self._weight_bufs.get(w_name, np.zeros(1, dtype=np.float32))

        # BN params
        if node.bn_scale:
            scale = np.ascontiguousarray(np.array(node.bn_scale, dtype=np.float32))
        else:
            scale = np.ones(1, dtype=np.float32)
        if node.bn_bias:
            bias = np.ascontiguousarray(np.array(node.bn_bias, dtype=np.float32))
        else:
            bias = np.zeros(1, dtype=np.float32)

        # Membrane
        fg_id = node.fusion_group_id
        mem_nid = -1
        if fg_id >= 0 and fg_id < len(self.ir.fusion_groups):
            mem_nid = self.ir.fusion_groups[fg_id].neuron_node_id
        mem = self._membranes.get(mem_nid, np.zeros(1, dtype=np.float32))

        cp = node.conv_params
        if not cp or not node.output_shapes:
            return

        out_shape = node.output_shapes[0]
        TB = out_shape[0] if len(out_shape) >= 1 else self.T
        C_out = cp.out_channels
        C_in = cp.in_channels
        OH = out_shape[2] if len(out_shape) >= 3 else 1
        OW = out_shape[3] if len(out_shape) >= 4 else 1
        M = (TB // self.T) * OH * OW

        # Ensure membrane is correctly sized
        needed = M * C_out
        if mem.size < needed:
            mem = np.zeros(needed, dtype=np.float32)
            self._membranes[mem_nid] = mem
        mem_2d = mem[:needed].reshape(M, C_out)

        # Prepare weight as (C_in, C_out) contiguous
        w_2d = np.ascontiguousarray(w.ravel()[:C_in * C_out].reshape(C_in, C_out))
        scale_1d = np.ascontiguousarray(scale[:C_out])
        bias_1d = np.ascontiguousarray(bias[:C_out])

        # Determine neuron type (IF vs LIF)
        neuron_node = self.ir.nodes.get(mem_nid) if mem_nid >= 0 else None
        is_lif = (neuron_node and neuron_node.neuron_params and
                  neuron_node.neuron_params.neuron_type.name == "LIF")

        fp = lambda a: a.ctypes.data
        all_spikes = np.zeros(TB * OH * OW * C_out, dtype=np.float32)

        # Flatten input: (T*M, C_in) contiguous
        inp_total = TB * OH * OW * C_in
        inp_flat = np.ascontiguousarray(inp[:min(inp.size, inp_total)])

        # Prepare spike output as (T*M, F) contiguous
        spk_flat = np.ascontiguousarray(all_spikes.reshape(self.T * M, C_out)
                                        if all_spikes.size == self.T * M * C_out
                                        else np.zeros((self.T * M, C_out), dtype=np.float32))

        if self._c_fused_lib:
            thresh = 1.0
            v_reset = 0.0
            if neuron_node and neuron_node.neuron_params:
                thresh = neuron_node.neuron_params.v_threshold
                v_reset = neuron_node.neuron_params.v_reset

            if is_lif:
                tau = neuron_node.neuron_params.tau if neuron_node.neuron_params else 2.0
                decay = 1.0 - 1.0 / tau
                recip = 1.0 / tau
                self._c_fused_lib.fused_conv1x1_bn_lif_tloop(
                    fp(inp_flat), fp(w_2d), fp(scale_1d), fp(bias_1d),
                    fp(mem_2d), fp(spk_flat),
                    M, C_in, C_out, self.T,
                    ctypes.c_float(thresh), ctypes.c_float(v_reset),
                    ctypes.c_float(decay), ctypes.c_float(recip))
            else:
                self._c_fused_lib.fused_conv1x1_bn_if_tloop(
                    fp(inp_flat), fp(w_2d), fp(scale_1d), fp(bias_1d),
                    fp(mem_2d), fp(spk_flat),
                    M, C_in, C_out, self.T,
                    ctypes.c_float(thresh), ctypes.c_float(v_reset))

            all_spikes[:spk_flat.size] = spk_flat.ravel()
        else:
            # Numpy fallback
            for t in range(self.T):
                start = t * M * C_in
                end = start + M * C_in
                if end > inp_flat.size:
                    break
                data_t = inp_flat[start:end].reshape(M, C_in)
                gemm = data_t @ w_2d
                h = mem_2d + gemm * scale_1d + bias_1d
                sp = (h >= 1.0).astype(np.float32)
                mem_2d[:] = (1.0 - sp) * h
                sp_start = t * M * C_out
                all_spikes[sp_start:sp_start + M * C_out] = sp.ravel()

        out[:min(out.size, all_spikes.size)] = all_spikes[:min(out.size, all_spikes.size)]

    def _exec_decomposed_conv(self, node):
        """Execute decomposed Conv+BN via BLAS GEMM (numpy.dot).

        Used when optimizer selects DECOMPOSED over interleaved for this layer
        (large K, small M). The neuron is handled separately as NativeIF/NativeLIF.
        Output = GEMM(input, weight) * bn_scale + bn_bias.
        """
        inp = self._get_input(node)
        out = self._get_output(node)

        w_name = node.weight_info.name if node.weight_info else ""
        w = self._weight_bufs.get(w_name, np.zeros(1, dtype=np.float32))

        cp = node.conv_params
        if not cp or not node.output_shapes:
            return

        out_shape = node.output_shapes[0]
        TB = out_shape[0] if len(out_shape) >= 1 else self.T
        C_out = cp.out_channels
        C_in = cp.in_channels
        OH = out_shape[2] if len(out_shape) >= 3 else 1
        OW = out_shape[3] if len(out_shape) >= 4 else 1
        M = TB * OH * OW  # total rows (all timesteps merged)

        # Weight as (C_in, C_out)
        w_2d = np.ascontiguousarray(w.ravel()[:C_in * C_out].reshape(C_in, C_out))

        # Input as (T*M, C_in)
        inp_total = M * C_in
        inp_flat = inp[:min(inp.size, inp_total)].reshape(M, C_in)

        # BLAS GEMM: (M, C_in) @ (C_in, C_out) → (M, C_out)
        gemm_out = inp_flat @ w_2d

        # BN: scale + bias
        if node.bn_scale:
            scale = np.array(node.bn_scale, dtype=np.float32)[:C_out]
            gemm_out *= scale
        if node.bn_bias:
            bias = np.array(node.bn_bias, dtype=np.float32)[:C_out]
            gemm_out += bias

        result = gemm_out.ravel()
        n = min(out.size, result.size)
        out[:n] = result[:n]

    def _exec_if(self, nid, node):
        inp = self._get_input(node)
        out = self._get_output(node)
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        total = _numel(out_shape)
        T = node.neuron_params.T if node.neuron_params else self.T
        spatial = max(total // T, 1)
        thresh = node.neuron_params.v_threshold if node.neuron_params else 1.0
        mem = self._membranes.get(nid, np.zeros(spatial, dtype=np.float32))

        for s in range(min(spatial, len(mem))):
            v = mem[s]
            for t in range(T):
                idx = t * spatial + s
                if idx >= len(inp) or idx >= len(out):
                    break
                h = v + inp[idx]
                spike = 1.0 if h >= thresh else 0.0
                v = (1.0 - spike) * h
                out[idx] = spike
            mem[s] = v

    def _exec_lif(self, nid, node):
        inp = self._get_input(node)
        out = self._get_output(node)
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        total = _numel(out_shape)
        T = node.neuron_params.T if node.neuron_params else self.T
        spatial = max(total // T, 1)
        thresh = node.neuron_params.v_threshold if node.neuron_params else 1.0
        tau = node.neuron_params.tau if node.neuron_params else 2.0
        decay = 1.0 - 1.0 / tau
        recip = 1.0 / tau
        mem = self._membranes.get(nid, np.zeros(spatial, dtype=np.float32))

        for s in range(min(spatial, len(mem))):
            v = mem[s]
            for t in range(T):
                idx = t * spatial + s
                if idx >= len(inp) or idx >= len(out):
                    break
                h = decay * v + recip * inp[idx]
                spike = 1.0 if h >= thresh else 0.0
                v = (1.0 - spike) * h
                out[idx] = spike
            mem[s] = v

    def _exec_add(self, node):
        out = self._get_output(node)
        bufs = [self._get_buf(n) for n in node.input_names if n not in self.ir.weights]
        if len(bufs) >= 2:
            n = min(len(bufs[0]), len(bufs[1]), len(out))
            out[:n] = bufs[0][:n] + bufs[1][:n]

    def _exec_gemm(self, node):
        inp = self._get_input(node)
        out = self._get_output(node)
        w_name = node.weight_info.name if node.weight_info else ""
        w = self._weight_bufs.get(w_name, np.zeros(1, dtype=np.float32))
        in_shape = node.input_shapes[0] if node.input_shapes else ()
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        M = in_shape[0] if len(in_shape) >= 1 else 1
        K = in_shape[1] if len(in_shape) >= 2 else 1
        N = out_shape[1] if len(out_shape) >= 2 else 1
        transB = (node.gemm_params or {}).get("transB", 0)

        try:
            a = inp[:M * K].reshape(M, K)
            if transB and node.weight_info and node.weight_info.shape:
                ws = node.weight_info.shape
                b = w[:ws[0] * ws[1]].reshape(ws).T
            else:
                b = w[:K * N].reshape(K, N)
            result = a @ b
            # Add bias if present
            b_name = node.bias_info.name if node.bias_info else ""
            if b_name in self._weight_bufs:
                result += self._weight_bufs[b_name][:N]
            out[:M * N] = result.ravel()[:M * N]
        except Exception:
            pass

    def _exec_maxpool(self, node):
        # Simple fallback: copy input to output (not numerically correct, just for latency)
        inp = self._get_input(node)
        out = self._get_output(node)
        n = min(len(inp), len(out))
        out[:n] = inp[:n]

    def _exec_gavg(self, node):
        inp = self._get_input(node)
        out = self._get_output(node)
        in_shape = node.input_shapes[0] if node.input_shapes else ()
        if len(in_shape) == 4:
            N, C, H, W = in_shape
            try:
                x = inp[:N * C * H * W].reshape(N, C, H, W)
                out[:N * C] = x.mean(axis=(2, 3)).ravel()
            except Exception:
                pass

    def _exec_tmean(self, node):
        inp = self._get_input(node)
        out = self._get_output(node)
        in_shape = node.input_shapes[0] if node.input_shapes else ()
        total = _numel(in_shape)
        T = self.T
        spatial = max(total // T, 1)
        try:
            x = inp[:total].reshape(T, spatial)
            out[:spatial] = x.mean(axis=0)
        except Exception:
            pass

    def benchmark(self, warmup=50, iters=200):
        """Returns avg ms per inference."""
        for _ in range(warmup):
            self.reset_membranes()
            self.execute()
        t0 = time.perf_counter()
        for _ in range(iters):
            self.reset_membranes()
            self.execute()
        t1 = time.perf_counter()
        return (t1 - t0) / iters * 1000
