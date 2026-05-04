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
    KernelVariant.TileLangDWConvBN,
    KernelVariant.TileLangFusedDWConvBNIF,
    KernelVariant.TileLangGroupedConvBN,
    KernelVariant.TileLangMatMulScale,
    KernelVariant.TileLangFusedMatMulLIF,
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
        self._use_python_runtime: bool = False
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
            build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     '.cache', f'sengine_B{batch_size}')
        eng._kernel_so_map = export_all_kernels(
            eng._kernels, eng._ir, build_dir, nvcc=nvcc, arch=arch)

        # 3. Wire up C++ executor
        if eng._py_engine._lazy_mode:
            logger.phase("BUILD", "Lazy mode: using Python runtime (OOM during pre-allocation)")
            eng._graph_captured = True
            eng._use_python_runtime = True
        else:
            eng._setup_cpp_executor()
            eng._cpp_exec.capture_graph()
            eng._cpp_exec.sync()
            torch.cuda.synchronize()
            eng._use_python_runtime = False

        eng._graph_captured = True

        elapsed = time.time() - t0
        mode = "Python lazy" if eng._use_python_runtime else "C++ CUDA Graph"
        logger.phase("BUILD", "Ready in %.1fs (%d ops, %s)", elapsed, len(eng._schedule), mode)
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
            build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     '.cache', f'sengine_B{batch_size}')
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
        # Python runtime path (CuDNNConv or lazy mode)
        if getattr(self, '_use_python_runtime', False):
            x_torch = torch.from_numpy(x).cuda()
            out = self._py_engine(x_torch)
            if out is not None:
                return out.float().cpu().numpy()
            return np.zeros(1)

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
        if getattr(self, '_use_python_runtime', False):
            return self._py_engine.benchmark(warmup=warmup, n_iters=iters)
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

    @staticmethod
    def _resolve_buf(act, ir, nid, max_depth=10):
        """Find the activation buffer for a node, tracing through ZeroCost predecessors."""
        buf = act.get(nid)
        if buf is not None:
            return buf
        cur = nid
        for _ in range(max_depth):
            pp = ir.predecessors(cur)
            if not pp:
                return None
            buf = act.get(pp[0])
            if buf is not None:
                return buf
            cur = pp[0]
        return None

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

        # Load TileLang .so files (skip tuples — fused attention loads separately)
        nid_tl_idx = {}
        for nid, so_path in self._kernel_so_map.items():
            if isinstance(so_path, tuple):
                continue  # fused attention: loaded in the handler below
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
            # Find the best input buffer from ALL predecessors.
            # Prefer LayoutTranspose (reformat) over ZeroCost chains.
            input_buf = None
            for pid in preds:
                buf = act.get(pid)
                if buf is not None:
                    input_buf = buf
                    break
            if input_buf is None:
                for pid in preds:
                    buf = self._resolve_buf(act, ir, pid)
                    if buf is not None:
                        input_buf = buf
                        break
            output_buf = act.get(nid)
            p = lambda t: t.data_ptr()  # noqa: E731

            if kv in TILELANG_VARIANTS and nid in nid_tl_idx:
                tl_idx = nid_tl_idx[nid]

                # MatMul variants: 3-arg TileLang (A, B, output) or cuBLAS fallback
                if kv in (KernelVariant.TileLangMatMul,
                          KernelVariant.TileLangMatMulScale):
                    if len(preds) >= 2 and output_buf is not None:
                        a = self._resolve_buf(act, ir, preds[0])
                        b = self._resolve_buf(act, ir, preds[1])
                        if a is not None and b is not None:
                            kern = engine.kernels.get(nid)
                            if kern is not None and nid in nid_tl_idx:
                                exe.set_tilelang_3(nid, nid_tl_idx[nid],
                                                    p(a), p(b), p(output_buf))
                            else:
                                # No TileLang kernel (dim alignment issue) → cuBLAS
                                M = a.shape[0]
                                K = a.shape[-1] if a.ndim >= 2 else 1
                                N = b.shape[-1] if b.ndim >= 2 else 1
                                exe.set_gemm_node(nid, p(a), p(b), p(output_buf), M, K, N)
                        else:
                            exe.set_skip_node(nid)
                    else:
                        exe.set_skip_node(nid)
                    continue

                # Fused Conv/Linear+BN+IF: 6-arg (input, weight, membrane, scale, bias, output)
                w = engine.weights.get(nid)
                if w is None:
                    w = engine.weights_1x1.get(nid)
                sc = engine.bn_scales.get(nid)
                bi = engine.bn_biases.get(nid)

                if input_buf is None or output_buf is None:
                    exe.set_skip_node(nid)
                    continue

                if kv in (KernelVariant.TileLangFusedConvBNIF,
                          KernelVariant.TileLangFusedConv1x1BNIF,
                          KernelVariant.TileLangLinearBNLIF,
                          KernelVariant.TileLangFusedMatMulLIF):
                    mem = None
                    for s in ir.successors(nid):
                        if s in engine.membranes:
                            mem = engine.membranes[s]
                            break
                    if w is not None and mem is not None and sc is not None and bi is not None:
                        exe.set_tilelang_6(nid, tl_idx,
                            p(input_buf), p(w), p(mem), p(sc), p(bi), p(output_buf))
                    elif w is not None and mem is not None:
                        # MatMul+LIF: 4-arg (A, B, membrane, spikes)
                        exe.set_tilelang_3(nid, tl_idx, p(input_buf), p(w), p(output_buf))
                    else:
                        exe.set_skip_node(nid)
                else:
                    # Conv+BN, Linear+BN: 5-arg (input, weight, scale, bias, output)
                    if w is not None and sc is not None and bi is not None:
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
                    a = self._resolve_buf(act, ir, preds[0])
                    b = self._resolve_buf(act, ir, preds[1])
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

            elif kv == KernelVariant.LayoutTranspose:
                perm = node.extra_attrs.get("perm")
                if input_buf is not None and output_buf is not None and perm and len(perm) == 4:
                    s = input_buf.shape
                    if len(s) == 4:
                        # direction: 0=NHWC→NCHW (perm 0,3,1,2), 1=NCHW→NHWC (perm 0,2,3,1)
                        direction = 1 if perm == [0, 2, 3, 1] else 0
                        N, d1, d2, d3 = s
                        if direction == 0:
                            # Input is NHWC (N,H,W,C): H=d1, W=d2, C=d3
                            exe.set_layout_transpose_node(nid, p(input_buf), p(output_buf),
                                                           N, d1, d2, d3, direction)
                        else:
                            # Input is NCHW (N,C,H,W): C=d1, H=d2, W=d3
                            exe.set_layout_transpose_node(nid, p(input_buf), p(output_buf),
                                                           N, d2, d3, d1, direction)
                    else:
                        exe.set_skip_node(nid)
                else:
                    exe.set_skip_node(nid)

            elif kv == KernelVariant.CuDNNConv:
                # Stem Conv (C_in<4): use naive CUDA Conv kernel
                w = engine.weights.get(nid)
                sc = engine.bn_scales.get(nid)
                bi = engine.bn_biases.get(nid)
                cp = node.conv_params
                if input_buf is not None and output_buf is not None and w is not None and cp and sc is not None and bi is not None:
                    s = input_buf.shape  # NHWC: (TB, H, W, C_in)
                    OH = (s[1] + 2*cp.pad_h - cp.kernel_h) // cp.stride_h + 1
                    OW = (s[2] + 2*cp.pad_w - cp.kernel_w) // cp.stride_w + 1
                    exe.set_naive_conv_node(nid,
                        p(input_buf), p(w), p(sc), p(bi), p(output_buf),
                        s[0], s[1], s[2], cp.in_channels, cp.out_channels,
                        cp.kernel_h, cp.kernel_w, cp.stride_h, cp.pad_h, OH, OW,
                        cp.groups)
                else:
                    exe.set_skip_node(nid)

            elif kv in (KernelVariant.FusedSpikformerAttn,
                        KernelVariant.FusedMaxformerAttn,
                        KernelVariant.FusedDSSAAttn,
                        KernelVariant.FusedTokenQKAttn):
                ap = node.attention_params
                so_paths = self._kernel_so_map.get(nid)
                if ap is None or not isinstance(so_paths, tuple) or len(so_paths) != 2:
                    exe.set_skip_node(nid)
                    continue

                # Load both TileLang .so kernels
                gemm1_idx = exe.load_tilelang(so_paths[0])
                gemm2_idx = exe.load_tilelang(so_paths[1])

                variant_map = {"spikformer": 0, "maxformer": 1, "dssa": 2, "token_qk": 3}
                variant = variant_map.get(ap.variant, 0)
                C = ap.num_heads * ap.head_dim
                heads = ap.num_heads
                hd = ap.head_dim

                # Gather predecessor buffers
                pred_bufs = [act.get(pid) for pid in preds if act.get(pid) is not None]
                q_buf = pred_bufs[0] if len(pred_bufs) > 0 else None
                k_buf = pred_bufs[1] if len(pred_bufs) > 1 else None
                v_buf = pred_bufs[2] if len(pred_bufs) > 2 else None

                if q_buf is None or output_buf is None:
                    exe.set_skip_node(nid)
                    continue

                mem = engine.membranes.get(nid)
                TB = q_buf.shape[0]
                N = ap.H * ap.W if ap.H > 0 else (q_buf.numel() // (TB * C))
                batch = TB * heads

                # Compute workspace: permuted Q/K/V + GEMM1 output
                if variant == 0:  # SpikFormer — still needs permute workspace
                    perm_size = TB * N * C
                    gemm1_size = batch * N * N
                    needs_permute = 0
                    ws_perm_q = 0
                    ws_perm_k = perm_size
                    ws_perm_v = perm_size * 2
                    ws_gemm1 = perm_size * 3
                    ws_total = perm_size * 3 + gemm1_size
                elif variant == 1:  # MaxFormer — fused NHWC kernels, only kv workspace
                    gemm1_size = batch * hd * hd
                    needs_permute = 0
                    ws_perm_q = 0
                    ws_perm_k = 0
                    ws_perm_v = 0
                    ws_gemm1 = 0
                    ws_total = gemm1_size
                else:
                    exe.set_skip_node(nid)
                    continue

                workspace = torch.empty(ws_total, dtype=torch.float16, device='cuda')
                self._attn_workspaces = getattr(self, '_attn_workspaces', [])
                self._attn_workspaces.append(workspace)

                lif_total = q_buf.numel()
                lif_spatial = mem.numel() if mem is not None else 0
                recip_tau = 1.0 / ap.attn_lif_tau if ap.attn_lif_tau > 0 else 0.5

                exe.set_fused_attn_node(
                    nid, variant, gemm1_idx, gemm2_idx,
                    p(q_buf), p(k_buf) if k_buf is not None else 0,
                    p(v_buf) if v_buf is not None else 0,
                    p(output_buf),
                    p(workspace), p(mem) if mem is not None else 0,
                    TB, heads, hd, N, ap.H, ap.W,
                    lif_total, lif_spatial,
                    ap.attn_lif_v_threshold, recip_tau,
                    needs_permute, 0, 0,
                    ws_gemm1, ws_perm_q, ws_perm_k, ws_perm_v)

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
