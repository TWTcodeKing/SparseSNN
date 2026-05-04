"""CUDA Graph-based SNN inference runtime.

This is the core execution engine. After the build phase (parse → optimize →
compile → schedule → memory plan), this class captures the full inference
path into a CUDA Graph for zero-overhead replay.

Usage:
    engine = CUDAGraphEngine()
    engine.build(ir, kernels, schedule, T=4, batch_size=16)
    output = engine(input_tensor)
    latency_ms = engine.benchmark()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sengine.ir import (
    OpType, KernelVariant, TensorLayout, BoundType, Node, EngineIR,
)
from sengine.logger import logger


class CUDAGraphEngine:
    """Python-wrapped CUDA Graph inference engine for SNNs.

    Captures the BA-MTTS-scheduled kernel sequence into a CUDA Graph
    and replays it for each inference call. All tensors are pre-allocated
    so the graph sees deterministic memory addresses.
    """

    def __init__(self):
        self.ir: Optional[EngineIR] = None
        self.schedule: list[int] = []
        self.kernels: dict[int, object] = {}
        self.T: int = 4
        self.B: int = 1
        self.TB: int = 4

        # GPU tensors
        self.weights: dict[int, torch.Tensor] = {}       # node_id → weight (NHWC FP16)
        self.weights_1x1: dict[int, torch.Tensor] = {}   # node_id → 1x1 weight (C_in, C_out) FP16
        self.bn_scales: dict[int, torch.Tensor] = {}     # node_id → scale (FP32)
        self.bn_biases: dict[int, torch.Tensor] = {}     # node_id → bias (FP32)
        self.membranes: dict[int, torch.Tensor] = {}     # neuron node_id → membrane (FP32)
        self.activations: dict[int, torch.Tensor] = {}   # node_id → output buffer

        # CUDA Graph
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_input: Optional[torch.Tensor] = None
        self._graph_output_nid: int = -1
        self._lazy_mode: bool = False

        # CUDA IF extension
        self._ext_if = None

    def build(self, ir: EngineIR, kernels: dict[int, object],
              schedule: list[int], T: int, batch_size: int):
        """Initialize the engine from build artifacts.

        Args:
            ir: Optimized EngineIR with shapes, weights, BN params
            kernels: node_id → compiled TileLang kernel (from TileLangCompiler)
            schedule: BA-MTTS execution order (list of node IDs)
            T: temporal steps
            batch_size: batch size
        """
        self.ir = ir
        self.kernels = kernels
        self.schedule = schedule
        self.T = T
        self.B = batch_size
        self.TB = T * batch_size

        self._allocate_weights()
        self._allocate_membranes()
        self._allocate_activations()

        # Load CUDA IF/LIF extension if any neuron nodes exist
        for nid in self.schedule:
            node = self.ir.nodes[nid]
            if node.assigned_kernel in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                from sengine.build.tilelang_compiler import get_cuda_if
                self._ext_if = get_cuda_if()
                break

        logger.phase("ENGINE", "Built: %d schedule ops, %d weights, %d membranes, %d buffers",
                     len(schedule), len(self.weights) + len(self.weights_1x1),
                     len(self.membranes), len(self.activations))

    def capture_graph(self):
        """Capture the inference path into a CUDA Graph.

        For transformer models with dynamic ZeroCost ops, CUDA Graph
        capture may fail. In that case, fall back to raw dispatch mode.
        """
        if self._ext_if is None:
            for nid in self.schedule:
                node = self.ir.nodes[nid]
                if node.assigned_kernel in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                    from sengine.build.tilelang_compiler import get_cuda_if
                    self._ext_if = get_cuda_if()
                    break

        try:
            # Dry run
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._execute_schedule()
            side.synchronize()

            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                with torch.cuda.graph(self.graph, stream=side):
                    self._execute_schedule()
            side.synchronize()
            logger.phase("ENGINE", "CUDA Graph captured (%d nodes in schedule)", len(self.schedule))
        except RuntimeError:
            # Graph capture failed (transformer models with dynamic ops)
            # Fall back to raw dispatch mode
            self.graph = None
            logger.phase("ENGINE", "CUDA Graph not available, using raw dispatch (%d nodes)", len(self.schedule))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference. Input shape: (B, C, H, W) NCHW FP16 or FP32."""
        if x.dtype != torch.float16:
            x = x.half()

        # Convert NCHW → NHWC, replicate T times
        if x.ndim == 4:
            if x.shape[0] == self.B:
                x = x.repeat(self.T, 1, 1, 1)  # (B,C,H,W) → (TB,C,H,W)
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # → (TB,H,W,C)
        else:
            x_nhwc = x.contiguous()

        # Copy into graph's input buffer
        if self._graph_input is not None:
            if self._graph_input.shape == x_nhwc.shape:
                self._graph_input.copy_(x_nhwc)
            else:
                # Shape mismatch — try to fit (the buffer might be larger due to padding)
                slices = tuple(slice(0, s) for s in x_nhwc.shape)
                self._graph_input[slices].copy_(x_nhwc)

        # Reset membranes
        for mem in self.membranes.values():
            mem.zero_()

        # Replay graph
        if self.graph is not None:
            self.graph.replay()
        else:
            self._execute_schedule()

        # Return output
        out = self.activations.get(self._graph_output_nid)
        if out is not None:
            return out.clone()
        return torch.tensor(0.0)

    def reset_state(self):
        """Reset all neuron membrane states to zero."""
        for mem in self.membranes.values():
            mem.zero_()

    def benchmark(self, warmup: int = 200, n_iters: int = 1000) -> float:
        """Return mean inference latency in milliseconds."""
        for mem in self.membranes.values():
            mem.zero_()

        replay = self.graph.replay if self.graph is not None else self._execute_schedule

        for _ in range(warmup):
            replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters

    # ─── Internal ───

    def _allocate_weights(self):
        """Convert ONNX weights to NHWC FP16 tensors on GPU."""
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if node.weight_info is None or node.weight_info.name not in self.ir.weights:
                continue

            w_np = self.ir.weights[node.weight_info.name]
            cp = node.conv_params

            if node.op_type == OpType.Conv2d and cp:
                if cp.groups > 1 and cp.groups == cp.in_channels:
                    # Depthwise Conv: (C, 1, K, K) → (C, K, K)
                    w = w_np.reshape(cp.out_channels, cp.kernel_h, cp.kernel_w).astype(np.float16)
                    self.weights[nid] = torch.from_numpy(w.copy()).cuda()
                elif cp.kernel_h == 1 and cp.kernel_w == 1 and cp.groups == 1:
                    # Standard 1x1 Conv: (C_out, C_in, 1, 1) → (C_in, C_out)
                    w = torch.from_numpy(w_np.astype(np.float16)).cuda()
                    self.weights_1x1[nid] = w.reshape(cp.out_channels, cp.in_channels).t().contiguous()
                else:
                    # General Conv: (C_out, C_in/groups, K, K) → (K, K, C_in/groups, C_out)
                    w_nhwc = w_np.transpose(2, 3, 1, 0).astype(np.float16)
                    self.weights[nid] = torch.from_numpy(w_nhwc.copy()).cuda()

            elif node.op_type == OpType.Gemm:
                # FC: (out, in) — keep as-is for cuBLAS
                self.weights[nid] = torch.from_numpy(w_np.astype(np.float16)).cuda()

            elif node.op_type == OpType.Linear:
                # Linear weight: (K, N) for matmul — keep as-is
                self.weights[nid] = torch.from_numpy(w_np.astype(np.float16)).cuda()

            # BN scale/bias
            if node.bn_scale is not None:
                self.bn_scales[nid] = torch.tensor(node.bn_scale, dtype=torch.float32).cuda()
            if node.bn_bias is not None:
                self.bn_biases[nid] = torch.tensor(node.bn_bias, dtype=torch.float32).cuda()

    def _conv_shape_ok(self, x, node):
        """Check if input is valid NHWC for the compiled TileLang Conv kernel.

        TileLang Conv kernels are compiled from ONNX NCHW output shapes
        (TB, C, OH, OW) and expect NHWC input (TB, H, W, C_in). The
        compiled kernel has specific M = TB * OH * OW baked in.

        Returns False if the actual activation doesn't match — this happens
        when attention Reshape/Transpose ops changed the tensor layout.
        """
        if x.ndim != 4:
            return False
        cp = node.conv_params
        if cp is None:
            return True
        # NHWC: last dim must be C_in
        if x.shape[-1] != cp.in_channels:
            return False
        # Verify total elements match compiled expectation
        if node.output_shapes and len(node.output_shapes[0]) == 4:
            TB_exp, C_exp, OH, OW = node.output_shapes[0]
            # Compiled M = TB * OH * OW. Input NHWC should have
            # shape[0]*shape[1]*shape[2] == TB * H * W (where H >= OH).
            compiled_M = TB_exp * OH * OW
            actual_spatial = x.shape[0] * x.shape[1] * x.shape[2]
            # For stride>1 conv, input spatial is larger than output
            stride = cp.stride_h
            if stride > 1:
                compiled_M_in = TB_exp * OH * stride * OW * stride
            else:
                compiled_M_in = compiled_M
            if actual_spatial != compiled_M_in:
                return False
        return True

    def _conv_bn_fallback(self, x, node, w, scale, bias):
        """cuDNN Conv+BN fallback when TileLang kernel shape doesn't match.

        Handles attention-path Conv projections where Reshape/Transpose ops
        changed the activation layout. Uses F.conv2d (cuDNN) in NCHW which
        handles any input shape natively. Zero perf impact — these are small
        1×1 or patch-size GEMMs where cuDNN is already optimal.
        """
        cp = node.conv_params
        if cp is None:
            return x

        # Ensure 4D NCHW input
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim == 4 and x.shape[1] != cp.in_channels:
            # NHWC (TB, H, W, C) → NCHW (TB, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()

        # Reshape weight to 4D NCHW (C_out, C_in/groups, KH, KW)
        if w.ndim == 2:
            # BN-folded 1×1 conv: (C_out, C_in) → (C_out, C_in, 1, 1)
            w = w.unsqueeze(-1).unsqueeze(-1)
        elif w.ndim == 4 and w.shape[0] == cp.kernel_h:
            # TileLang NHWC weight (KH, KW, C_in, C_out) → NCHW
            w = w.permute(3, 2, 0, 1).contiguous()

        stride = (cp.stride_h, cp.stride_w if cp.stride_w else cp.stride_h)
        padding = (cp.pad_h, cp.pad_w if cp.pad_w else cp.pad_h)
        dilation = (cp.dilation_h, cp.dilation_w if cp.dilation_w else cp.dilation_h)

        result = F.conv2d(x.float(), w.float(), stride=stride,
                          padding=padding, dilation=dilation, groups=cp.groups)

        # Apply BN: output = result * scale + bias (channel-wise)
        if scale is not None and bias is not None:
            result = result * scale.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        return result.half()

    def _allocate_membranes(self):
        """Allocate FP32 membrane state tensors for neuron and fused attention nodes."""
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]

            # Fused attention: allocate membrane for the internal attn_lif
            if node.op_type == OpType.FusedAttention and node.attention_params:
                ap = node.attention_params
                shape = node.output_shapes[0] if node.output_shapes else ()
                if ap.variant == "spikformer":
                    # attn_lif operates on (TB, N, C) after merge-heads.
                    # Derive from total elements: total = TB * N * C
                    C = ap.num_heads * ap.head_dim
                    total = 1
                    for d in shape:
                        total *= d
                    TB_val = shape[0] if shape else self.TB
                    N = total // (TB_val * C) if (TB_val * C) > 0 else 1
                    B = max(TB_val // self.T, 1)
                    self.membranes[nid] = torch.zeros(B * N, C,
                                                      dtype=torch.float32, device='cuda')
                elif ap.variant == "token_qk" and len(shape) == 4:
                    # attn_lif operates on (TB, heads, 1, N) — small membrane
                    N, C, H, W = shape
                    B = max(N // self.T, 1)
                    num_h = ap.num_heads
                    spatial = H * W  # N tokens
                    self.membranes[nid] = torch.zeros(B * num_h, spatial,
                                                      dtype=torch.float32, device='cuda')
                elif ap.variant in ("maxformer", "dssa") and len(shape) == 4:
                    # Output (TB, C, H, W): attn_lif operates on merged spatial
                    N, C, H, W = shape
                    B = max(N // self.T, 1)
                    self.membranes[nid] = torch.zeros(B * H * W, C,
                                                      dtype=torch.float32, device='cuda')
                continue

            if node.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
                continue
            if not node.output_shapes:
                continue

            shape = node.output_shapes[0]
            if len(shape) == 4:
                N, C, H, W = shape
                B = max(N // self.T, 1)  # clamp for post-temporal-mean nodes
                # 2D membrane: (B*H*W, C) for temporal-safe CUDA IF kernel
                self.membranes[nid] = torch.zeros(B * H * W, C,
                                                  dtype=torch.float32, device='cuda')
            elif len(shape) == 2:
                M, N_dim = shape
                spatial = max(M // self.T, 1)
                self.membranes[nid] = torch.zeros(spatial, N_dim,
                                                  dtype=torch.float32, device='cuda')

    def _allocate_activations(self):
        """Pre-allocate output buffers for all nodes in the schedule.

        ZeroCost nodes (Reshape, Transpose, Identity) don't allocate new
        buffers — they're resolved at dispatch time by reshaping the input.
        This saves significant GPU memory for transformer models.

        If pre-allocation OOMs, falls back to allocating only the input
        buffer and creating intermediate buffers lazily at dispatch time.
        """
        _ZEROCOST_OPS = {OpType.Reshape, OpType.Transpose, OpType.Identity,
                         OpType.Flatten}
        def _fix_shape(shape):
            """Replace 0-valued dims (ONNX dynamic) with runtime values."""
            fixed = list(shape)
            for i, d in enumerate(fixed):
                if d == 0:
                    fixed[i] = self.B if i == 0 else 1
            return tuple(fixed)

        try:
            for nid in self.ir.topo_order:
                node = self.ir.nodes[nid]
                if not node.output_shapes:
                    continue

                # Only pre-allocate buffers for non-ZeroCost nodes.
                # ZeroCost nodes (Reshape, Transpose, Identity, Flatten) store
                # their results at dispatch time — no pre-allocation needed.
                if (node.op_type in _ZEROCOST_OPS
                        and node.assigned_kernel != KernelVariant.LayoutTranspose):
                    continue
                shape = _fix_shape(node.output_shapes[0])
                # Determine buffer shape based on kernel type:
                # - Conv/Pool/Neuron: NHWC (N,H,W,C) for TileLang spatial kernels
                # - MatMul/Linear: flattened 2D (M,N) for GEMM kernels
                # - Everything else: ONNX shape as-is
                _NHWC_OPS = {OpType.Conv2d, OpType.MaxPool, OpType.GlobalAvgPool,
                             OpType.Add, OpType.IF, OpType.LIF, OpType.MS,
                             OpType.Tile, OpType.Sub, OpType.Mul, OpType.Scale}
                _GEMM_OPS = {OpType.MatMul, OpType.Linear}
                if len(shape) == 4 and node.op_type in _NHWC_OPS:
                    N, C, H, W = shape
                    self.activations[nid] = torch.zeros(N, H, W, C,
                                                        dtype=torch.float16, device='cuda')
                elif (len(shape) == 4
                      and node.assigned_kernel == KernelVariant.LayoutTranspose):
                    # LayoutTranspose: allocate buffer in the TARGET layout.
                    perm = node.extra_attrs.get("perm", [])
                    if perm == [0, 2, 3, 1]:
                        # NCHW→NHWC: output is physically (N, H, W, C)
                        N, C, H, W = shape
                        self.activations[nid] = torch.zeros(N, H, W, C,
                                                            dtype=torch.float16, device='cuda')
                    elif perm == [0, 3, 1, 2]:
                        # NHWC→NCHW: output is physically (N, C, H, W)
                        N, C, H, W = shape
                        self.activations[nid] = torch.zeros(N, C, H, W,
                                                            dtype=torch.float16, device='cuda')
                    else:
                        self.activations[nid] = torch.zeros(*shape,
                                                            dtype=torch.float16, device='cuda')
                elif node.op_type in _GEMM_OPS:
                    # Flatten to 2D for GEMM kernel: total_elements = M * N_out
                    total = 1
                    for d in shape:
                        total *= d
                    N_out = shape[-1] if len(shape) >= 1 else 1
                    M = total // N_out if N_out > 0 else total
                    self.activations[nid] = torch.zeros(M, N_out,
                                                        dtype=torch.float16, device='cuda')
                elif len(shape) >= 1:
                    self.activations[nid] = torch.zeros(*shape,
                                                        dtype=torch.float16, device='cuda')
        except torch.cuda.OutOfMemoryError:
            # OOM during pre-allocation — free everything and allocate
            # only the graph input buffer. The rest will be created
            # lazily at dispatch time (slower but fits in memory).
            logger.phase("ENGINE", "OOM during pre-allocation, switching to lazy mode")
            self.activations.clear()
            torch.cuda.empty_cache()
            self._lazy_mode = True
            # Allocate just the input (Tile) node
            for nid in self.ir.topo_order:
                node = self.ir.nodes[nid]
                if node.op_type == OpType.Tile and node.output_shapes:
                    shape = node.output_shapes[0]
                    if len(shape) == 4:
                        N, C, H, W = shape
                        self.activations[nid] = torch.zeros(N, H, W, C,
                                                            dtype=torch.float16, device='cuda')
                    break

        # Neuron buffers: derive shape from predecessor
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if node.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
                continue
            preds = self.ir.predecessors(nid)
            if preds and preds[0] in self.activations:
                self.activations[nid] = torch.zeros_like(self.activations[preds[0]])

        # Identify graph output: the node that produces the model's final
        # output. Walk topo_order backwards to find the last node that has
        # a pre-allocated activation buffer (TemporalMean, Gemm, MatMul,
        # or any non-ZeroCost node). BA-MTTS reorders execution, so
        # schedule[-1] is NOT necessarily the output node.
        self._graph_output_nid = self.schedule[-1] if self.schedule else -1
        _OUTPUT_OPS = {OpType.TemporalMean, OpType.Gemm, OpType.MatMul,
                       OpType.Linear, OpType.GlobalAvgPool}
        for nid in reversed(self.ir.topo_order):
            node = self.ir.nodes[nid]
            if node.op_type in _OUTPUT_OPS and nid in self.activations:
                self._graph_output_nid = nid
                break

        # Identify graph input: the Tile node's output buffer (TB, H, W, C)
        # The user will provide (B, C, H, W) NCHW which gets replicated T times
        # and converted to NHWC before being copied into this buffer.
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if node.op_type == OpType.Tile and nid in self.activations:
                self._graph_input = self.activations[nid]
                self._tile_nid = nid
                break
        else:
            # No Tile node — first node with no predecessors is the input
            for nid in self.ir.topo_order:
                node = self.ir.nodes[nid]
                if not self.ir.predecessors(nid) and nid in self.activations:
                    self._graph_input = self.activations[nid]
                    self._tile_nid = -1
                    break

    def _execute_schedule(self):
        """Execute the BA-MTTS schedule (one full inference pass)."""
        # Clear any deferred CUDA errors from previous execution
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        # Use BA-MTTS schedule order for hardware overlap (C↔M transitions).
        # With fused attention ops, ZeroCost dependency violations in the
        # attention path are eliminated — all attention ops are single nodes.
        order = self.schedule
        for nid in order:
            self._dispatch_node(nid)

    def _dispatch_node(self, nid: int):
        """Dispatch a single node based on its kernel variant."""
        node = self.ir.nodes[nid]
        kv = node.assigned_kernel

        # Gather input activations from predecessors.
        # For spatial kernels (Conv etc.), prefer LayoutTranspose reformats
        # as the primary input since they provide correctly-formatted NHWC data.
        _SPATIAL_KV = {
            KernelVariant.TileLangConvBN, KernelVariant.TileLangConv1x1BN,
            KernelVariant.TileLangStemConvBN, KernelVariant.TileLangDWConvBN,
            KernelVariant.TileLangGroupedConvBN,
            KernelVariant.TileLangFusedConvBNIF,
            KernelVariant.TileLangFusedDWConvBNIF,
            KernelVariant.CuDNNConv, KernelVariant.CuDNNPool,
        }
        inputs = []
        layout_buf = None
        for pred_nid in self.ir.predecessors(nid):
            buf = self.activations.get(pred_nid)
            if buf is not None:
                pn = self.ir.nodes.get(pred_nid)
                if (kv in _SPATIAL_KV and pn
                        and pn.assigned_kernel == KernelVariant.LayoutTranspose):
                    layout_buf = buf
                else:
                    inputs.append(buf)
        if layout_buf is not None:
            inputs.insert(0, layout_buf)

        if not inputs and nid in self.activations:
            inputs = [self.activations[nid]]

        if not inputs:
            return

        x = inputs[0]
        out_buf = self.activations.get(nid)

        if kv in (KernelVariant.TileLangConvBN, KernelVariant.TileLangGroupedConvBN):
            kern = self.kernels.get(nid)
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            if kern is not None and w is not None and sc is not None and bi is not None:
                result = kern(x, w, sc, bi)
                if out_buf is not None:
                    out_buf.copy_(result)
                else:
                    self.activations[nid] = result

        elif kv == KernelVariant.TileLangConv1x1BN:
            kern = self.kernels.get(nid)
            w = self.weights_1x1.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            if kern is not None and w is not None and sc is not None and bi is not None:
                result = kern(x, w, sc, bi)
                if out_buf is not None:
                    out_buf.copy_(result)
                else:
                    self.activations[nid] = result

        elif kv in (KernelVariant.TileLangFusedConvBNIF,
                    KernelVariant.TileLangFusedConv1x1BNIF):
            # Per-timestep fused Conv+BN+IF: call T times, once per timestep
            kern = self.kernels.get(nid)
            w = self.weights.get(nid)
            if kv == KernelVariant.TileLangFusedConv1x1BNIF:
                w = self.weights_1x1.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            # Find successor neuron's membrane
            succs = self.ir.successors(nid)
            mem = None
            for s in succs:
                if s in self.membranes:
                    mem = self.membranes[s]
                    break
            if kern is not None and w is not None and sc is not None and bi is not None:
                if mem is not None:
                    # Reshape membrane from 2D to 4D for fused kernel
                    cp = node.conv_params
                    if cp and node.output_shapes:
                        TB_out, C_out, OH_out, OW_out = node.output_shapes[0]
                        B_out = TB_out // self.T
                        mem_4d = mem.reshape(B_out, OH_out, OW_out, C_out)
                        # Call T times — each processes one timestep
                        spk_frames = []
                        for t in range(self.T):
                            frame = x[t*B_out:(t+1)*B_out]  # (B, H, W, C) for timestep t
                            spk_t = kern(frame, w, mem_4d, sc, bi)
                            spk_frames.append(spk_t)
                        result = torch.cat(spk_frames, dim=0)  # (TB, OH, OW, C)
                        self.activations[nid] = result
                    else:
                        # No membrane — just run as Conv+BN
                        result = kern(x, w, sc, bi)
                        self.activations[nid] = result

        elif kv == KernelVariant.TileLangDWConvBN:
            kern = self.kernels.get(nid)
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            if kern is not None and w is not None and sc is not None and bi is not None:
                result = kern(x, w, sc, bi)
                self.activations[nid] = result

        elif kv == KernelVariant.TileLangFusedDWConvBNIF:
            kern = self.kernels.get(nid)
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            succs = self.ir.successors(nid)
            mem = None
            for s in succs:
                if s in self.membranes:
                    mem = self.membranes[s]
                    break
            if kern is not None and w is not None and sc is not None and bi is not None and mem is not None:
                cp = node.conv_params
                if cp and node.output_shapes:
                    TB_out, C_out, OH_out, OW_out = node.output_shapes[0]
                    B_out = TB_out // self.T
                    mem_4d = mem.reshape(B_out, OH_out, OW_out, C_out)
                    spk_frames = []
                    for t in range(self.T):
                        frame = x[t*B_out:(t+1)*B_out]
                        spk_t = kern(frame, w, mem_4d, sc, bi)
                        spk_frames.append(spk_t)
                    self.activations[nid] = torch.cat(spk_frames, dim=0)

        elif kv == KernelVariant.CuDNNConv:
            # cuDNN Conv+BN: NHWC input → NCHW compute → NHWC output
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            cp = node.conv_params
            if w is not None and cp:
                # NHWC → NCHW for F.conv2d
                if x.ndim == 4 and x.shape[-1] == cp.in_channels:
                    x_nchw = x.permute(0, 3, 1, 2).to(torch.float32)
                elif x.ndim == 4 and x.shape[1] == cp.in_channels:
                    x_nchw = x.to(torch.float32)  # already NCHW
                else:
                    x_nchw = x.to(torch.float32)
                # Weight: NHWC (KH,KW,Cin,Cout) → NCHW (Cout,Cin,KH,KW)
                if w.ndim == 4 and w.shape[0] != cp.out_channels:
                    w_nchw = w.permute(3, 2, 0, 1).to(torch.float32)
                elif w.ndim == 2:
                    w_nchw = w.unsqueeze(-1).unsqueeze(-1).to(torch.float32)
                else:
                    w_nchw = w.to(torch.float32)
                conv_out = F.conv2d(x_nchw, w_nchw, stride=cp.stride_h, padding=cp.pad_h,
                                    dilation=cp.dilation_h, groups=cp.groups)
                if sc is not None and bi is not None:
                    conv_out = conv_out * sc.view(1, -1, 1, 1) + bi.view(1, -1, 1, 1)
                # Output NHWC (contract says NHWC)
                result = conv_out.half().permute(0, 2, 3, 1).contiguous()
                self.activations[nid] = result

        elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
            mem = self.membranes.get(nid)
            if self._ext_if is not None and mem is not None:
                # Flatten to 2D for the CUDA kernel
                x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                v_thresh = 1.0
                if node.neuron_params:
                    v_thresh = node.neuron_params.v_threshold

                if kv == KernelVariant.CUDAVec4LIF:
                    # LIF: use decay parameter from neuron params
                    recip_tau = 0.5
                    if node.neuron_params and node.neuron_params.tau > 0:
                        recip_tau = 1.0 / node.neuron_params.tau
                    result = self._ext_if.lif_neuron(x_flat, mem, v_thresh, recip_tau)
                else:
                    result = self._ext_if.if_neuron(x_flat, mem, v_thresh)

                result_shaped = result.reshape(x.shape)
                self.activations[nid] = result_shaped

        elif kv == KernelVariant.TileLangLinearBN:
            kern = self.kernels.get(nid)
            # For Linear, weight comes from the node's weight_info
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            if kern is not None and w is not None and sc is not None and bi is not None:
                result = kern(x, w, sc, bi)
                self.activations[nid] = result

        elif kv in (KernelVariant.TileLangMatMul, KernelVariant.TileLangMatMulScale):
            # Attention matmuls: use torch.matmul (cuBLAS) which handles
            # any input dimensionality (2D, 3D batched, 4D batched).
            # For classifier MatMul, the weight comes via a Transpose node
            # that reads from ir.weights — resolve it if missing.
            if len(inputs) < 2:
                # Try to find weight from predecessor Transpose that has no buffer
                for pid in self.ir.predecessors(nid):
                    if pid not in self.activations:
                        pn = self.ir.nodes.get(pid)
                        if pn and pn.op_type == OpType.Transpose:
                            perm = pn.extra_attrs.get("perm")
                            for in_name in pn.input_names:
                                w = self.ir.weights.get(in_name)
                                if w is not None:
                                    wt = torch.from_numpy(w).half().cuda() if not isinstance(w, torch.Tensor) else w.half().cuda()
                                    if perm and len(perm) == wt.ndim:
                                        wt = wt.permute(*perm).contiguous()
                                    inputs.append(wt)
                                    self.activations[pid] = wt
                                    break
            if len(inputs) >= 2:
                result = torch.matmul(inputs[0], inputs[1])
                scale = node.extra_attrs.get("scale_value", 1.0)
                if scale != 1.0:
                    result = result * scale
                self.activations[nid] = result

        elif kv == KernelVariant.TileLangFusedMatMulLIF:
            # Per-timestep fused MatMul+LIF: call T times, once per timestep
            # Same pattern as TileLangFusedConvBNIF (lines above)
            kern = self.kernels.get(nid)
            if len(inputs) >= 2:
                A, B_mat = inputs[0], inputs[1]
                # Find successor neuron's membrane
                succs = self.ir.successors(nid)
                mem = None
                for s in succs:
                    if s in self.membranes:
                        mem = self.membranes[s]
                        break
                if kern is not None and mem is not None:
                    M_full = A.shape[0]
                    M_per_t = M_full // self.T
                    spk_frames = []
                    for t in range(self.T):
                        a_t = A[t * M_per_t : (t + 1) * M_per_t]
                        b_t = B_mat[t * M_per_t : (t + 1) * M_per_t]
                        spk_t = kern(a_t, b_t, mem)
                        spk_frames.append(spk_t)
                    result = torch.cat(spk_frames, dim=0)
                    self.activations[nid] = result
                else:
                    # Fallback: torch.matmul (no membrane or no kernel)
                    result = torch.matmul(A, B_mat)
                    self.activations[nid] = result

        elif kv == KernelVariant.Elementwise:
            if node.op_type == OpType.Add and len(inputs) >= 2:
                a, b = inputs[0], inputs[1]
                # Handle shape mismatches from reshape/transpose chains
                if a.shape != b.shape:
                    try:
                        result = a + b  # broadcasting
                    except RuntimeError:
                        # Last resort: flatten, align, add
                        n = min(a.numel(), b.numel())
                        result = a.flatten()[:n] + b.flatten()[:n]
                        result = result.reshape(a.shape if a.numel() <= b.numel() else b.shape)
                else:
                    result = a + b
                self.activations[nid] = result
            elif node.op_type == OpType.Scale:
                scale = node.extra_attrs.get("scale_value", 1.0)
                self.activations[nid] = inputs[0] * scale
            elif node.op_type == OpType.Mul and len(inputs) >= 2:
                self.activations[nid] = inputs[0] * inputs[1]
            elif len(inputs) >= 1:
                self.activations[nid] = inputs[0]

        elif kv == KernelVariant.CuDNNPool:
            if node.op_type == OpType.MaxPool and node.pool_params:
                pp = node.pool_params
                ks = pp.get('kernel_size', 3)
                stride = pp.get('stride', 2)
                pad = pp.get('padding', 1)
                # Input is NHWC (contract). Permute to NCHW for F.max_pool2d.
                x_nchw = x.permute(0, 3, 1, 2)
                result_nchw = F.max_pool2d(x_nchw, ks, stride, pad)
                result = result_nchw.permute(0, 2, 3, 1).contiguous()
                self.activations[nid] = result
            elif node.op_type == OpType.GlobalAvgPool:
                # NHWC: (N, H, W, C) → NCHW → AdaptiveAvgPool → NHWC
                x_nchw = x.permute(0, 3, 1, 2)
                result = F.adaptive_avg_pool2d(x_nchw, 1).permute(0, 2, 3, 1).contiguous()
                self.activations[nid] = result

        elif kv == KernelVariant.TemporalMean:
            # (TB, ...) → reshape (T, B, ...) → mean over T → (B, ...)
            shape = x.shape
            if shape[0] == self.TB:
                # Input has T*B first dim → collapse T
                T_shape = (self.T, self.B) + shape[1:]
                result = x.reshape(T_shape).mean(dim=0)
            else:
                # Input already has B first dim (T already collapsed upstream)
                result = x
            self.activations[nid] = result

        elif kv == KernelVariant.CuBLASGemm:
            w = self.weights.get(nid)
            if w is not None:
                # Flatten to 2D: (B, in_features) for linear classifier
                if x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)
                result = F.linear(x, w)
                self.activations[nid] = result

        elif kv in (KernelVariant.FusedSpikformerAttn,
                    KernelVariant.FusedMaxformerAttn,
                    KernelVariant.FusedDSSAAttn,
                    KernelVariant.FusedTokenQKAttn):
            self._dispatch_fused_attention(nid, node, inputs)

        elif kv == KernelVariant.TileRepeat:
            pass

        elif kv == KernelVariant.LayoutTranspose:
            # Explicit layout reformat: NCHW↔NHWC permutation
            perm = node.extra_attrs.get("perm")
            if perm and inputs:
                result = inputs[0].permute(*perm).contiguous()
                self.activations[nid] = result

        elif kv == KernelVariant.ZeroCost:
            if node.op_type == OpType.Transpose:
                perm = node.extra_attrs.get("perm")
                # For weight transposes, the input may be in ir.weights
                if perm and not inputs:
                    for in_name in node.input_names:
                        w = self.ir.weights.get(in_name)
                        if w is not None:
                            inputs = [w.half().to('cuda')]
                            break
                if perm and inputs:
                    t = inputs[0]
                    if len(perm) == t.ndim:
                        self.activations[nid] = t.permute(*perm)
                    elif len(perm) > t.ndim:
                        offset = len(perm) - t.ndim
                        adj = [p - offset for p in perm[offset:]]
                        self.activations[nid] = t.permute(*adj)
                    else:
                        self.activations[nid] = t
                elif inputs:
                    self.activations[nid] = inputs[0]
            elif node.op_type == OpType.Reshape:
                target = node.extra_attrs.get("target_shape")
                # Use only the first input (data), not shape constants
                if target and inputs:
                    inputs = [inputs[0]]
                    # Handle -1 and 0 dimensions
                    shape = list(target)
                    for i, d in enumerate(shape):
                        if d == 0 and i < inputs[0].ndim:
                            shape[i] = inputs[0].shape[i]
                    reshaped = inputs[0].reshape(*shape)
                    # Attention exit: Reshape back to spatial 4D whose successor
                    # is a spatial op (LIF/Conv). Convert NCHW → NHWC so
                    # the spatial kernel receives correct layout.
                    # Detect: input shape != output shape (structural reshape,
                    # not just a view).
                    if (len(shape) == 4
                            and list(inputs[0].shape) != list(reshaped.shape)):
                        succs = self.ir.successors(nid)
                        needs_nhwc = any(
                            self.ir.nodes.get(s)
                            and self.ir.nodes[s].op_type in (
                                OpType.IF, OpType.LIF, OpType.MS, OpType.Conv2d)
                            for s in succs
                        )
                        if needs_nhwc:
                            reshaped = reshaped.permute(0, 2, 3, 1).contiguous()
                    self.activations[nid] = reshaped
                elif inputs:
                    self.activations[nid] = inputs[0]
            elif node.op_type == OpType.Flatten and inputs:
                axis = node.extra_attrs.get("axis", 1)
                t = inputs[0]
                if axis == 1:
                    self.activations[nid] = t.reshape(t.shape[0], -1)
                else:
                    pre = 1
                    for d in t.shape[:axis]:
                        pre *= d
                    self.activations[nid] = t.reshape(pre, -1)
            elif node.extra_attrs.get("original_op") == "Slice" and inputs:
                # Actual Slice execution using parsed starts/ends/axes
                axes = node.extra_attrs.get("axes")
                starts = node.extra_attrs.get("starts")
                ends = node.extra_attrs.get("ends")
                t = inputs[0]
                if axes is not None and starts is not None and ends is not None:
                    for a, s, e in zip(axes, starts, ends):
                        dim_size = t.shape[a]
                        s_c = max(0, s) if s >= 0 else max(0, dim_size + s)
                        e_c = min(e, dim_size) if e > 0 else max(0, dim_size + e)
                        t = t.narrow(a, s_c, e_c - s_c)
                self.activations[nid] = t.contiguous()
            elif inputs:
                self.activations[nid] = inputs[0]

    def _dispatch_fused_attention(self, nid: int, node, inputs):
        """Dispatch fused attention custom ops.

        Handles NHWC↔NCHW conversion internally — no layout reformats needed.
        Uses torch.matmul (cuBLAS) for the matmul ops and the CUDA LIF extension
        for the embedded attention neuron.
        """
        ap = node.attention_params
        if ap is None:
            return
        mem = self.membranes.get(nid)
        kv = node.assigned_kernel

        if kv == KernelVariant.FusedSpikformerAttn and len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            C = ap.num_heads * ap.head_dim
            TB = q.shape[0]

            # Flatten to (TB, N, C) — input may be 3D (TB,N,C) or 4D (TB,H,W,C)
            # from shape propagation. Total elements = TB * N * C, derive N.
            total = q.numel() // TB
            N = total // C
            q = q.reshape(TB, N, C)
            k = k.reshape(TB, N, C)
            v = v.reshape(TB, N, C)

            # Reshape to multi-head: (TB, N, C) → (TB, heads, N, head_dim)
            q = q.view(TB, N, ap.num_heads, ap.head_dim).permute(0, 2, 1, 3)
            k = k.view(TB, N, ap.num_heads, ap.head_dim).permute(0, 2, 1, 3)
            v = v.view(TB, N, ap.num_heads, ap.head_dim).permute(0, 2, 1, 3)

            # Attention: (Q @ K^T) * scale → @ V
            attn = (q @ k.transpose(-2, -1)) * ap.scale
            x = attn @ v  # (TB, heads, N, head_dim)

            # Merge heads → (TB, N, C)
            x = x.transpose(1, 2).reshape(TB, N, C)

            # attn_lif
            x = self._run_attn_lif(x, mem, ap)

            # Reshape to match the pre-allocated output buffer (may be 4D)
            out_buf = self.activations.get(nid)
            if out_buf is not None:
                x = x.reshape(out_buf.shape)
            self.activations[nid] = x

        elif kv == KernelVariant.FusedMaxformerAttn and len(inputs) >= 3:
            q, k, v = inputs[0], inputs[1], inputs[2]
            TB = q.shape[0]
            C = ap.num_heads * ap.head_dim
            H, W = ap.H, ap.W
            N = H * W

            # Input is NHWC (TB, H, W, C) → NCHW for multi-head reshape
            q = q.permute(0, 3, 1, 2).contiguous()  # (TB, C, H, W)
            k = k.permute(0, 3, 1, 2).contiguous()
            v = v.permute(0, 3, 1, 2).contiguous()

            # Reshape to multi-head: (TB, heads, head_dim, N) → transpose
            q = q.view(TB, ap.num_heads, ap.head_dim, N).transpose(-2, -1)
            k = k.view(TB, ap.num_heads, ap.head_dim, N).transpose(-2, -1)
            v = v.view(TB, ap.num_heads, ap.head_dim, N).transpose(-2, -1)

            # Linear attention: K^T @ V → Q @ result * scale
            kv_prod = k.transpose(-2, -1) @ v  # (TB, heads, head_dim, head_dim)
            x = (q @ kv_prod) * ap.scale        # (TB, heads, N, head_dim)

            # Merge heads → NCHW
            x = x.transpose(-2, -1).reshape(TB, C, H, W)

            # attn_lif
            x = self._run_attn_lif(x, mem, ap)

            # Output NHWC
            x = x.permute(0, 2, 3, 1).contiguous()
            self.activations[nid] = x

        elif kv == KernelVariant.FusedTokenQKAttn and len(inputs) >= 2:
            q, k = inputs[0], inputs[1]
            C = ap.num_heads * ap.head_dim
            TB = q.shape[0]
            H, W = ap.H, ap.W
            N = H * W

            # NHWC → NCHW for multi-head reshape
            q = q.permute(0, 3, 1, 2).contiguous().view(TB, ap.num_heads, ap.head_dim, N)
            k = k.permute(0, 3, 1, 2).contiguous().view(TB, ap.num_heads, ap.head_dim, N)

            # Scalar attention: sum(Q, head_dim) → LIF → mul(attn, K)
            attn = q.sum(dim=2, keepdim=True)           # (TB, heads, 1, N)
            attn = self._run_attn_lif(attn, mem, ap)
            x = torch.mul(attn, k)                      # (TB, heads, head_dim, N)

            # Merge heads → NCHW → NHWC
            x = x.reshape(TB, C, H, W)
            x = x.permute(0, 2, 3, 1).contiguous()
            self.activations[nid] = x

        elif kv == KernelVariant.FusedDSSAAttn and len(inputs) >= 2:
            # Identify y_kv (2C channels) vs x_query (C channels) by NHWC last dim
            C = ap.num_heads * ap.head_dim
            if inputs[0].shape[-1] == 2 * C:
                y_kv, x_q = inputs[0], inputs[1]
            elif len(inputs) > 1 and inputs[1].shape[-1] == 2 * C:
                y_kv, x_q = inputs[1], inputs[0]
            else:
                y_kv, x_q = inputs[0], inputs[1]

            C = ap.num_heads * ap.head_dim
            H_in, W_in = ap.H, ap.W

            # NHWC → NCHW
            y = y_kv.permute(0, 3, 1, 2).contiguous()  # (TB, 2C, h', w')
            x = x_q.permute(0, 3, 1, 2).contiguous()   # (TB, C, H, W)

            h_out, w_out = y.shape[2], y.shape[3]
            spatial = h_out * w_out
            spatial_q = H_in * W_in
            hd = ap.head_dim

            # Head split for K/V
            y = y.view(-1, ap.num_heads, 2 * hd, spatial)
            y1 = y[:, :, :hd, :]           # keys
            y2 = y[:, :, hd:2 * hd, :]     # values

            # Query from input
            xq = x.view(-1, ap.num_heads, hd, spatial_q)

            # Attention: K^T @ Q * scale1
            attn = torch.matmul(y1.transpose(-2, -1), xq)

            # Apply scale1/scale2 tensors (stored in weights)
            s1_name = node.extra_attrs.get("scale1_name")
            s2_name = node.extra_attrs.get("scale2_name")
            scale1 = self._get_dssa_scale(s1_name)
            scale2 = self._get_dssa_scale(s2_name)
            attn = attn * scale1

            # attn_lif
            attn = self._run_attn_lif(attn, mem, ap)

            # Output: V @ attn * scale2
            out = torch.matmul(y2, attn) * scale2

            # Reshape back to 4D NCHW → NHWC
            out = out.reshape(-1, C, H_in, W_in)
            out = out.permute(0, 2, 3, 1).contiguous()
            self.activations[nid] = out

    def _run_attn_lif(self, x, mem, ap):
        """Run LIF neuron on attention output using CUDA extension or fallback."""
        if self._ext_if is not None and mem is not None:
            x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
            recip_tau = 1.0 / ap.attn_lif_tau if ap.attn_lif_tau > 0 else 0.5
            result = self._ext_if.lif_neuron(x_flat, mem, ap.attn_lif_v_threshold,
                                              recip_tau)
            return result.reshape(x.shape)
        # Fallback: simple threshold
        return (x >= ap.attn_lif_v_threshold).to(x.dtype)

    def _get_dssa_scale(self, name):
        """Get DSSA scale tensor from weights, converting to CUDA FP16."""
        if name is None:
            return 1.0
        w = self.ir.weights.get(name)
        if w is None:
            return 1.0
        if isinstance(w, torch.Tensor):
            return w.half().cuda()
        import numpy as np
        return torch.from_numpy(np.array(w)).half().cuda()
