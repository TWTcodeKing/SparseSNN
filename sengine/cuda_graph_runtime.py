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
        """Capture the inference path into a CUDA Graph."""
        # Ensure IF extension is loaded
        if self._ext_if is None:
            for nid in self.schedule:
                node = self.ir.nodes[nid]
                if node.assigned_kernel in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
                    from sengine.build.tilelang_compiler import get_cuda_if
                self._ext_if = get_cuda_if()
                break

        # Dry run to populate all activation buffers
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
        assert self.graph is not None, "Call capture_graph() first"

        # Reset membranes
        for mem in self.membranes.values():
            mem.zero_()

        for _ in range(warmup):
            self.graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            self.graph.replay()
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
                if cp.kernel_h == 1 and cp.kernel_w == 1:
                    # 1x1 Conv: (C_out, C_in, 1, 1) → (C_in, C_out)
                    w = torch.from_numpy(w_np.astype(np.float16)).cuda()
                    self.weights_1x1[nid] = w.reshape(cp.out_channels, cp.in_channels).t().contiguous()
                else:
                    # Regular 3x3: (C_out, C_in, K, K) → (K, K, C_in, C_out)
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

    def _allocate_membranes(self):
        """Allocate FP32 membrane state tensors for all neuron nodes."""
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if node.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
                continue
            if not node.output_shapes:
                continue

            shape = node.output_shapes[0]
            if len(shape) == 4:
                N, C, H, W = shape
                B = N // self.T
                # 2D membrane: (B*H*W, C) for temporal-safe CUDA IF kernel
                self.membranes[nid] = torch.zeros(B * H * W, C,
                                                  dtype=torch.float32, device='cuda')
            elif len(shape) == 2:
                M, N = shape
                spatial = M // self.T
                self.membranes[nid] = torch.zeros(spatial, N,
                                                  dtype=torch.float32, device='cuda')

    def _allocate_activations(self):
        """Pre-allocate output buffers for all nodes in the schedule."""
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if not node.output_shapes:
                continue

            shape = node.output_shapes[0]
            if len(shape) == 4:
                N, C, H, W = shape
                # NHWC buffer
                self.activations[nid] = torch.zeros(N, H, W, C,
                                                    dtype=torch.float16, device='cuda')
            elif len(shape) == 3:
                self.activations[nid] = torch.zeros(*shape,
                                                    dtype=torch.float16, device='cuda')
            elif len(shape) == 2:
                self.activations[nid] = torch.zeros(*shape,
                                                    dtype=torch.float16, device='cuda')
            elif len(shape) == 1:
                self.activations[nid] = torch.zeros(*shape,
                                                    dtype=torch.float16, device='cuda')

        # Fix neuron buffers: derive shape from predecessor (neuron output = input)
        # This ensures correct shapes even after .sengine deserialization
        for nid in self.ir.topo_order:
            node = self.ir.nodes[nid]
            if node.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
                continue
            preds = self.ir.predecessors(nid)
            if preds and preds[0] in self.activations:
                self.activations[nid] = torch.zeros_like(self.activations[preds[0]])

        # Identify graph output: last node in schedule
        if self.schedule:
            self._graph_output_nid = self.schedule[-1]

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
        for nid in self.schedule:
            self._dispatch_node(nid)

    def _dispatch_node(self, nid: int):
        """Dispatch a single node based on its kernel variant."""
        node = self.ir.nodes[nid]
        kv = node.assigned_kernel

        # Gather input activations
        inputs = []
        for pred_nid in self.ir.predecessors(nid):
            if pred_nid in self.activations:
                inputs.append(self.activations[pred_nid])

        if not inputs and nid in self.activations:
            inputs = [self.activations[nid]]

        if not inputs:
            return

        x = inputs[0]
        out_buf = self.activations.get(nid)

        if kv == KernelVariant.TileLangConvBN:
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

        elif kv == KernelVariant.CuDNNConv:
            # Fallback: cuDNN Conv+BN via PyTorch (for stem or unaligned layers)
            w = self.weights.get(nid)
            sc = self.bn_scales.get(nid)
            bi = self.bn_biases.get(nid)
            cp = node.conv_params
            if w is not None and cp:
                # NHWC input → NCHW for torch conv
                x_nchw = x.permute(0, 3, 1, 2).contiguous().float()
                w_nchw = w.permute(3, 2, 0, 1).contiguous().float()  # (K,K,Cin,Cout)→(Cout,Cin,K,K)
                conv_out = F.conv2d(x_nchw, w_nchw, stride=cp.stride_h, padding=cp.pad_h,
                                    dilation=cp.dilation_h, groups=cp.groups)
                # Apply BN scale + bias in FP32
                if sc is not None and bi is not None:
                    conv_out = conv_out * sc.view(1, -1, 1, 1) + bi.view(1, -1, 1, 1)
                # Back to NHWC FP16
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

        elif kv == KernelVariant.TileLangMatMul:
            if len(inputs) >= 2:
                result = torch.matmul(inputs[0], inputs[1])
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
                # NHWC → NCHW for F.max_pool2d
                x_nchw = x.permute(0, 3, 1, 2)
                result_nchw = F.max_pool2d(x_nchw, ks, stride, pad)
                result = result_nchw.permute(0, 2, 3, 1).contiguous()
                if out_buf is not None:
                    out_buf.copy_(result)
                else:
                    self.activations[nid] = result
            elif node.op_type == OpType.GlobalAvgPool:
                # NHWC: (N, H, W, C) → mean over H,W → (N, 1, 1, C)
                result = x.mean(dim=(1, 2), keepdim=True)
                self.activations[nid] = result

        elif kv == KernelVariant.TemporalMean:
            # (TB, ...) → reshape (T, B, ...) → mean over T → (B, ...)
            shape = x.shape
            T_shape = (self.T, self.B) + shape[1:]
            result = x.reshape(T_shape).mean(dim=0)
            self.activations[nid] = result

        elif kv == KernelVariant.CuBLASGemm:
            w = self.weights.get(nid)
            if w is not None:
                # x: (B, in_features), w: (out_features, in_features)
                result = F.linear(x, w)
                self.activations[nid] = result

        elif kv == KernelVariant.TileRepeat:
            pass

        elif kv == KernelVariant.ZeroCost:
            if node.op_type == OpType.Transpose:
                perm = node.extra_attrs.get("perm")
                if perm and inputs:
                    t = inputs[0]
                    if len(perm) == t.ndim:
                        self.activations[nid] = t.permute(*perm).contiguous()
                    elif len(perm) > t.ndim:
                        # 5D perm on 4D tensor: collapse first dims
                        # e.g. perm=[0,1,3,2,4] on (TB,H,N,C) → try (0,2,1,3)
                        offset = len(perm) - t.ndim
                        adj = [p - offset for p in perm[offset:]]
                        self.activations[nid] = t.permute(*adj).contiguous()
                    else:
                        self.activations[nid] = t
                elif inputs:
                    self.activations[nid] = inputs[0]
            elif node.op_type == OpType.Reshape:
                target = node.extra_attrs.get("target_shape")
                if target and inputs:
                    # Handle -1 and 0 dimensions
                    shape = list(target)
                    for i, d in enumerate(shape):
                        if d == 0 and i < inputs[0].ndim:
                            shape[i] = inputs[0].shape[i]
                    self.activations[nid] = inputs[0].reshape(*shape)
                elif inputs:
                    self.activations[nid] = inputs[0]
            elif inputs:
                self.activations[nid] = inputs[0]
