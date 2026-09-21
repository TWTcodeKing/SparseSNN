"""Graph optimization passes for sengine_cpu.

Passes (applied in order):
1. BN folding:       Absorb BatchNorm into preceding Conv/Gemm
2. Dead node elim:   Remove dangling Identity nodes
3. Fusion detection: Find Conv→(transparent)→Neuron patterns
4. Shape propagation: Compute spatial dims for every node
5. Bound classification + CPU kernel assignment
6. Adaptive kernel selection: interleaved vs decomposed per fusion group
"""

from __future__ import annotations
import numpy as np
from sengine_cpu.logger import log

from sengine_cpu.ir import (
    OpType, NeuronType, CPUKernelVariant, DataLayout, BoundType,
    ConvParams, NeuronParams, Node, Edge, FusionGroup, EngineIR,
)


def _native_conv_backend() -> bool:
    """True when conv layers should use the native C kernel instead of TVM.

    SENGINE_CPU_BACKEND=native|tvm forces a choice; the default ('auto') uses
    TVM only when its isolated interpreter exists (see sengine_cpu/tvm_env.py).
    """
    import os
    from sengine_cpu.tvm_env import tvm_available, require_tvm_python
    mode = os.environ.get("SENGINE_CPU_BACKEND", "auto").lower()
    if mode == "native":
        return True
    if mode == "tvm":
        require_tvm_python()
        return False
    return not tvm_available()


def optimize_ir(ir: EngineIR, batch_size: int = 1, T: int = 4) -> EngineIR:
    """Apply all optimization passes. Mutates ir in place."""
    log.info("Running optimization passes on %d nodes", len(ir.nodes))

    fold_batchnorm(ir)
    eliminate_dead_nodes(ir)
    ir.build_edges()
    ir.compute_topo_order()

    detect_fusion_groups(ir)
    propagate_shapes(ir, batch_size=batch_size)
    classify_bound_and_assign_cpu_kernels(ir, batch_size=batch_size)
    select_interleaved_vs_decomposed(ir, T=T, batch_size=batch_size)

    log.info("Done: %d nodes, %d fusion groups",
             len(ir.nodes), len(ir.fusion_groups))
    return ir


# ============================================================
# Pass 0: Dead Node Elimination
# ============================================================

def eliminate_dead_nodes(ir: EngineIR):
    """Remove Identity nodes whose outputs have no consumers."""
    consumers: dict[str, set[int]] = {}
    for node in ir.nodes.values():
        for in_name in node.input_names:
            consumers.setdefault(in_name, set()).add(node.id)

    removed_total = 0
    changed = True
    while changed:
        changed = False
        to_remove = []
        for nid, node in ir.nodes.items():
            if node.op_type != OpType.Identity:
                continue
            has_consumer = False
            for out_name in node.output_names:
                for cid in consumers.get(out_name, set()):
                    if cid in ir.nodes and cid != nid:
                        has_consumer = True; break
                if has_consumer: break
            if not has_consumer:
                to_remove.append(nid)

        for nid in to_remove:
            node = ir.nodes[nid]
            for in_name in node.input_names:
                if in_name in consumers:
                    consumers[in_name].discard(nid)
            del ir.nodes[nid]
            changed = True
            removed_total += 1

    if removed_total:
        ir._topo_order = []
    log.info("DCE: eliminated %d dead Identity nodes", removed_total)


# ============================================================
# Pass 1: BatchNorm Folding
# ============================================================

def fold_batchnorm(ir: EngineIR):
    """Fold BN into preceding Conv/Gemm.

    Instead of modifying weights (which loses precision in FP16),
    we store bn_scale and bn_bias on the Conv node for the fused
    kernel epilogue: output = GEMM(x, w) * bn_scale + bn_bias
    """
    bn_nodes = [n for n in ir.nodes.values() if n.op_type == OpType.BatchNorm]
    removed = 0

    for bn in bn_nodes:
        preds = ir.predecessors(bn.id)
        if not preds:
            continue
        conv = ir.nodes.get(preds[0])
        if conv is None or conv.op_type not in (OpType.Conv2d, OpType.Gemm):
            continue

        scale_name = bn.extra_attrs.get("bn_scale_name", "")
        bias_name = bn.extra_attrs.get("bn_bias_name", "")
        mean_name = bn.extra_attrs.get("bn_mean_name", "")
        var_name = bn.extra_attrs.get("bn_var_name", "")
        eps = bn.extra_attrs.get("epsilon", 1e-5)

        if not all(name in ir.weights for name in [scale_name, bias_name, mean_name, var_name]):
            continue

        gamma = ir.weights[scale_name].astype(np.float64)
        beta = ir.weights[bias_name].astype(np.float64)
        mean = ir.weights[mean_name].astype(np.float64)
        var = ir.weights[var_name].astype(np.float64)

        inv_std = 1.0 / np.sqrt(var + eps)
        scale = gamma * inv_std

        if conv.bias_info and conv.bias_info.name in ir.weights:
            b = ir.weights[conv.bias_info.name].astype(np.float64)
        else:
            b = np.zeros_like(mean)
        bn_bias = scale * (b - mean) + beta

        conv.bn_scale = scale.astype(np.float32).tolist()
        conv.bn_bias = bn_bias.astype(np.float32).tolist()

        ir.remove_node(bn.id)
        removed += 1

    if removed:
        ir._topo_order = []
    log.info("BN fold: folded %d BatchNorm into Conv/Gemm", removed)


# ============================================================
# Pass 2: Fusion Group Detection
# ============================================================

_TRANSPARENT_OPS = {OpType.Reshape, OpType.Transpose, OpType.Identity, OpType.Flatten}


def _find_neuron_through_transparent(ir: EngineIR, start_nid: int, max_depth: int = 3):
    """Follow single-consumer transparent ops to find a neuron node."""
    cur = start_nid
    for _ in range(max_depth):
        succs = ir.successors(cur)
        if len(succs) != 1:
            return None
        s = ir.nodes.get(succs[0])
        if s is None:
            return None
        if s.op_type in (OpType.IF, OpType.LIF, OpType.MS, OpType.ILIF):
            return s
        if s.op_type in _TRANSPARENT_OPS:
            cur = s.id
            continue
        return None
    return None


def detect_fusion_groups(ir: EngineIR):
    """Detect Conv→(Reshape/Transpose)→Neuron patterns."""
    ir.fusion_groups.clear()
    group_id = 0

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None or node.op_type != OpType.Conv2d:
            continue
        if node.fusion_group_id >= 0:
            continue

        neuron = _find_neuron_through_transparent(ir, nid)
        if neuron is None or neuron.fusion_group_id >= 0:
            continue

        fg = FusionGroup(
            group_id=group_id,
            conv_node_id=nid,
            neuron_node_id=neuron.id,
            bn_scale=node.bn_scale,
            bn_bias=node.bn_bias,
        )
        ir.fusion_groups.append(fg)
        node.fusion_group_id = group_id
        neuron.fusion_group_id = group_id
        group_id += 1

    log.info("Fusion: detected %d Conv→Neuron fusion groups", len(ir.fusion_groups))


# ============================================================
# Pass 3: Shape Propagation
# ============================================================

def propagate_shapes(ir: EngineIR, batch_size: int = 1):
    """Propagate spatial dims through the graph from model input."""
    shape_map: dict[str, tuple] = {}
    model_input = ir.model_input_shape
    if model_input and len(model_input) >= 1:
        onnx_B = model_input[0]
        if onnx_B > 0 and batch_size != onnx_B:
            model_input = (batch_size,) + model_input[1:]

    produced = set()
    consumed = set()
    for node in ir.nodes.values():
        produced.update(node.output_names)
        consumed.update(node.input_names)
    graph_inputs = consumed - produced

    if model_input:
        for tensor_name in graph_inputs:
            if tensor_name not in ir.weights:
                shape_map[tensor_name] = model_input

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node and node.op_type == OpType.Transpose:
            for in_name in node.input_names:
                if in_name in ir.weights and in_name not in shape_map:
                    shape_map[in_name] = tuple(int(d) for d in ir.weights[in_name].shape)

    for nid in ir.topo_order:
        node = ir.nodes[nid]
        in_shapes = [shape_map[n] for n in node.input_names if n in shape_map]
        if not in_shapes and node.input_shapes:
            in_shapes = list(node.input_shapes)
        if in_shapes:
            node.input_shapes = in_shapes

        if (node.op_type == OpType.Identity
                and node.extra_attrs.get("original_op") == "Split"
                and len(node.output_names) > 1 and in_shapes):
            axis = node.extra_attrs.get("axis", 0)
            num_outputs = len(node.output_names)
            inp_shape = list(in_shapes[0])
            if 0 <= axis < len(inp_shape):
                split_dim = inp_shape[axis] // num_outputs
                for out_name in node.output_names:
                    s = list(inp_shape); s[axis] = split_dim
                    shape_map[out_name] = tuple(s)
                node.output_shapes = [tuple(inp_shape[:axis] + [split_dim] + inp_shape[axis+1:])]
        else:
            out_shape = _compute_output_shape(node, in_shapes, ir)
            if out_shape:
                node.output_shapes = [out_shape]
                for out_name in node.output_names:
                    shape_map[out_name] = out_shape


def _compute_output_shape(node: Node, in_shapes: list[tuple], ir: EngineIR) -> tuple:
    if not in_shapes:
        return ()
    inp = in_shapes[0]
    T = ir.T

    if node.op_type == OpType.Conv2d and node.conv_params:
        cp = node.conv_params
        if len(inp) != 4: return ()
        N, C, H, W = inp
        OH = (H + 2*cp.pad_h - cp.dilation_h*(cp.kernel_h-1) - 1) // cp.stride_h + 1
        OW = (W + 2*cp.pad_w - cp.dilation_w*(cp.kernel_w-1) - 1) // cp.stride_w + 1
        return (N, cp.out_channels, OH, OW)

    if node.op_type == OpType.MaxPool and node.pool_params:
        pp = node.pool_params
        if len(inp) < 4: return inp
        N, C, H, W = inp
        kh = pp.get("kernel_h", pp.get("kernel_size", 3))
        kw = pp.get("kernel_w", pp.get("kernel_size", 3))
        sh = pp.get("stride_h", pp.get("stride", 2))
        sw = pp.get("stride_w", pp.get("stride", 2))
        ph = pp.get("pad_h", pp.get("padding", 1))
        pw = pp.get("pad_w", pp.get("padding", 1))
        # Handle list-type kernel_shape/strides/pads from ONNX
        if isinstance(kh, list): kh = kh[0]
        if isinstance(kw, list): kw = kw[-1] if len(kw) > 1 else kw[0]
        if isinstance(sh, list): sh = sh[0]
        if isinstance(sw, list): sw = sw[-1] if len(sw) > 1 else sw[0]
        if isinstance(ph, list): ph = ph[0]
        if isinstance(pw, list): pw = pw[-1] if len(pw) > 1 else pw[0]
        # ONNX pool_params come as kernel_shape, strides, pads
        if "kernel_shape" in pp:
            ks = pp["kernel_shape"]
            kh = ks[0]; kw = ks[1] if len(ks) > 1 else ks[0]
        if "strides" in pp:
            st = pp["strides"]
            sh = st[0]; sw = st[1] if len(st) > 1 else st[0]
        if "pads" in pp:
            pa = pp["pads"]
            ph = pa[0]; pw = pa[1] if len(pa) > 1 else pa[0]
        OH = (H + 2*ph - kh) // sh + 1
        OW = (W + 2*pw - kw) // sw + 1
        return (N, C, OH, OW)

    if node.op_type == OpType.GlobalAvgPool:
        return (inp[0], inp[1], 1, 1) if len(inp) >= 4 else inp

    if node.op_type in (OpType.IF, OpType.LIF, OpType.MS, OpType.ILIF,
                        OpType.Add, OpType.Mul, OpType.Sub, OpType.Scale,
                        OpType.Softmax):
        return inp

    if node.op_type == OpType.Resize:
        scales = node.extra_attrs.get("scales")
        sizes = node.extra_attrs.get("sizes")
        if sizes and len(sizes) == 4: return tuple(sizes)
        if scales and len(scales) == 4 and len(inp) == 4:
            N, C, H, W = inp
            return (N, C, int(H*scales[2]), int(W*scales[3]))
        return inp

    if node.op_type == OpType.Slice:
        starts = node.extra_attrs.get("starts", [])
        ends = node.extra_attrs.get("ends", [])
        axes = node.extra_attrs.get("axes", list(range(len(starts))))
        out = list(inp)
        for a, s, e in zip(axes, starts, ends):
            if 0 <= a < len(out):
                dim = out[a]
                e_c = min(e, dim) if e >= 0 else max(0, dim + e)
                s_c = max(0, s) if s >= 0 else max(0, dim + s)
                out[a] = e_c - s_c
        return tuple(out)

    if node.op_type == OpType.Concat:
        axis = node.extra_attrs.get("axis", 1)
        if len(in_shapes) >= 2 and all(len(s) == len(in_shapes[0]) for s in in_shapes):
            out = list(in_shapes[0])
            ax = axis if axis >= 0 else len(out) + axis
            if 0 <= ax < len(out):
                out[ax] = sum(s[ax] for s in in_shapes)
            return tuple(out)
        return inp

    if node.op_type == OpType.Flatten:
        if len(inp) >= 4:
            return (inp[0], inp[1]*inp[2]*inp[3])
        return inp

    if node.op_type == OpType.Gemm and node.gemm_params:
        out_features = node.gemm_params.get("N", 0)
        if out_features == 0 and node.weight_info and node.weight_info.shape:
            ws = node.weight_info.shape
            transB = node.gemm_params.get("transB", 0)
            out_features = ws[0] if transB else (ws[1] if len(ws) >= 2 else 0)
        return (inp[0], out_features) if len(inp) >= 2 else inp

    if node.op_type == OpType.Linear:
        if node.gemm_params:
            N_out = node.gemm_params.get("N", 0)
            if N_out > 0: return inp[:-1] + (N_out,)
        if node.weight_info and node.weight_info.shape:
            return inp[:-1] + (node.weight_info.shape[-1],)
        return inp

    if node.op_type == OpType.MatMul:
        if len(in_shapes) >= 2:
            a, b = in_shapes[0], in_shapes[1]
            if len(a) >= 2 and len(b) >= 2:
                return a[:-1] + (b[-1],)
        return inp

    if node.op_type == OpType.Tile:
        return (inp[0] * T,) + inp[1:] if inp else inp

    if node.op_type == OpType.TemporalMean:
        if len(inp) == 5:
            # (T, B, C, H, W) -> (B, C, H, W): reduce over the leading T axis
            return inp[1:]
        if len(inp) >= 1:
            B = inp[0] // T if T > 0 else inp[0]
            return (max(B, 1),) + inp[1:]
        return inp

    if node.op_type == OpType.Transpose:
        perm = node.extra_attrs.get("perm")
        if perm and inp and len(perm) == len(inp):
            return tuple(inp[p] for p in perm)
        if node.output_shapes: return node.output_shapes[0]
        return inp

    if node.op_type == OpType.Reshape:
        target = node.extra_attrs.get("target_shape")
        if target:
            result = list(target)
            total = 1
            for d in inp: total *= d
            known = 1; neg_idx = -1
            for i, d in enumerate(result):
                if d == -1: neg_idx = i
                elif d == 0 and i < len(inp): result[i] = inp[i]; known *= inp[i]
                else: known *= d
            if neg_idx >= 0 and known > 0:
                result[neg_idx] = total // known
            # The ONNX constant was traced at export batch size (1): if the
            # element count no longer matches the runtime input, rescale the
            # batch axis (dim 1 of a (T, B, ...) split, else dim 0).
            prod = 1
            for d in result:
                prod *= d
            if total > 0 and prod != total:
                for idx in ((1, 0) if len(result) >= 5 else (0, 1)):
                    if idx < len(result) and result[idx] > 0 and prod % result[idx] == 0:
                        rest = prod // result[idx]
                        if rest > 0 and total % rest == 0:
                            result[idx] = total // rest
                            break
            return tuple(result)
        if node.output_shapes: return node.output_shapes[0]
        return inp

    if node.op_type == OpType.Identity:
        if node.output_shapes: return node.output_shapes[0]
        return inp

    if node.op_type == OpType.FusedAttention:
        ap = node.attention_params
        if ap and ap.variant == "dssa" and len(in_shapes) >= 2:
            return in_shapes[1]
        return inp

    return inp


# ============================================================
# Pass 4: Bound Classification + CPU Kernel Assignment
# ============================================================

_NEURON_OPS = {OpType.IF, OpType.LIF, OpType.MS, OpType.ILIF}
_COMPUTE_OPS = {OpType.Conv2d, OpType.Linear, OpType.MatMul, OpType.Gemm}
_MEMORY_OPS = {OpType.Add, OpType.Mul, OpType.Sub, OpType.MaxPool,
               OpType.GlobalAvgPool, OpType.TemporalMean, OpType.Scale,
               OpType.Resize, OpType.Concat, OpType.Softmax, OpType.Slice}
_ZERO_OPS = {OpType.Flatten, OpType.Reshape, OpType.Transpose, OpType.Identity,
             OpType.Tile, OpType.InputConvert, OpType.OutputConvert,
             OpType.ReduceMean}


def classify_bound_and_assign_cpu_kernels(ir: EngineIR, batch_size: int = 1):
    """Classify each node as COMPUTE/MEMORY/ZERO and assign CPU kernel variants."""
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue

        # --- Bound classification ---
        if node.op_type in _COMPUTE_OPS:
            node.bound_type = BoundType.COMPUTE
        elif node.op_type in _NEURON_OPS:
            node.bound_type = BoundType.MEMORY
        elif node.op_type in _MEMORY_OPS:
            node.bound_type = BoundType.MEMORY
        elif node.op_type in _ZERO_OPS:
            node.bound_type = BoundType.ZERO
        elif node.op_type == OpType.FusedAttention:
            node.bound_type = BoundType.COMPUTE
        elif node.op_type == OpType.BatchNorm:
            node.bound_type = BoundType.MEMORY
        else:
            node.bound_type = BoundType.ZERO

        # --- CPU kernel assignment ---
        if node.op_type == OpType.Conv2d and _native_conv_backend():
            cp = node.conv_params
            if node.fusion_group_id >= 0:
                fg = ir.fusion_groups[node.fusion_group_id]
                neuron_node = ir.nodes.get(fg.neuron_node_id)
                nt = neuron_node.neuron_params.neuron_type if neuron_node and neuron_node.neuron_params else NeuronType.IF
                node.assigned_kernel = (CPUKernelVariant.NativeConvBNIF if nt == NeuronType.IF
                                        else CPUKernelVariant.NativeConvBNLIF)
            else:
                node.assigned_kernel = CPUKernelVariant.NativeConvBN

        elif node.op_type == OpType.Conv2d:
            cp = node.conv_params
            if node.fusion_group_id >= 0:
                # Fused with neuron — determine variant
                fg = ir.fusion_groups[node.fusion_group_id]
                neuron_node = ir.nodes.get(fg.neuron_node_id)
                nt = neuron_node.neuron_params.neuron_type if neuron_node and neuron_node.neuron_params else NeuronType.IF
                if cp and cp.kernel_h == 1 and cp.kernel_w == 1:
                    node.assigned_kernel = (CPUKernelVariant.TVMConv1x1BNIF
                                           if nt == NeuronType.IF
                                           else CPUKernelVariant.TVMConv1x1BNLIF)
                else:
                    node.assigned_kernel = (CPUKernelVariant.TVMConvBNIF
                                           if nt == NeuronType.IF
                                           else CPUKernelVariant.TVMConvBNLIF)
            else:
                # Decomposed conv
                if cp and cp.kernel_h == 1 and cp.kernel_w == 1:
                    node.assigned_kernel = CPUKernelVariant.TVMConv1x1BN
                else:
                    node.assigned_kernel = CPUKernelVariant.TVMConvBN

        elif node.op_type == OpType.Linear:
            node.assigned_kernel = CPUKernelVariant.TVMLinearBN

        elif node.op_type == OpType.MatMul:
            node.assigned_kernel = CPUKernelVariant.TVMMatMul

        elif node.op_type == OpType.Gemm:
            node.assigned_kernel = CPUKernelVariant.NativeGemm

        elif node.op_type == OpType.IF:
            if node.fusion_group_id >= 0:
                node.assigned_kernel = CPUKernelVariant.Skip
                node.bound_type = BoundType.ZERO
            else:
                node.assigned_kernel = CPUKernelVariant.NativeIF

        elif node.op_type == OpType.LIF:
            if node.fusion_group_id >= 0:
                node.assigned_kernel = CPUKernelVariant.Skip
                node.bound_type = BoundType.ZERO
            else:
                node.assigned_kernel = CPUKernelVariant.NativeLIF

        elif node.op_type in (OpType.MS, OpType.ILIF):
            if node.fusion_group_id >= 0:
                node.assigned_kernel = CPUKernelVariant.Skip
                node.bound_type = BoundType.ZERO
            else:
                node.assigned_kernel = CPUKernelVariant.NativeLIF

        elif node.op_type == OpType.Add:
            node.assigned_kernel = CPUKernelVariant.NativeAdd

        elif node.op_type == OpType.MaxPool:
            node.assigned_kernel = CPUKernelVariant.NativeMaxPool

        elif node.op_type == OpType.GlobalAvgPool:
            node.assigned_kernel = CPUKernelVariant.NativeGlobalAvgPool

        elif node.op_type == OpType.TemporalMean:
            node.assigned_kernel = CPUKernelVariant.NativeTemporalMean

        elif node.op_type == OpType.Softmax:
            node.assigned_kernel = CPUKernelVariant.NativeSoftmax

        elif node.op_type == OpType.Resize:
            node.assigned_kernel = CPUKernelVariant.NativeResize

        elif node.op_type == OpType.Concat:
            node.assigned_kernel = CPUKernelVariant.NativeConcat

        elif node.op_type in _ZERO_OPS:
            node.assigned_kernel = CPUKernelVariant.ZeroCost

        # Layout: NHWC for all spatial ops on CPU
        if node.op_type in (OpType.Conv2d, OpType.MaxPool, OpType.GlobalAvgPool,
                            OpType.Resize, OpType.Concat):
            node.layout = DataLayout.NHWC
        elif node.op_type in (OpType.Linear, OpType.MatMul, OpType.Gemm):
            node.layout = DataLayout.ND

    log.info("Kernel assignment: %d nodes classified",
             sum(1 for n in ir.nodes.values()
                 if n.assigned_kernel != CPUKernelVariant.ZeroCost))


# ============================================================
# Pass 5: Adaptive Kernel Selection (Interleaved vs Decomposed)
# ============================================================
#
# The fused (interleaved) Conv+BN+LIF micro-kernel keeps GEMM output
# in registers and fuses the neuron epilogue. This wins when:
#   - M (spatial positions) is large → amortizes weight loading
#   - K (input channels × kernel area) is small → fits in cache
#
# For large K (>~512) with small M, BLAS GEMM + separate neuron is
# faster because BLAS has proper K-blocking and weight packing.
#
# This pass uses a simple analytical cost model (no runtime profiling)
# to decide per fusion group. The model is calibrated from the VGG-9
# benchmark where we measured the crossover point.

# Crossover heuristic: if K/M > threshold, BLAS wins.
# From benchmarking (VGG-9 on 32×32, T=4):
#   M=1024, K=27:   fused wins  (K/M = 0.026)
#   M=256,  K=576:  fused wins  (K/M = 2.25)   — borderline but fused ok
#   M=64,   K=1152: BLAS wins   (K/M = 18.0)
#   M=64,   K=2304: BLAS wins   (K/M = 36.0)
#   M=16,   K=2304: BLAS wins   (K/M = 144.0)
#   M=16,   K=4608: BLAS wins   (K/M = 288.0)
#
# Heuristic: use fused when K ≤ FUSED_K_THRESHOLD and K/M ≤ KM_RATIO_THRESHOLD
# This keeps most convolutions fused (especially 1×1 convs and early 3×3 convs)
# while falling back to BLAS for the deep 3×3 layers with huge im2col expansion.

FUSED_K_THRESHOLD = 576       # K ≤ this → always fused (covers most 1×1 and early 3×3)
KM_RATIO_THRESHOLD = 8.0      # K/M ≤ this → fused (covers medium-sized layers)
FUSED_M_THRESHOLD = 256       # M ≥ this → fused IF FLOP budget allows
FUSED_FLOP_LIMIT = 50_000_000 # 50 MFLOP: above this, BLAS wins regardless of M


def select_interleaved_vs_decomposed(ir: EngineIR, T: int = 4,
                                      batch_size: int = 1):
    """Decide per fusion group: interleaved (fused) vs decomposed (BLAS+native).

    For groups where decomposed wins, reassign:
      Conv node:   TVMConvBN{IF,LIF} → TVMConvBN / TVMConv1x1BN  (GEMM only)
      Neuron node: Skip → NativeIF / NativeLIF                   (separate T-loop)
      FusionGroup: removed (conv.fusion_group_id = -1)
    """
    if not ir.fusion_groups:
        return
    if _native_conv_backend():
        log.info("Native conv backend: all %d Conv->neuron groups stay fused", len(ir.fusion_groups))
        return

    n_fused = 0
    n_decomposed = 0

    for fg in ir.fusion_groups:
        conv_node = ir.nodes.get(fg.conv_node_id)
        neuron_node = ir.nodes.get(fg.neuron_node_id)
        if conv_node is None or neuron_node is None:
            continue

        cp = conv_node.conv_params
        if not cp or not conv_node.output_shapes:
            n_fused += 1
            continue

        # Compute GEMM dimensions
        out_shape = conv_node.output_shapes[0]
        TB = out_shape[0] if len(out_shape) >= 1 else T * batch_size
        OH = out_shape[2] if len(out_shape) >= 3 else 1
        OW = out_shape[3] if len(out_shape) >= 4 else 1
        M = (TB // T) * OH * OW      # spatial positions per timestep
        K = cp.in_channels * cp.kernel_h * cp.kernel_w   # GEMM K dim
        F = cp.out_channels            # GEMM N dim

        # Decision logic
        flops = 2 * M * K * F   # GEMM FLOPs per timestep
        use_fused = True
        reason = "default"

        if flops > FUSED_FLOP_LIMIT:
            # Large GEMM: BLAS dominates regardless of shape ratios
            use_fused = False
            reason = f"FLOP={flops/1e6:.0f}M>{FUSED_FLOP_LIMIT/1e6:.0f}M"
        elif M >= FUSED_M_THRESHOLD:
            use_fused = True
            reason = f"M={M}≥{FUSED_M_THRESHOLD}"
        elif K <= FUSED_K_THRESHOLD:
            use_fused = True
            reason = f"K={K}≤{FUSED_K_THRESHOLD}"
        elif K / max(M, 1) <= KM_RATIO_THRESHOLD:
            use_fused = True
            reason = f"K/M={K/max(M,1):.1f}≤{KM_RATIO_THRESHOLD}"
        else:
            use_fused = False
            reason = f"K/M={K/max(M,1):.1f}>{KM_RATIO_THRESHOLD}"

        if use_fused:
            n_fused += 1
            log.debug("FG%d Conv(%d→%d) M=%d K=%d: INTERLEAVED (%s)",
                       fg.group_id, cp.in_channels, cp.out_channels, M, K, reason)
        else:
            n_decomposed += 1
            log.info("FG%d Conv(%d→%d) M=%d K=%d: DECOMPOSED (%s)",
                      fg.group_id, cp.in_channels, cp.out_channels, M, K, reason)

            # Reassign conv: fused → decomposed GEMM
            if cp.kernel_h == 1 and cp.kernel_w == 1:
                conv_node.assigned_kernel = CPUKernelVariant.TVMConv1x1BN
            else:
                conv_node.assigned_kernel = CPUKernelVariant.TVMConvBN
            conv_node.fusion_group_id = -1

            # Reassign neuron: Skip → native standalone
            nt = neuron_node.neuron_params.neuron_type if neuron_node.neuron_params else NeuronType.IF
            if nt == NeuronType.IF:
                neuron_node.assigned_kernel = CPUKernelVariant.NativeIF
            else:
                neuron_node.assigned_kernel = CPUKernelVariant.NativeLIF
            neuron_node.fusion_group_id = -1
            neuron_node.bound_type = BoundType.MEMORY

    log.info("Kernel selection: %d interleaved, %d decomposed (%.0f%% fused)",
             n_fused, n_decomposed,
             n_fused / max(n_fused + n_decomposed, 1) * 100)
