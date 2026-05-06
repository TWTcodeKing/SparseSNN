"""Graph optimization passes for SEngine.

Passes (applied in order):
1. BN folding:     Absorb BatchNorm into preceding Conv/Gemm weights
2. Fusion:         Detect Conv→(BN→)Neuron fusion groups
3. Sparsity:       Validate 2:4 pattern on Conv weights, mark eligible layers
4. Layout:         Annotate NHWC for tensor-core paths
5. Kernel assign:  Select kernel variant per node based on fusion/sparsity/layout
6. Shape propagation: Compute spatial dims (H, W) for every node from model input
7. Bound classify: Label each node as Compute/Memory/Zero, assign TileLang kernels
"""

from __future__ import annotations
import numpy as np
from sengine.logger import logger

from sengine.ir import (
    OpType, NeuronType, KernelVariant, TensorLayout, BoundType,
    Node, Edge, FusionGroup, EngineIR,
    KERNEL_CONTRACTS,
)


def optimize_ir(ir: EngineIR, tilelang: bool = False,
                batch_size: int = 1) -> EngineIR:
    """Apply all optimization passes to the IR. Mutates ir in place.

    Args:
        tilelang: If True, also run shape propagation + bound classification
                  passes that assign TileLang decomposed kernels for BA-MTTS.
        batch_size: Batch size (needed for hybrid kernel strategy selection).
    """
    n_passes = 7 if tilelang else 5
    logger.phase("OPT", "Running %d optimization passes on %d nodes", n_passes, len(ir.nodes))

    fold_batchnorm(ir)
    eliminate_dead_nodes(ir)
    ir.build_edges()
    ir.compute_topo_order()

    detect_fusion_groups(ir)
    mark_sparsity(ir)
    annotate_layout(ir)
    assign_kernels(ir)

    if tilelang:
        propagate_shapes(ir, batch_size=batch_size)
        classify_bound_and_assign_tilelang(ir, batch_size=batch_size)
        propagate_edge_layouts(ir)
        n_reformats = insert_layout_reformats(ir)
        if n_reformats > 0:
            ir.build_edges()
            ir.compute_topo_order()
            propagate_edge_layouts(ir)

    logger.phase("OPT", "Done: %d nodes, %d fusion groups, %d sparse layers",
                 len(ir.nodes), len(ir.fusion_groups),
                 sum(1 for n in ir.nodes.values() if n.sparse_weight))
    return ir


# ============================================================
# Pass 0: Dead Node Elimination
# ============================================================

def eliminate_dead_nodes(ir: EngineIR):
    """Remove Identity nodes that don't carry data between real ops.

    Identity nodes come from ONNX ops the parser doesn't handle (Slice,
    Shape, Constant, etc.). Many are dangling — they produce tensors that
    no other node consumes, or they consume initializers without feeding
    into any real computation.

    This pass removes nodes that:
    1. Have op_type == Identity AND
    2. None of their outputs are consumed by a non-Identity node
    """
    # Build consumer map: tensor_name → set of consumer node ids
    consumers: dict[str, set[int]] = {}
    for node in ir.nodes.values():
        for in_name in node.input_names:
            if in_name not in consumers:
                consumers[in_name] = set()
            consumers[in_name].add(node.id)

    # Iteratively remove dangling Identity nodes
    removed_total = 0
    changed = True
    while changed:
        changed = False
        to_remove = []
        for nid, node in ir.nodes.items():
            if node.op_type != OpType.Identity:
                continue
            # Check if any output is consumed by a non-removed node
            has_consumer = False
            for out_name in node.output_names:
                for consumer_id in consumers.get(out_name, set()):
                    if consumer_id in ir.nodes and consumer_id != nid:
                        has_consumer = True
                        break
                if has_consumer:
                    break
            if not has_consumer:
                to_remove.append(nid)

        for nid in to_remove:
            node = ir.nodes[nid]
            # Remove from consumer lists
            for in_name in node.input_names:
                if in_name in consumers:
                    consumers[in_name].discard(nid)
            del ir.nodes[nid]
            changed = True
            removed_total += 1

    if removed_total:
        ir._topo_order = []
    logger.phase("DCE", "Eliminated %d dead Identity nodes", removed_total)


# ============================================================
# Pass 1: BatchNorm Folding
# ============================================================

def fold_batchnorm(ir: EngineIR):
    """Fold BN parameters into the preceding Conv/Gemm weight.

    BN formula: y = gamma/sqrt(var+eps) * (x - mean) + beta
    Folded:     W' = gamma/sqrt(var+eps) * W
                b' = gamma/sqrt(var+eps) * (b - mean) + beta

    After folding, the BN node is replaced with Identity (removed from graph).
    The folded scale and bias are stored on the Conv node for the fused kernel epilogue.
    """
    bn_nodes = [n for n in ir.nodes.values() if n.op_type == OpType.BatchNorm]
    removed = 0

    for bn in bn_nodes:
        # Find preceding Conv node
        preds = ir.predecessors(bn.id)
        if not preds:
            continue
        conv_id = preds[0]
        conv = ir.nodes.get(conv_id)
        if conv is None or conv.op_type not in (OpType.Conv2d, OpType.Gemm):
            continue

        # Extract BN parameters
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

        # Compute BN scale: gamma / sqrt(var + eps)
        inv_std = 1.0 / np.sqrt(var + eps)
        scale = gamma * inv_std

        # DO NOT fold scale into weight — keep weight in original precision.
        # The TileLang kernel epilogue applies: output = GEMM(x, w) * bn_scale + bn_bias
        # This preserves FP32 precision in the BN computation, avoiding the
        # FP16 quantization error from baking scale into the weight.

        # Compute epilogue bias: scale * (b - mean) + beta
        if conv.bias_info and conv.bias_info.name in ir.weights:
            b = ir.weights[conv.bias_info.name].astype(np.float64)
        else:
            b = np.zeros_like(mean)
        bn_bias = scale * (b - mean) + beta

        # Store BN scale and bias on Conv node for the fused kernel epilogue
        # Kernel computes: y = gemm_out * bn_scale + bn_bias
        conv.bn_scale = scale.astype(np.float32).tolist()
        conv.bn_bias = bn_bias.astype(np.float32).tolist()

        # Remove BN node from graph
        ir.remove_node(bn.id)
        removed += 1

    if removed:
        ir._topo_order = []  # invalidate
    logger.phase("BN_FOLD", "Folded %d BatchNorm layers into Conv/Gemm weights", removed)


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
        if s.op_type in (OpType.IF, OpType.LIF, OpType.MS):
            return s
        if s.op_type in _TRANSPARENT_OPS:
            cur = s.id
            continue
        return None
    return None


def _trace_through_zerocost(ir: EngineIR, start_nid: int,
                            target_ops: set, max_depth: int = 5):
    """Follow single-consumer transparent ops to find a node of target type.

    Returns the target Node, or None.
    """
    cur = start_nid
    for _ in range(max_depth):
        succs = ir.successors(cur)
        if len(succs) != 1:
            return None
        s = ir.nodes.get(succs[0])
        if s is None:
            return None
        if s.op_type in target_ops:
            return s
        if s.op_type in _TRANSPARENT_OPS:
            cur = s.id
            continue
        return None
    return None


def _find_scale_node(ir: EngineIR, start_nid: int, max_depth: int = 3):
    """Find a Scale or broadcastable-constant Mul successor through zero-cost ops.

    Handles two patterns:
      - OpType.Scale: Mul where one operand is a scalar constant (parser-detected)
      - OpType.Mul with constant: Mul where one operand is a constant tensor
        (e.g. DSSA4D scale buffers with shape (1, heads, 1, 1))

    Returns (node, scale_value) or (None, None).
    """
    cur = start_nid
    for _ in range(max_depth):
        succs = ir.successors(cur)
        if len(succs) != 1:
            return None, None
        s = ir.nodes.get(succs[0])
        if s is None:
            return None, None
        if s.op_type == OpType.Scale:
            return s, s.extra_attrs.get("scale_value", 1.0)
        if s.op_type == OpType.Mul:
            # Check if one input is a constant (broadcastable scale).
            # DSSA4D emits per-head scale tensors like (1, heads, 1, 1) —
            # if all elements are the same value, treat as uniform scalar.
            for inp_name in s.input_names:
                if inp_name in ir.weights:
                    w = ir.weights[inp_name]
                    if w.size == 0:
                        continue
                    if w.size == 1:
                        return s, float(w.flat[0])
                    # Small constant with uniform values → scalar scale
                    if w.size <= 64 and w.min() == w.max():
                        return s, float(w.flat[0])
            return None, None
        if s.op_type in _TRANSPARENT_OPS:
            cur = s.id
            continue
        return None, None
    return None, None


def _detect_attention_matmul_patterns(ir: EngineIR, FUSED_THRESHOLD: int):
    """Detect attention matmul patterns and assign optimized kernels.

    Supported patterns (all architectures):

    SpikFormer SSA:       Q@K^T → Scale(0.125) → attn@V → ... → attn_lif
    MaxFormer SSA:        K^T@V → Q@KV → Scale(0.125) → ... → attn_lif
    SpikingResformer DSSA: Y1^T@X → Mul(scale1) → attn_lif → Y2@attn → Mul(scale2) → lif

    Detection rules per MatMul node:
    1. If successor (through zero-cost ops) is Scale/Mul-constant:
       → Absorb scale into MatMul epilogue (TileLangMatMulScale)
       → Then check if Scale's successor is a LIF → fused/decomposed
    2. If successor (through zero-cost ops) is LIF/IF directly:
       → Apply fused/decomposed threshold (no scale absorption)
    3. Otherwise: keep as plain TileLangMatMul

    Returns (n_matmulscale, n_fused_matmullif, n_decomposed) for logging.
    """
    n_matmulscale = n_fused = n_decomposed = 0

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None or node.op_type != OpType.MatMul:
            continue
        if node.assigned_kernel != KernelVariant.TileLangMatMul:
            continue  # already reassigned

        # Try to find a Scale/Mul-constant successor
        scale_node, scale_value = _find_scale_node(ir, nid, max_depth=3)

        if scale_node is not None and scale_value is not None:
            # Absorb Scale into MatMul epilogue
            node.assigned_kernel = KernelVariant.TileLangMatMulScale
            node.extra_attrs["scale_value"] = scale_value
            node.bound_type = BoundType.COMPUTE
            scale_node.assigned_kernel = KernelVariant.ZeroCost
            scale_node.bound_type = BoundType.ZERO

            # Now check: does the Scale feed into a LIF through zero-cost ops?
            # This handles MatMul→Scale→...→LIF chains (MaxFormer, DSSA)
            neuron = _find_neuron_through_transparent(ir, scale_node.id, max_depth=5)
            if neuron is not None:
                n_matmulscale += 1  # scale was absorbed
                # Context-aware: MatMul→Scale→LIF in attention is always
                # sequential (neuron depends on this matmul) — always fuse.
                M_per_t = _matmul_M_per_t(node, ir)
                if _is_sequential_context(ir, nid) or M_per_t >= FUSED_THRESHOLD:
                    node.assigned_kernel = KernelVariant.TileLangFusedMatMulLIF
                    node.neuron_params = neuron.neuron_params
                    neuron.assigned_kernel = KernelVariant.ZeroCost
                    neuron.bound_type = BoundType.ZERO
                    n_fused += 1
                else:
                    n_decomposed += 1
            else:
                # Scale absorbed but no LIF — just MatMulScale
                n_matmulscale += 1
            continue

        if scale_node is not None and scale_value is None:
            # Mul with non-scalar constant (e.g. per-head scale tensor) —
            # can't embed in kernel epilogue, leave as separate Elementwise
            pass

        # No Scale found — check for direct LIF successor
        neuron = _find_neuron_through_transparent(ir, nid, max_depth=5)
        if neuron is not None:
            M_per_t = _matmul_M_per_t(node, ir)
            # Context-aware: always fuse sequential MatMul→LIF
            if _is_sequential_context(ir, nid) or M_per_t >= FUSED_THRESHOLD:
                node.assigned_kernel = KernelVariant.TileLangFusedMatMulLIF
                node.neuron_params = neuron.neuron_params
                node.bound_type = BoundType.COMPUTE
                neuron.assigned_kernel = KernelVariant.ZeroCost
                neuron.bound_type = BoundType.ZERO
                n_fused += 1
            else:
                node.assigned_kernel = KernelVariant.TileLangMatMulScale
                node.extra_attrs["scale_value"] = 1.0
                node.bound_type = BoundType.COMPUTE
                n_decomposed += 1
            continue

    return n_matmulscale, n_fused, n_decomposed


def _matmul_M_per_t(node, ir):
    """Compute per-timestep M dimension for a MatMul node."""
    if not node.output_shapes or len(node.output_shapes[0]) < 2:
        return 0
    M_full = node.output_shapes[0][0]
    T_val = ir.T if ir.T > 0 else 4
    return M_full // T_val


_ATTN_KERNEL_VARIANTS = {
    KernelVariant.TileLangMatMulScale,
    KernelVariant.TileLangFusedMatMulLIF,
    KernelVariant.TileLangMatMul,
}


def _mark_attention_conv_cudnn(ir: EngineIR) -> int:
    """Mark Conv nodes adjacent to attention MatMul ops as CuDNNConv.

    Attention Conv projections (Q/K/V/Wproj) have ONNX shapes that may not
    match TileLang's NHWC layout after attention Reshape/Transpose ops.
    cuDNN handles any layout natively and is equally fast for these small ops.

    Detection: walk backwards from each attention MatMul through zero-cost
    ops to find Conv/LIF/BN predecessors, then walk further to find their
    Conv predecessors.
    """
    # Collect all attention MatMul node IDs
    attn_matmul_nids = set()
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node and node.assigned_kernel in _ATTN_KERNEL_VARIANTS:
            attn_matmul_nids.add(nid)

    if not attn_matmul_nids:
        return 0

    # BFS both directions from attention MatMuls to find ALL Conv/Linear
    # nodes within attention blocks (Q/K/V projections, proj, MLP convs).
    _CONTINUE_OPS = _TRANSPARENT_OPS | _NEURON_OPS | {
        OpType.Add, OpType.Scale, OpType.Mul, OpType.Sub,
    }
    attn_conv_nids = set()
    visited = set(attn_matmul_nids)
    queue = list(attn_matmul_nids)
    max_depth = 12

    for _ in range(max_depth):
        next_queue = []
        for nid in queue:
            # Walk both predecessors and successors
            neighbors = ir.predecessors(nid) + ir.successors(nid)
            for nb_nid in neighbors:
                if nb_nid in visited:
                    continue
                visited.add(nb_nid)
                nb = ir.nodes.get(nb_nid)
                if nb is None:
                    continue
                if nb.op_type in (OpType.Conv2d, OpType.Linear):
                    attn_conv_nids.add(nb_nid)
                    next_queue.append(nb_nid)
                elif nb.op_type in _CONTINUE_OPS:
                    next_queue.append(nb_nid)
        queue = next_queue

    # Mark attention Conv/Linear nodes as cuDNN/cuBLAS fallback
    count = 0
    for nid in attn_conv_nids:
        node = ir.nodes[nid]
        if node.op_type == OpType.Conv2d:
            if node.assigned_kernel not in (KernelVariant.CuDNNConv,):
                node.assigned_kernel = KernelVariant.CuDNNConv
                count += 1
        elif node.op_type == OpType.Linear:
            if node.assigned_kernel not in (KernelVariant.CuBLASGemm,):
                node.assigned_kernel = KernelVariant.CuBLASGemm
                count += 1

    return count


def detect_fusion_groups(ir: EngineIR):
    """Detect Conv→(Reshape/Transpose)→Neuron patterns and create FusionGroups.

    Looks through transparent ops (Reshape, Transpose, Identity) between
    Conv and Neuron. This handles transformer models where TDL inserts
    reshapes between Conv output and the neuron.
    """
    ir.fusion_groups.clear()
    group_id = 0

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None or node.op_type != OpType.Conv2d:
            continue
        if node.fusion_group_id >= 0:
            continue

        # Find neuron through transparent ops
        neuron = _find_neuron_through_transparent(ir, nid)
        if neuron is None:
            continue
        if neuron.fusion_group_id >= 0:
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
        cp = node.conv_params
        if cp:
            logger.debug("  Fusion %d: Conv(%d→%d) + %s",
                         group_id, cp.in_channels, cp.out_channels, neuron.op_type.name)
        group_id += 1

    logger.phase("FUSION", "Detected %d Conv→Neuron fusion groups", len(ir.fusion_groups))


# ============================================================
# Pass 3: Sparsity Marking
# ============================================================

def validate_2_4_pattern(weight: np.ndarray) -> bool:
    """Check if a weight matrix has valid 2:4 structured sparsity.

    For Conv2d [C_out, C_in, kH, kW]: SBC prunes in columnslast order,
    i.e., reshape to [C_out, kH*kW*C_in] (R*S*C) then check groups of 4.
    For 1x1 convs this is equivalent to standard (C_out, C_in) order.
    For Linear [out, in]: check along the in_features axis.
    """
    if weight.ndim == 4:
        K, C, R, S = weight.shape
        # Columnslast: (K, C, R, S) → transpose(0,2,3,1) → (K, R, S, C) → (K, R*S*C)
        flat = weight.transpose(0, 2, 3, 1).reshape(K, -1)
    elif weight.ndim == 2:
        flat = weight
    else:
        return False

    rows, cols = flat.shape
    if cols < 4 or cols % 4 != 0:
        return False

    # Check each group of 4 has exactly 2 zeros
    groups = flat.reshape(rows, cols // 4, 4)
    zero_counts = np.sum(np.abs(groups) < 1e-10, axis=2)
    return bool(np.all(zero_counts == 2))


def mark_sparsity(ir: EngineIR):
    """Mark Conv nodes whose weights have valid 2:4 structure."""
    for node in ir.nodes.values():
        if node.op_type != OpType.Conv2d:
            continue
        if node.weight_info is None:
            continue

        weight_name = node.weight_info.name
        if weight_name not in ir.weights:
            continue

        weight = ir.weights[weight_name]

        # Stem conv (in_channels=3) is not eligible for 2:4 sparsity
        if node.conv_params and node.conv_params.in_channels < 4:
            node.sparse_weight = False
            continue

        if validate_2_4_pattern(weight):
            node.sparse_weight = True
            if node.weight_info:
                node.weight_info.is_sparse = True
            cp = node.conv_params
            if cp:
                logger.debug("  Sparse: Conv(%d→%d, %dx%d) %s",
                             cp.in_channels, cp.out_channels,
                             cp.kernel_h, cp.kernel_w, weight_name)


# ============================================================
# Pass 4: Layout Annotation
# ============================================================

def annotate_layout(ir: EngineIR):
    """Annotate tensor layout for each node.

    NHWC for all tensor-core compute paths (Conv, MatMul).
    NCHW for the stem (first Conv with in_channels=3) input boundary.
    """
    for node in ir.nodes.values():
        if node.op_type in (OpType.Conv2d, OpType.IF, OpType.LIF, OpType.MS,
                            OpType.Add, OpType.MaxPool, OpType.GlobalAvgPool):
            node.layout = TensorLayout.NHWC
        else:
            node.layout = TensorLayout.NCHW


# ============================================================
# Pass 4b: Per-Edge Layout Propagation
# ============================================================

def propagate_edge_layouts(ir: EngineIR):
    """Propagate tensor layouts along edges based on kernel contracts.

    After this pass, every edge.layout reflects the actual data layout
    that the producer writes. This enables the reformat insertion pass
    to detect layout mismatches and insert explicit LayoutTranspose nodes.

    Layout rules:
      - Kernel with contract: output layout from KERNEL_CONTRACTS
      - Inherit-type kernel (Elementwise, TemporalMean, etc.): copy predecessor
      - Reshape to non-4D: ND
      - Reshape 4D→4D: inherit predecessor
      - Flatten: ND
      - Transpose perm (0,2,3,1) on NCHW → NHWC
      - Transpose perm (0,3,1,2) on NHWC → NCHW
      - Transpose other perm: ND
      - Identity: inherit
    """
    # Build edge lookup: src_nid → [Edge]
    src_edges: dict[int, list[Edge]] = {}
    for e in ir.edges:
        src_edges.setdefault(e.src_id, []).append(e)

    # Build predecessor-edge lookup: dst_nid → [Edge]
    dst_edges: dict[int, list[Edge]] = {}
    for e in ir.edges:
        dst_edges.setdefault(e.dst_id, []).append(e)

    # The graph input is NCHW (from ONNX), but __call__ converts to NHWC
    # before the first node. Mark the Tile/Repeat node's output as NHWC
    # since it produces the first activation after NCHW→NHWC conversion.
    graph_input_layout = TensorLayout.NHWC  # after entry conversion

    propagated = 0
    for nid in ir.topo_order:
        node = ir.nodes[nid]
        kv = node.assigned_kernel

        # Determine this node's output layout
        out_layout = _infer_node_output_layout(
            node, kv, dst_edges.get(nid, []), graph_input_layout)

        # Stamp onto all outgoing edges
        for e in src_edges.get(nid, []):
            if e.layout != out_layout:
                e.layout = out_layout
                propagated += 1

    logger.phase("LAYOUT", "Propagated edge layouts for %d edges", propagated)


def _infer_node_output_layout(node, kv, incoming_edges, default_layout):
    """Determine what layout a node's output tensor has."""
    contract = KERNEL_CONTRACTS.get(kv)

    # 1. Kernel with explicit contract → use it
    if contract is not None:
        return contract.output_layout

    # 2. Inherit-type kernel → copy from first predecessor edge
    pred_layout = _get_predecessor_layout(incoming_edges, default_layout)

    # 3. ZeroCost ops: layout depends on op semantics
    if node.op_type == OpType.Flatten:
        return TensorLayout.ND

    if node.op_type == OpType.Reshape:
        out_shape = node.output_shapes[0] if node.output_shapes else ()
        if len(out_shape) != 4:
            return TensorLayout.ND
        # 4D→4D reshape: preserve predecessor layout
        return pred_layout

    if node.op_type == OpType.Transpose:
        perm = node.extra_attrs.get("perm", [])
        if len(perm) == 4:
            if perm == [0, 2, 3, 1] and pred_layout == TensorLayout.NCHW:
                return TensorLayout.NHWC
            if perm == [0, 3, 1, 2] and pred_layout == TensorLayout.NHWC:
                return TensorLayout.NCHW
            # Other 4D permutations: could be multi-head reshuffles
            return TensorLayout.ND
        return TensorLayout.ND

    if node.op_type == OpType.Identity:
        return pred_layout

    # For Tile/Repeat at graph entry, output NHWC (after __call__ conversion)
    if node.op_type == OpType.Tile:
        return default_layout

    # Default: inherit
    return pred_layout


def _get_predecessor_layout(incoming_edges, default):
    """Get the layout from the first incoming edge, or default."""
    for e in incoming_edges:
        return e.layout
    return default


# ============================================================
# Pass 4c: Layout Reformat Insertion
# ============================================================

def insert_layout_reformats(ir: EngineIR) -> int:
    """Insert LayoutTranspose nodes where layout mismatches occur.

    Two cases:
    1. Kernel contract mismatch: producer layout ≠ consumer expected layout
    2. Reshape reinterpretation: NHWC data enters a 4D→4D Reshape whose
       target_shape assumes NCHW. ONNX shapes are always NCHW, so any
       Reshape that changes spatial structure on NHWC data needs NHWC→NCHW
       conversion first.

    Returns the number of reformat nodes inserted.
    """
    edges_by_dst: dict[int, list[Edge]] = {}
    for e in ir.edges:
        edges_by_dst.setdefault(e.dst_id, []).append(e)

    reformats_to_insert = []

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue

        # Case 1: Kernel contract mismatch
        contract = KERNEL_CONTRACTS.get(node.assigned_kernel)
        if contract is not None and contract.input_layout not in (TensorLayout.ND, None):
            expected_input = contract.input_layout
            for e in edges_by_dst.get(nid, []):
                if e.layout == expected_input:
                    continue
                if e.layout == TensorLayout.ND:
                    continue  # ND: compatible with any layout
                elif e.layout == TensorLayout.NCHW and expected_input == TensorLayout.NHWC:
                    perm = [0, 2, 3, 1]
                elif e.layout == TensorLayout.NHWC and expected_input == TensorLayout.NCHW:
                    perm = [0, 3, 1, 2]
                else:
                    continue
                src = ir.nodes.get(e.src_id)
                if src and src.assigned_kernel == KernelVariant.LayoutTranspose:
                    continue  # avoid chaining reformats
                reformats_to_insert.append((e.src_id, nid, perm, e.layout, expected_input))
            continue

        # Case 2: Reshape in attention path receiving NHWC data.
        # Only insert NHWC→NCHW before Reshape nodes that feed into
        # MatMul (attention Q@K^T, attn@V). Detected by checking if
        # the Reshape's output chain reaches a MatMul through ZeroCost ops.
        if node.op_type == OpType.Reshape:
            # Check if this Reshape feeds a MatMul (attention path)
            feeds_matmul = False
            cur = nid
            for _ in range(5):
                succs = ir.successors(cur)
                if not succs:
                    break
                s = ir.nodes.get(succs[0])
                if s is None:
                    break
                if s.op_type == OpType.MatMul:
                    feeds_matmul = True
                    break
                if s.op_type in (OpType.Reshape, OpType.Transpose,
                                 OpType.Identity, OpType.Scale):
                    cur = succs[0]
                    continue
                break
            if not feeds_matmul:
                continue
            for e in edges_by_dst.get(nid, []):
                if e.layout != TensorLayout.NHWC:
                    continue
                src = ir.nodes.get(e.src_id)
                if src is None or src.assigned_kernel == KernelVariant.LayoutTranspose:
                    continue
                src_shape = src.output_shapes[0] if src.output_shapes else ()
                if len(src_shape) != 4:
                    continue
                reformats_to_insert.append((
                    e.src_id, nid, [0, 3, 1, 2],
                    TensorLayout.NHWC, TensorLayout.NCHW))

        # Case 3: Multi-input node (Add) with mixed layouts.
        # Both inputs must match. If one is NHWC and one is NCHW,
        # convert the NCHW one to NHWC (since Conv output is NHWC).
        if node.op_type == OpType.Add:
            in_edges = edges_by_dst.get(nid, [])
            if len(in_edges) >= 2:
                layouts = [e.layout for e in in_edges]
                if TensorLayout.NHWC in layouts and TensorLayout.NCHW in layouts:
                    for e in in_edges:
                        src = ir.nodes.get(e.src_id)
                        if (e.layout == TensorLayout.NCHW
                                and src and src.assigned_kernel != KernelVariant.LayoutTranspose):
                            reformats_to_insert.append((
                                e.src_id, nid, [0, 2, 3, 1],
                                TensorLayout.NCHW, TensorLayout.NHWC))

    # Insert reformat nodes
    inserted = 0
    for src_id, dst_id, perm, src_layout, dst_layout in reformats_to_insert:
        src_node = ir.nodes[src_id]

        # Output shape is SAME as input shape — LayoutTranspose changes
        # physical memory layout, not logical NCHW shape.
        src_shape = src_node.output_shapes[0] if src_node.output_shapes else ()

        reformat_name = f"reformat_{src_layout.name}_to_{dst_layout.name}_{src_id}_{dst_id}"
        out_tensor_name = f"{reformat_name}_output"
        reformat_node = Node(
            id=ir._next_id,
            name=reformat_name,
            op_type=OpType.Transpose,
            input_names=list(src_node.output_names),
            output_names=[out_tensor_name],
            input_shapes=[src_shape] if src_shape else [],
            output_shapes=[src_shape] if src_shape else [],  # same shape
            extra_attrs={"perm": perm},
            assigned_kernel=KernelVariant.LayoutTranspose,
            bound_type=BoundType.MEMORY,
            layout=dst_layout,
        )
        ir._next_id += 1
        ir.nodes[reformat_node.id] = reformat_node

        # Register in producer/consumer tracking (needed for build_edges)
        ir._tensor_producer[out_tensor_name] = reformat_node.id
        for in_name in reformat_node.input_names:
            if in_name not in ir._tensor_consumers:
                ir._tensor_consumers[in_name] = []
            ir._tensor_consumers[in_name].append(reformat_node.id)

        # Rewire: dst node consumes reformat's output instead of src's
        dst_node = ir.nodes[dst_id]
        for i, inp_name in enumerate(dst_node.input_names):
            if inp_name in src_node.output_names:
                dst_node.input_names[i] = out_tensor_name
                # Update consumer tracking
                if out_tensor_name not in ir._tensor_consumers:
                    ir._tensor_consumers[out_tensor_name] = []
                ir._tensor_consumers[out_tensor_name].append(dst_id)
                break

        inserted += 1

    if inserted > 0:
        logger.phase("REFORMAT", "Inserted %d layout reformat nodes", inserted)

    return inserted


# ============================================================
# Pass 5: Kernel Assignment
# ============================================================

def assign_kernels(ir: EngineIR):
    """Assign kernel variants based on fusion groups and sparsity."""
    for fg in ir.fusion_groups:
        conv = ir.nodes[fg.conv_node_id]
        neuron = ir.nodes[fg.neuron_node_id]

        if conv.sparse_weight:
            fg.variant = KernelVariant.FusedSparseConvBNLIF
            conv.assigned_kernel = KernelVariant.FusedSparseConvBNLIF
            neuron.assigned_kernel = KernelVariant.FusedSparseConvBNLIF
        else:
            # Non-sparse Conv: use cuDNN + standalone neuron
            conv.assigned_kernel = KernelVariant.CuDNNConv
            neuron.assigned_kernel = KernelVariant.StandaloneLIF
            fg.variant = KernelVariant.CuDNNConv

    # Non-fused nodes keep their default assignments from parser
    for node in ir.nodes.values():
        if node.fusion_group_id >= 0:
            continue  # already assigned
        # Standalone Conv without fusion
        if node.op_type == OpType.Conv2d:
            if node.sparse_weight:
                node.assigned_kernel = KernelVariant.FusedSparseConvBNLIF
            else:
                node.assigned_kernel = KernelVariant.CuDNNConv


# ============================================================
# Pass 6: Shape Propagation
# ============================================================

def propagate_shapes(ir: EngineIR, batch_size: int = 1):
    """Propagate spatial dimensions through the graph from model input.

    After this pass, every node has correct input_shapes and output_shapes
    in NCHW order (N, C, H, W). Uses the RUNTIME batch_size (not ONNX original).
    """
    shape_map: dict[str, tuple] = {}

    # Scale model input shape to runtime batch size
    # ONNX was exported at B=1; runtime may use B=64
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

    # Assign model_input_shape to all graph input tensors that look like data inputs
    if model_input:
        for tensor_name in graph_inputs:
            if tensor_name not in ir.weights:
                shape_map[tensor_name] = model_input

    # Seed shape_map with weight shapes for Transpose-of-weight nodes
    # (needed for SpikingResFormer's classifier weight transpose)
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node and node.op_type == OpType.Transpose:
            for in_name in node.input_names:
                if in_name in ir.weights and in_name not in shape_map:
                    shape_map[in_name] = tuple(int(d) for d in ir.weights[in_name].shape)

    propagated = 0
    for nid in ir.topo_order:
        node = ir.nodes[nid]

        # Collect input shapes from shape_map
        in_shapes = []
        for in_name in node.input_names:
            if in_name in shape_map:
                in_shapes.append(shape_map[in_name])

        if not in_shapes and node.input_shapes:
            in_shapes = list(node.input_shapes)

        if in_shapes:
            node.input_shapes = in_shapes

        # Compute output shape based on op type
        # Special handling for Split: each output has a different shape
        if (node.op_type == OpType.Identity
                and node.extra_attrs.get("original_op") == "Split"
                and len(node.output_names) > 1 and in_shapes):
            axis = node.extra_attrs.get("axis", 0)
            num_outputs = len(node.output_names)
            inp_shape = list(in_shapes[0])
            if 0 <= axis < len(inp_shape):
                split_dim = inp_shape[axis] // num_outputs
                for i, out_name in enumerate(node.output_names):
                    s = list(inp_shape)
                    s[axis] = split_dim
                    shape_map[out_name] = tuple(s)
                node.output_shapes = [tuple(list(inp_shape[:axis])
                                            + [split_dim]
                                            + list(inp_shape[axis+1:]))]
                propagated += 1
        else:
            out_shape = _compute_output_shape(node, in_shapes, ir.T)
            if out_shape:
                if not node.output_shapes or node.output_shapes[0] != out_shape:
                    node.output_shapes = [out_shape]
                    propagated += 1
                for out_name in node.output_names:
                    shape_map[out_name] = out_shape

    logger.phase("SHAPES", "Propagated shapes for %d nodes", propagated)


def _compute_output_shape(node: Node, in_shapes: list[tuple], T: int) -> tuple:
    """Compute output shape for a node given its input shapes."""
    if not in_shapes:
        return ()

    inp = in_shapes[0]

    if node.op_type == OpType.Conv2d and node.conv_params:
        cp = node.conv_params
        if len(inp) == 4:
            N, C, H, W = inp
        elif len(inp) == 3:
            N, H, W = inp[0], inp[1], inp[2]
            C = cp.in_channels
        else:
            return ()
        OH = (H + 2 * cp.pad_h - cp.dilation_h * (cp.kernel_h - 1) - 1) // cp.stride_h + 1
        OW = (W + 2 * cp.pad_w - cp.dilation_w * (cp.kernel_w - 1) - 1) // cp.stride_w + 1
        return (N, cp.out_channels, OH, OW)

    if node.op_type == OpType.MaxPool and node.pool_params:
        pp = node.pool_params
        if len(inp) < 4:
            return inp
        N, C, H, W = inp
        kh = pp.get("kernel_h", pp.get("kernel_size", 3))
        kw = pp.get("kernel_w", pp.get("kernel_size", 3))
        sh = pp.get("stride_h", pp.get("stride", 2))
        sw = pp.get("stride_w", pp.get("stride", 2))
        ph = pp.get("pad_h", pp.get("padding", 1))
        pw = pp.get("pad_w", pp.get("padding", 1))
        OH = (H + 2 * ph - kh) // sh + 1
        OW = (W + 2 * pw - kw) // sw + 1
        return (N, C, OH, OW)

    if node.op_type == OpType.GlobalAvgPool:
        if len(inp) >= 4:
            return (inp[0], inp[1], 1, 1)
        if len(inp) == 3:
            return (inp[0], inp[1], 1)
        return inp

    if node.op_type in (OpType.IF, OpType.LIF, OpType.MS,
                        OpType.Add, OpType.Mul, OpType.Sub, OpType.Scale):
        return inp

    if node.op_type == OpType.Flatten:
        if len(inp) >= 4:
            return (inp[0], inp[1] * inp[2] * inp[3])
        return inp

    if node.op_type == OpType.Gemm and node.gemm_params:
        out_features = node.gemm_params.get("N", 0)
        # Derive N from weight shape if not in gemm_params
        if out_features == 0 and node.weight_info and node.weight_info.shape:
            ws = node.weight_info.shape
            transB = node.gemm_params.get("transB", 0)
            out_features = ws[0] if transB else ws[1] if len(ws) >= 2 else 0
        if len(inp) >= 2:
            return (inp[0], out_features)
        return inp

    if node.op_type == OpType.Linear:
        # Linear (from MatMul with weight): input × weight → output
        if node.gemm_params:
            N_out = node.gemm_params.get("N", 0)
            if N_out > 0:
                return inp[:-1] + (N_out,)
        if node.weight_info and node.weight_info.shape:
            N_out = node.weight_info.shape[-1]
            return inp[:-1] + (N_out,)
        return inp

    if node.op_type == OpType.MatMul:
        # Dynamic matmul: (..., M, K) × (..., K, N) → (..., M, N)
        if len(in_shapes) >= 2:
            a, b = in_shapes[0], in_shapes[1]
            if len(a) >= 2 and len(b) >= 2:
                return a[:-1] + (b[-1],)
        return inp

    if node.op_type == OpType.Tile:
        if len(inp) >= 1:
            return (inp[0] * T,) + inp[1:]
        return inp

    if node.op_type == OpType.TemporalMean:
        if len(inp) >= 1:
            B = inp[0] // T if T > 0 else inp[0]
            return (max(B, 1),) + inp[1:]  # clamp for dynamic dims
        return inp

    if node.op_type == OpType.Transpose:
        perm = node.extra_attrs.get("perm")
        if perm and inp and len(perm) == len(inp):
            return tuple(inp[p] for p in perm)
        # Fallback: if input is a weight constant, use its shape
        if perm and not inp:
            for name in node.input_names:
                if name in ir.weights:
                    w_shape = ir.weights[name].shape
                    if len(perm) == len(w_shape):
                        return tuple(w_shape[p] for p in perm)
        if node.output_shapes:
            return node.output_shapes[0]
        return inp

    if node.op_type == OpType.Reshape:
        target = node.extra_attrs.get("target_shape")
        if target:
            result = list(target)
            total = 1
            for d in inp:
                total *= d
            known = 1
            neg_idx = -1
            for i, d in enumerate(result):
                if d == -1:
                    neg_idx = i
                elif d == 0 and i < len(inp):
                    result[i] = inp[i]
                    known *= inp[i]
                else:
                    known *= d
            if neg_idx >= 0 and known > 0:
                result[neg_idx] = total // known
            return tuple(result)
        if node.output_shapes:
            return node.output_shapes[0]
        return inp

    if node.op_type == OpType.Identity:
        original_op = node.extra_attrs.get("original_op", "")
        # Slice: output shape differs from input on the sliced axis
        if original_op == "Slice" and inp:
            axes = node.extra_attrs.get("axes")
            starts = node.extra_attrs.get("starts")
            ends = node.extra_attrs.get("ends")
            if axes is not None and starts is not None and ends is not None:
                out = list(inp)
                for a, s, e in zip(axes, starts, ends):
                    if 0 <= a < len(out):
                        dim_size = out[a]
                        # Clamp per ONNX Slice spec
                        e_clamped = min(e, dim_size) if e > 0 else max(0, dim_size + e)
                        s_clamped = max(0, s) if s >= 0 else max(0, dim_size + s)
                        out[a] = e_clamped - s_clamped
                return tuple(out)
        if node.output_shapes:
            return node.output_shapes[0]
        return inp

    if node.op_type == OpType.ReduceMean:
        if node.output_shapes:
            return node.output_shapes[0]
        return inp

    if node.op_type == OpType.FusedAttention:
        ap = node.attention_params
        if ap and ap.variant == "dssa":
            # DSSA: output is (TB, C, H_in, W_in) from the query path
            # Use the second input shape (x_query), not the first (y_kv)
            if len(in_shapes) >= 2:
                return in_shapes[1]  # x_query shape
            # Fallback: construct from params
            C = ap.num_heads * ap.head_dim
            return (inp[0], C, ap.H, ap.W)
        # SpikFormer: (TB, N, C) → (TB, N, C)
        # MaxFormer: (TB, C, H, W) → (TB, C, H, W)
        return inp

    return inp


# ============================================================
# Pass 7: Bound Classification + TileLang Kernel Assignment
# ============================================================

_NEURON_OPS = {OpType.IF, OpType.LIF, OpType.MS}
_COMPUTE_OPS = {OpType.Conv2d, OpType.Linear, OpType.MatMul, OpType.Gemm}
_MEMORY_OPS = {OpType.Add, OpType.Mul, OpType.Sub, OpType.MaxPool,
               OpType.GlobalAvgPool, OpType.TemporalMean, OpType.Scale}
_ZERO_OPS = {OpType.Flatten, OpType.Reshape, OpType.Transpose, OpType.Identity,
             OpType.Tile, OpType.InputConvert, OpType.OutputConvert,
             OpType.Concat, OpType.ReduceMean}


def _is_sequential_context(ir: EngineIR, conv_nid: int) -> bool:
    """Return True if this Conv is in a sequential chain with no parallel siblings.

    A Conv is in a sequential context if its data-producing predecessors do NOT
    fan out to multiple compute-bound consumers. Sequential chains (backbone
    Conv→IF→Conv→IF) should always fuse because there is no parallel work
    available to overlap with standalone neuron kernels.

    Multi-branch contexts (e.g., Q/K/V parallel projections sharing the same
    input) can benefit from decomposition: BA-MTTS interleaves the separate
    compute and memory ops for hardware overlap.
    """
    for pred_nid in ir.predecessors(conv_nid):
        pred = ir.nodes.get(pred_nid)
        if pred is None:
            continue
        # Walk through transparent ops to find the real data producer
        if pred.op_type in _ZERO_OPS:
            # Check the transparent node's predecessors recursively (1 level)
            for gpred_nid in ir.predecessors(pred_nid):
                succs = ir.successors(gpred_nid)
                compute_succs = sum(1 for s in succs
                                    if ir.nodes.get(s) and
                                    ir.nodes[s].op_type in _COMPUTE_OPS)
                if compute_succs >= 2:
                    return False
        succs = ir.successors(pred_nid)
        compute_succs = sum(1 for s in succs
                            if ir.nodes.get(s) and
                            ir.nodes[s].op_type in _COMPUTE_OPS)
        if compute_succs >= 2:
            return False
    return True


def classify_bound_and_assign_tilelang(ir: EngineIR, batch_size: int = 1):
    """Classify each node as compute/memory/zero-bound and assign TileLang kernels.

    Hybrid strategy:
    - Small batch (M_per_t < threshold): DECOMPOSED Conv+BN + IF with BA-MTTS
    - Large batch (M_per_t >= threshold): PER-TIMESTEP FUSED Conv+BN+IF (T=1/launch)

    The threshold is when the per-timestep GEMM saturates GPU SMs (~50K elements
    for RTX 4090 with 128 SMs).
    """
    # SM saturation threshold: per-timestep M must produce enough GEMM tiles
    # to fill all SMs. With block_M=64, need M/64 >= SM_count → M >= SM*64
    try:
        import torch
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        sm_count = 128
    FUSED_THRESHOLD = sm_count * 384  # ~49K on RTX 4090 (128*384)

    n_compute = n_memory = n_zero = 0
    n_fused = n_decomposed = 0

    # Build map: conv_node_id → successor_neuron_node_id (from fusion groups)
    conv_to_neuron = {}
    for fg in ir.fusion_groups:
        conv_to_neuron[fg.conv_node_id] = fg.neuron_node_id

    for nid in ir.topo_order:
        node = ir.nodes[nid]

        # Classify bound type
        if node.op_type in _COMPUTE_OPS:
            node.bound_type = BoundType.COMPUTE
            n_compute += 1
        elif node.op_type in _NEURON_OPS:
            node.bound_type = BoundType.MEMORY
            n_memory += 1
        elif node.op_type in _MEMORY_OPS:
            node.bound_type = BoundType.MEMORY
            n_memory += 1
        else:
            node.bound_type = BoundType.ZERO
            n_zero += 1

        # Assign TileLang kernel variants — HYBRID strategy
        if node.op_type == OpType.Conv2d and node.conv_params:
            cp = node.conv_params
            K_red = cp.kernel_h * cp.kernel_w * cp.in_channels
            if cp.groups > 1 and cp.groups == cp.in_channels:
                # Depthwise conv: use TileLang DW kernel
                M_per_t = 0
                if node.output_shapes and len(node.output_shapes[0]) == 4:
                    TB_out, C_out, OH_out, OW_out = node.output_shapes[0]
                    B_out = TB_out // ir.T if ir.T > 0 else TB_out
                    M_per_t = B_out * OH_out * OW_out
                neuron_nid = conv_to_neuron.get(nid)
                # Context-aware: DWConv in backbone is sequential, always fuse
                if neuron_nid is None:
                    dw_fused = False
                elif _is_sequential_context(ir, nid):
                    dw_fused = True
                else:
                    dw_fused = M_per_t >= FUSED_THRESHOLD
                if dw_fused:
                    node.assigned_kernel = KernelVariant.TileLangFusedDWConvBNIF
                    node.bound_type = BoundType.COMPUTE
                    n_fused += 1
                    if neuron_nid in ir.nodes:
                        ir.nodes[neuron_nid].assigned_kernel = KernelVariant.ZeroCost
                        ir.nodes[neuron_nid].bound_type = BoundType.ZERO
                else:
                    node.assigned_kernel = KernelVariant.TileLangDWConvBN
                    n_decomposed += 1
            elif cp.groups > 1 and cp.groups != cp.in_channels:
                # Grouped conv (not depthwise): use TileLang grouped kernel
                node.assigned_kernel = KernelVariant.TileLangGroupedConvBN
                n_decomposed += 1
            elif cp.in_channels < 4 or K_red % 8 != 0:
                # Stem conv (C_in<4) or misaligned dims: cuDNN fallback
                node.assigned_kernel = KernelVariant.CuDNNConv
            else:
                # Compute M_per_timestep from output shapes
                M_per_t = 0
                if node.output_shapes and len(node.output_shapes[0]) == 4:
                    TB_out, C_out, OH_out, OW_out = node.output_shapes[0]
                    B_out = TB_out // ir.T if ir.T > 0 else TB_out
                    M_per_t = B_out * OH_out * OW_out

                neuron_nid = conv_to_neuron.get(nid)
                # Context-aware fusion: sequential chains always fuse (no
                # overlap opportunity for standalone neuron); multi-branch
                # (Q/K/V) uses threshold (BA-MTTS can interleave C↔M).
                if neuron_nid is None:
                    use_fused = False
                elif _is_sequential_context(ir, nid):
                    use_fused = True
                else:
                    use_fused = M_per_t >= FUSED_THRESHOLD

                if use_fused:
                    # Per-timestep fused: Conv+BN+IF in one kernel (T=1/launch)
                    # Both Conv and neuron are handled by the fused kernel
                    #
                    # Check if neuron feeds into a residual Add that can be
                    # absorbed into the epilogue (avoids extra DRAM round-trip)
                    has_residual = False
                    if neuron_nid is not None and neuron_nid in ir.nodes:
                        neuron_succs = ir.successors(neuron_nid)
                        if len(neuron_succs) == 1:
                            add_cand = ir.nodes.get(neuron_succs[0])
                            if (add_cand is not None and
                                    add_cand.op_type == OpType.Add and
                                    add_cand.assigned_kernel != KernelVariant.ZeroCost):
                                has_residual = True
                                add_cand.assigned_kernel = KernelVariant.ZeroCost
                                add_cand.bound_type = BoundType.ZERO
                                node.extra_attrs["has_residual_add"] = True
                                node.extra_attrs["residual_add_nid"] = add_cand.id

                    if has_residual:
                        if cp.kernel_h == 1 and cp.kernel_w == 1:
                            node.assigned_kernel = KernelVariant.TileLangFusedConv1x1BNIFAdd
                        else:
                            node.assigned_kernel = KernelVariant.TileLangFusedConvBNIFAdd
                    else:
                        if cp.kernel_h == 1 and cp.kernel_w == 1:
                            node.assigned_kernel = KernelVariant.TileLangFusedConv1x1BNIF
                        else:
                            node.assigned_kernel = KernelVariant.TileLangFusedConvBNIF
                    node.bound_type = BoundType.COMPUTE  # fused = single compute op
                    n_fused += 1
                    # Mark the neuron as handled (zero-cost passthrough)
                    if neuron_nid is not None and neuron_nid in ir.nodes:
                        ir.nodes[neuron_nid].assigned_kernel = KernelVariant.ZeroCost
                        ir.nodes[neuron_nid].bound_type = BoundType.ZERO
                else:
                    # Decomposed: separate Conv+BN and IF for BA-MTTS
                    if cp.kernel_h == 1 and cp.kernel_w == 1:
                        node.assigned_kernel = KernelVariant.TileLangConv1x1BN
                    else:
                        node.assigned_kernel = KernelVariant.TileLangConvBN
                    n_decomposed += 1
            node.layout = TensorLayout.NHWC

        elif node.op_type == OpType.IF:
            # Only assign IF kernel if not already handled by fused Conv
            if node.assigned_kernel != KernelVariant.ZeroCost:
                node.assigned_kernel = KernelVariant.CUDAVec4IF
            node.layout = TensorLayout.NHWC

        elif node.op_type in (OpType.LIF, OpType.MS):
            if node.assigned_kernel != KernelVariant.ZeroCost:
                node.assigned_kernel = KernelVariant.CUDAVec4LIF
            node.layout = TensorLayout.NHWC

        elif node.op_type == OpType.Linear:
            node.assigned_kernel = KernelVariant.TileLangLinearBN

        elif node.op_type == OpType.MatMul:
            node.assigned_kernel = KernelVariant.TileLangMatMul

        elif node.op_type == OpType.FusedAttention:
            # Already assigned by parser — just set bound type
            node.bound_type = BoundType.COMPUTE

        elif node.op_type == OpType.Gemm:
            # Small classifier FC: keep cuBLAS
            node.assigned_kernel = KernelVariant.CuBLASGemm

        elif node.op_type in (OpType.Add, OpType.Sub, OpType.Mul, OpType.Scale):
            node.assigned_kernel = KernelVariant.Elementwise

        elif node.op_type in (OpType.MaxPool, OpType.GlobalAvgPool):
            node.assigned_kernel = KernelVariant.CuDNNPool

        elif node.op_type == OpType.TemporalMean:
            node.assigned_kernel = KernelVariant.TemporalMean

        elif node.op_type == OpType.Tile:
            node.assigned_kernel = KernelVariant.TileRepeat

    # Attention pattern detection: absorb Scale into first MatMul,
    # apply fused/decomposed for second MatMul + LIF
    n_attn_scale, n_attn_fused, n_attn_dec = _detect_attention_matmul_patterns(
        ir, FUSED_THRESHOLD)

    logger.phase("BOUND", "Classified %d compute, %d memory, %d zero-cost nodes "
                 "(hybrid: %d fused, %d decomposed, threshold M_per_t=%d; "
                 "attn: %d matmul+scale, %d fused matmul+lif, %d decomposed)",
                 n_compute, n_memory, n_zero, n_fused, n_decomposed, FUSED_THRESHOLD,
                 n_attn_scale, n_attn_fused, n_attn_dec)
