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
    Node, FusionGroup, EngineIR,
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

def detect_fusion_groups(ir: EngineIR):
    """Detect Conv→Neuron patterns and create FusionGroups.

    A fusion group requires:
    - Conv node has exactly one consumer
    - That consumer is a neuron node (IF/LIF/MS)
    - No other node consumes the Conv output

    The BN is already folded into Conv, so the pattern is Conv→Neuron directly.
    """
    ir.fusion_groups.clear()
    group_id = 0

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None or node.op_type != OpType.Conv2d:
            continue
        if node.fusion_group_id >= 0:
            continue  # already in a group

        # Check if Conv has exactly one consumer that is a neuron
        succs = ir.successors(nid)
        if len(succs) != 1:
            continue
        succ = ir.nodes.get(succs[0])
        if succ is None or succ.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
            continue

        # Check that the neuron's only input comes from this Conv
        neuron_preds = ir.predecessors(succ.id)
        if len(neuron_preds) != 1 or neuron_preds[0] != nid:
            continue

        # Create fusion group
        fg = FusionGroup(
            group_id=group_id,
            conv_node_id=nid,
            neuron_node_id=succ.id,
            bn_scale=node.bn_scale,
            bn_bias=node.bn_bias,
        )
        ir.fusion_groups.append(fg)
        node.fusion_group_id = group_id
        succ.fusion_group_id = group_id
        cp = node.conv_params
        if cp:
            logger.debug("  Fusion %d: Conv(%d→%d) + %s",
                         group_id, cp.in_channels, cp.out_channels, succ.op_type.name)
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
    # (skip weight/bias initializers which are in ir.weights)
    if model_input:
        for tensor_name in graph_inputs:
            if tensor_name not in ir.weights:
                shape_map[tensor_name] = model_input

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
            return (B,) + inp[1:]
        return inp

    if node.op_type == OpType.Transpose:
        perm = node.extra_attrs.get("perm")
        if perm and inp and len(perm) == len(inp):
            return tuple(inp[p] for p in perm)
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
        if node.output_shapes:
            return node.output_shapes[0]
        return inp

    if node.op_type == OpType.ReduceMean:
        if node.output_shapes:
            return node.output_shapes[0]
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


def _should_use_fused_per_t(cp, ir: EngineIR, batch_size: int,
                            sm_count: int = 128) -> bool:
    """Decide whether a Conv+IF pair should use per-timestep fused kernel.

    The per-timestep fused kernel (T=1 per launch) wins when the per-timestep
    GEMM M dimension saturates SMs. This happens at large batch sizes where
    B * OH * OW is large enough for efficient tile parallelism.

    Returns True if fused is expected to be faster.
    """
    T = ir.T
    if T <= 1:
        return True  # no temporal issue, always fuse

    OH = (cp.out_channels  # placeholder, real OH computed from shapes
          if not hasattr(cp, '_oh') else cp._oh)
    # Estimate OH from conv params (approximate)
    # This is set by propagate_shapes on the node's output_shapes
    return False  # conservative: need output shapes to decide


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
            if cp.in_channels < 4 or K_red % 8 != 0:
                node.assigned_kernel = KernelVariant.CuDNNConv
            else:
                # Compute M_per_timestep from output shapes
                M_per_t = 0
                if node.output_shapes and len(node.output_shapes[0]) == 4:
                    TB_out, C_out, OH_out, OW_out = node.output_shapes[0]
                    B_out = TB_out // ir.T if ir.T > 0 else TB_out
                    M_per_t = B_out * OH_out * OW_out

                neuron_nid = conv_to_neuron.get(nid)
                use_fused = (M_per_t >= FUSED_THRESHOLD and neuron_nid is not None)

                if use_fused:
                    # Per-timestep fused: Conv+BN+IF in one kernel (T=1/launch)
                    # Both Conv and neuron are handled by the fused kernel
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

    logger.phase("BOUND", "Classified %d compute, %d memory, %d zero-cost nodes "
                 "(hybrid: %d fused, %d decomposed, threshold M_per_t=%d)",
                 n_compute, n_memory, n_zero, n_fused, n_decomposed, FUSED_THRESHOLD)
