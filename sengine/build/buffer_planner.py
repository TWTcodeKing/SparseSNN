"""Buffer Planner: compute all GPU buffer allocations and kernel bindings at build time.

Replaces the fragile _setup_cpp_executor() Python dispatch with a clean
data-driven approach. The planner walks the optimized IR and produces:

1. BufferManifest: every GPU buffer (activations, weights, membranes, workspaces)
2. NodeExecPlan: per-node kernel config with buffer ID references (not pointers)

The output is serializable — it goes into the .sengine file. At load time,
the C++ executor reads the plan, allocates GPU memory, binds pointers, done.

Usage:
    from sengine.build.buffer_planner import plan_buffers
    manifest, exec_plan = plan_buffers(ir, schedule, T, batch_size)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sengine.ir import EngineIR, OpType, KernelVariant, BoundType, TensorLayout


# ─── Data structures (serializable) ───

@dataclass
class BufferDesc:
    """A GPU buffer allocation."""
    buf_id: int
    shape: tuple
    dtype: str          # "fp16" or "fp32"
    layout: str         # "nhwc", "nchw", "2d", "1d"
    category: str       # "activation", "weight", "weight_1x1", "bn_scale", "bn_bias",
                        # "membrane", "workspace", "graph_input", "graph_output"
    source_nid: int     # node that produces this buffer (-1 for weights/inputs)
    weight_key: Optional[str] = None  # key into IR weight dict


@dataclass
class NodeExecPlan:
    """Resolved execution descriptor for one node."""
    nid: int
    kernel_type: str        # "tilelang_5", "tilelang_6", "tilelang_3",
                            # "if_neuron", "lif_neuron", "add", "maxpool",
                            # "global_avgpool", "temporal_mean", "gemm",
                            # "fused_attn", "skip", "alias", "layout_transpose"
    so_key: Optional[str] = None  # shape key for .so lookup (None for native kernels)

    # Buffer bindings (indices into BufferManifest, -1 = not applicable)
    input_bufs: list[int] = field(default_factory=list)
    output_buf: int = -1
    weight_buf: int = -1
    scale_buf: int = -1
    bias_buf: int = -1
    membrane_buf: int = -1
    residual_buf: int = -1

    # Precomputed kernel parameters
    params: dict = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan — the contract between build and runtime."""
    buffers: list[BufferDesc]
    nodes: list[NodeExecPlan]
    schedule: list[int]
    T: int
    batch_size: int


# ─── ZeroCost ops that don't allocate buffers ───

_ZEROCOST_OPS = {OpType.Reshape, OpType.Transpose, OpType.Identity, OpType.Flatten}
_NHWC_OPS = {OpType.Conv2d, OpType.MaxPool, OpType.GlobalAvgPool,
             OpType.TemporalMean,
             OpType.Add, OpType.IF, OpType.LIF, OpType.MS,
             OpType.Tile, OpType.Sub, OpType.Mul, OpType.Scale,
             OpType.FusedAttention}
_GEMM_OPS = {OpType.MatMul, OpType.Linear}


def plan_buffers(ir: EngineIR, schedule: list[int],
                 T: int, batch_size: int,
                 kernel_so_map: dict = None,
                 precision: str = "fp16") -> ExecutionPlan:
    """Compute all buffer allocations and kernel bindings.

    Args:
        ir: Optimized EngineIR with kernel assignments.
        schedule: BA-MTTS execution order.
        T: Temporal steps.
        batch_size: Batch size.
        kernel_so_map: dict mapping nid → .so path (or tuple for fused attn).

    Returns:
        ExecutionPlan with all bindings resolved.
    """
    TB = T * batch_size
    B = batch_size
    bufs: list[BufferDesc] = []
    buf_map: dict[int, int] = {}  # nid → buf_id for activations
    weight_bufs: dict[str, int] = {}  # weight_key → buf_id
    membrane_bufs: dict[int, int] = {}  # neuron_nid → buf_id
    next_buf_id = 0

    def _alloc(shape, dtype, layout, category, source_nid=-1, weight_key=None):
        nonlocal next_buf_id
        bid = next_buf_id
        next_buf_id += 1
        bufs.append(BufferDesc(bid, tuple(shape), dtype, layout, category, source_nid, weight_key))
        return bid

    # ── Phase 1: Allocate weight/BN buffers ──
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue

        cp = node.conv_params
        if cp and node.op_type == OpType.Conv2d:
            # Weight buffer
            if cp.kernel_h == 1 and cp.kernel_w == 1:
                w_shape = (cp.in_channels, cp.out_channels)
                w_key = f"weight_1x1_{nid}"
                wbid = _alloc(w_shape, precision, "2d", "weight_1x1", nid, w_key)
            else:
                C_in_w = cp.in_channels
                # Stem conv (C_in < 4): pad to 16 for tensor core alignment
                if cp.in_channels < 4 and cp.groups == 1:
                    C_in_w = 16
                w_shape = (cp.kernel_h, cp.kernel_w, C_in_w, cp.out_channels)
                w_key = f"weight_{nid}"
                wbid = _alloc(w_shape, precision, "4d", "weight", nid, w_key)
            weight_bufs[f"weight_{nid}"] = wbid

            # BN scale/bias (stored as weight blobs with __bn_scale_/bias_ prefix)
            if node.bn_scale is not None:
                sbid = _alloc((cp.out_channels,), "fp32", "1d", "bn_scale", nid,
                              weight_key=f"__bn_scale_{nid}")
                weight_bufs[f"scale_{nid}"] = sbid
                bbid = _alloc((cp.out_channels,), "fp32", "1d", "bn_bias", nid,
                              weight_key=f"__bn_bias_{nid}")
                weight_bufs[f"bias_{nid}"] = bbid

        # Fused MatMul+LIF (uses conv1x1_bn_if/lif template with identity BN)
        elif node.assigned_kernel == KernelVariant.TileLangFusedMatMulLIF:
            if node.output_shapes:
                N_out = node.output_shapes[0][-1]
                # Weight is the second input predecessor — tracked as activation
                # buffer, not weight buffer. The plan Phase 4 will resolve it
                # from the second predecessor's activation buffer.
                # We still need dummy identity BN scale (ones) and bias (zeros).
                sbid = _alloc((N_out,), "fp32", "1d", "bn_scale", nid,
                              weight_key=f"__matmul_lif_scale_{nid}")
                weight_bufs[f"scale_{nid}"] = sbid
                bbid = _alloc((N_out,), "fp32", "1d", "bn_bias", nid,
                              weight_key=f"__matmul_lif_bias_{nid}")
                weight_bufs[f"bias_{nid}"] = bbid

        # Classifier Gemm (cuBLAS): weight (N_out, K) + optional bias (N_out,)
        elif node.op_type == OpType.Gemm and node.weight_info is not None:
            ws = tuple(node.weight_info.shape) if node.weight_info.shape else (1, 1)
            wbid = _alloc(ws, precision, "2d", "weight", nid, f"weight_{nid}")
            weight_bufs[f"weight_{nid}"] = wbid
            if node.bias_info is not None:
                bbid = _alloc((ws[0],), "fp32", "1d", "gemm_bias", nid,
                              weight_key=f"__gemm_bias_{nid}")
                weight_bufs[f"bias_{nid}"] = bbid

    # ── Phase 2: Allocate membrane buffers ──
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue
        if node.op_type not in (OpType.IF, OpType.LIF, OpType.MS):
            continue
        if not node.output_shapes:
            continue
        shape = node.output_shapes[0]
        if len(shape) == 4:
            N, C, H, W = shape
            Bm = max(N // T, 1)
            mem_shape = (Bm * H * W, C)
        elif len(shape) == 3:
            # 3D: (TB, spatial, channels) — from MatMul outputs
            TB_val, spatial, channels = shape
            Bm = max(TB_val // T, 1)
            mem_shape = (Bm * spatial, channels)
        elif len(shape) == 2:
            M, N_dim = shape
            spatial = max(M // T, 1)
            mem_shape = (spatial, N_dim)
        else:
            continue
        mbid = _alloc(mem_shape, "fp32", "2d", "membrane", nid)
        membrane_bufs[nid] = mbid

    # ── Phase 3: Allocate activation buffers ──
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None or not node.output_shapes:
            continue
        shape = node.output_shapes[0]

        # ZeroCost: no allocation (alias resolved later)
        if (node.op_type in _ZEROCOST_OPS
                and node.assigned_kernel != KernelVariant.LayoutTranspose):
            continue
        if node.assigned_kernel == KernelVariant.ZeroCost:
            continue

        # Determine shape and layout
        if len(shape) == 4 and node.op_type in _NHWC_OPS:
            N, C, H, W = shape
            alloc_shape = (N, H, W, C)
            layout = "nhwc"
        elif len(shape) == 4 and node.assigned_kernel == KernelVariant.LayoutTranspose:
            perm = node.extra_attrs.get("perm", [])
            N, C, H, W = shape
            if perm == [0, 2, 3, 1]:
                alloc_shape = (N, H, W, C); layout = "nhwc"
            elif perm == [0, 3, 1, 2]:
                alloc_shape = (N, C, H, W); layout = "nchw"
            else:
                alloc_shape = shape; layout = "nchw"
        elif node.op_type in _GEMM_OPS:
            total = 1
            for d in shape: total *= d
            N_out = shape[-1] if shape else 1
            M = total // N_out if N_out > 0 else total
            alloc_shape = (M, N_out)
            layout = "2d"
        else:
            alloc_shape = shape
            layout = "nd"

        bid = _alloc(alloc_shape, precision, layout, "activation", nid)
        buf_map[nid] = bid

    # ── Phase 3b: Alias absorbed nodes to their anchor's buffer ──
    # When fusion marks LIF/IF/Add as ZeroCost (absorbed into Conv), downstream
    # nodes still look for the absorbed node's output buffer. Create aliases
    # so buf_map[absorbed_nid] = buf_map[anchor_nid].
    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        if not absorbed or nid not in buf_map:
            continue
        anchor_buf = buf_map[nid]
        for ab_nid in absorbed:
            if ab_nid not in buf_map:
                buf_map[ab_nid] = anchor_buf

    # ── Phase 4: Resolve per-node kernel bindings ──
    nodes: list[NodeExecPlan] = []

    def _find_input_buf(nid, preds):
        """Find activation buffer from predecessors, tracing through ZeroCost."""
        for pid in preds:
            if pid in buf_map:
                return buf_map[pid]
        # Trace through ZeroCost
        for pid in preds:
            cur = pid
            for _ in range(10):
                pp = ir.predecessors(cur)
                if not pp:
                    break
                if pp[0] in buf_map:
                    return buf_map[pp[0]]
                cur = pp[0]
        return -1

    def _find_membrane(nid, absorbed_nids=None):
        """Find membrane buffer for a compute node's neuron successor."""
        # Direct successors
        for s in ir.successors(nid):
            if s in membrane_bufs:
                return membrane_bufs[s]
        # Walk absorbed chain
        if absorbed_nids:
            for ab in absorbed_nids:
                if ab in membrane_bufs:
                    return membrane_bufs[ab]
                for s in ir.successors(ab):
                    if s in membrane_bufs:
                        return membrane_bufs[s]
        return -1

    for nid in schedule:
        node = ir.nodes.get(nid)
        if node is None:
            nodes.append(NodeExecPlan(nid=nid, kernel_type="skip"))
            continue

        kv = node.assigned_kernel
        preds = ir.predecessors(nid)
        input_bid = _find_input_buf(nid, preds)
        output_bid = buf_map.get(nid, -1)
        so_key = kernel_so_map.get(nid) if kernel_so_map else None
        if isinstance(so_key, tuple):
            so_key = None  # fused attention handled separately

        # Weight/BN lookups
        w_bid = weight_bufs.get(f"weight_{nid}", -1)
        s_bid = weight_bufs.get(f"scale_{nid}", -1)
        b_bid = weight_bufs.get(f"bias_{nid}", -1)

        # Absorbed nodes (from fusion strategy)
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        mem_bid = _find_membrane(nid, absorbed)

        # Residual buffer (for Add fusion patterns)
        res_bid = -1
        res_nid = node.extra_attrs.get("residual_nid")
        if res_nid is not None and res_nid in buf_map:
            res_bid = buf_map[res_nid]

        # --- Dispatch by kernel variant ---

        if kv in (KernelVariant.TileLangConvBN, KernelVariant.TileLangConv1x1BN,
                  KernelVariant.TileLangStemConvBN, KernelVariant.TileLangDWConvBN,
                  KernelVariant.TileLangGroupedConvBN, KernelVariant.TileLangLinearBN):
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="tilelang_5", so_key=so_key,
                input_bufs=[input_bid], output_buf=output_bid,
                weight_buf=w_bid, scale_buf=s_bid, bias_buf=b_bid))

        elif kv in (KernelVariant.TileLangFusedConvBNIF,
                    KernelVariant.TileLangFusedConv1x1BNIF,
                    KernelVariant.TileLangLinearBNLIF,
                    KernelVariant.TileLangFusedGroupedConvBNLIF):
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="tilelang_6", so_key=so_key,
                input_bufs=[input_bid], output_buf=output_bid,
                weight_buf=w_bid, scale_buf=s_bid, bias_buf=b_bid,
                membrane_buf=mem_bid))

        elif kv == KernelVariant.TileLangFusedMatMulLIF:
            # MatMul+LIF uses conv1x1_bn_if/lif template with identity BN.
            # Weight is the second predecessor's activation buffer (constant weight).
            # Scale/bias are identity (allocated in Phase 1 above).
            matmul_w_bid = -1
            if len(preds) >= 2:
                matmul_w_bid = buf_map.get(preds[1], -1)
                if matmul_w_bid < 0:
                    matmul_w_bid = _find_input_buf(nid, [preds[1]])
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="tilelang_6", so_key=so_key,
                input_bufs=[input_bid], output_buf=output_bid,
                weight_buf=matmul_w_bid, scale_buf=s_bid, bias_buf=b_bid,
                membrane_buf=mem_bid))

        elif kv in (KernelVariant.TileLangMatMul, KernelVariant.TileLangMatMulScale):
            # Two input buffers (A, B)
            input_bids = []
            for pid in preds:
                bid = buf_map.get(pid, -1)
                if bid == -1:
                    bid = _find_input_buf(nid, [pid])
                if bid >= 0:
                    input_bids.append(bid)
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="tilelang_3", so_key=so_key,
                input_bufs=input_bids, output_buf=output_bid))

        elif kv in (KernelVariant.CUDAVec4IF, KernelVariant.CUDAVec4LIF):
            total = 0
            if node.output_shapes:
                for d in node.output_shapes[0]: total = total * d if total else d
            spatial = total // T if T > 0 else total
            v_thresh = node.neuron_params.v_threshold if node.neuron_params else 1.0
            params = {"total_elems": total, "spatial_elems": spatial,
                      "v_threshold": v_thresh}
            if kv == KernelVariant.CUDAVec4LIF:
                tau = node.neuron_params.tau if node.neuron_params and node.neuron_params.tau > 0 else 2.0
                params["recip_tau"] = 1.0 / tau
            nodes.append(NodeExecPlan(
                nid=nid,
                kernel_type="lif_neuron" if kv == KernelVariant.CUDAVec4LIF else "if_neuron",
                input_bufs=[input_bid], output_buf=output_bid,
                membrane_buf=membrane_bufs.get(nid, -1), params=params))

        elif kv == KernelVariant.Elementwise:
            if node.op_type == OpType.Add and len(preds) >= 2:
                a_bid = _find_input_buf(nid, [preds[0]])
                b_bid = _find_input_buf(nid, [preds[1]])
                n_elems = 0
                if node.output_shapes:
                    n_elems = 1
                    for d in node.output_shapes[0]: n_elems *= d
                nodes.append(NodeExecPlan(
                    nid=nid, kernel_type="add",
                    input_bufs=[a_bid, b_bid], output_buf=output_bid,
                    params={"n_elems": n_elems}))
            else:
                nodes.append(NodeExecPlan(
                    nid=nid, kernel_type="alias",
                    input_bufs=[input_bid], output_buf=output_bid))

        elif kv == KernelVariant.CuDNNPool:
            pp = node.pool_params or {}
            params = {}
            if node.op_type == OpType.MaxPool:
                params = {"kernel_size": pp.get("kernel_shape", [3,3])[0],
                          "stride": pp.get("strides", [2,2])[0],
                          "padding": pp.get("pads", [1,1,1,1])[0]}
                if node.output_shapes and len(node.output_shapes[0]) == 4:
                    params["OH"] = node.output_shapes[0][2]
                    params["OW"] = node.output_shapes[0][3]
                nodes.append(NodeExecPlan(
                    nid=nid, kernel_type="maxpool",
                    input_bufs=[input_bid], output_buf=output_bid, params=params))
            elif node.op_type == OpType.GlobalAvgPool:
                nodes.append(NodeExecPlan(
                    nid=nid, kernel_type="global_avgpool",
                    input_bufs=[input_bid], output_buf=output_bid))
            else:
                nodes.append(NodeExecPlan(nid=nid, kernel_type="skip"))

        elif kv == KernelVariant.TemporalMean:
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="temporal_mean",
                input_bufs=[input_bid], output_buf=output_bid,
                params={"T": T}))

        elif kv == KernelVariant.CuBLASGemm:
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="gemm",
                input_bufs=[input_bid], output_buf=output_bid,
                weight_buf=w_bid, bias_buf=b_bid))

        elif kv == KernelVariant.CuDNNConv:
            # Use cuDNN for large convolutions (validator-REVERT'd),
            # naive kernel for small/stem convolutions (C_in < 8 or non-standard)
            cp_node = node.conv_params if node else None
            use_cudnn = (cp_node and cp_node.in_channels >= 8
                         and cp_node.kernel_h <= 7 and cp_node.groups <= 1)
            kt = "cudnn_conv" if use_cudnn else "naive_conv"
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type=kt,
                input_bufs=[input_bid], output_buf=output_bid,
                weight_buf=w_bid, scale_buf=s_bid, bias_buf=b_bid))

        elif kv == KernelVariant.LayoutTranspose:
            perm = node.extra_attrs.get("perm", [])
            direction = 1 if perm == [0, 2, 3, 1] else 0
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="layout_transpose",
                input_bufs=[input_bid], output_buf=output_bid,
                params={"direction": direction}))

        elif kv == KernelVariant.ZeroCost:
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="skip",
                input_bufs=[input_bid], output_buf=output_bid))

        elif kv == KernelVariant.TileRepeat:
            nodes.append(NodeExecPlan(nid=nid, kernel_type="skip"))

        elif kv in (KernelVariant.TileLangFusedAddLIF, KernelVariant.TileLangFusedPoolLIF):
            # Fused Add+LIF or Pool+LIF (5-arg TileLang)
            input_bids = []
            for pid in preds:
                bid = buf_map.get(pid, -1)
                if bid == -1: bid = _find_input_buf(nid, [pid])
                if bid >= 0: input_bids.append(bid)
            lif_nid = node.extra_attrs.get("fused_lif_nid")
            mbid = membrane_bufs.get(lif_nid, -1) if lif_nid else -1
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="tilelang_5", so_key=so_key,
                input_bufs=input_bids, output_buf=output_bid,
                membrane_buf=mbid))

        elif kv in (KernelVariant.FusedSpikformerAttn, KernelVariant.FusedMaxformerAttn,
                    KernelVariant.FusedDSSAAttn, KernelVariant.FusedTokenQKAttn):
            # Fused attention: complex, keep simplified for now
            input_bids = [buf_map.get(pid, -1) for pid in preds if buf_map.get(pid, -1) >= 0]
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="fused_attn",
                input_bufs=input_bids, output_buf=output_bid,
                membrane_buf=membrane_bufs.get(nid, -1),
                params={"variant": node.attention_params.variant if node.attention_params else "unknown"}))

        elif kv == KernelVariant.CUDAResize:
            scales = node.extra_attrs.get("scales", [1, 1, 2, 2])
            sh = int(scales[2]) if len(scales) > 2 else 2
            sw = int(scales[3]) if len(scales) > 3 else 2
            # Get input shape from buffer manifest
            in_shape = ()
            for bd in bufs:
                if bd.buf_id == input_bid:
                    in_shape = bd.shape; break
            params = {"scale_h": sh, "scale_w": sw}
            if len(in_shape) == 4:
                params.update({"N": in_shape[0], "H": in_shape[1],
                               "W": in_shape[2], "C": in_shape[3]})
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="resize",
                input_bufs=[input_bid], output_buf=output_bid, params=params))

        elif kv == KernelVariant.CUDAConcat:
            # Concat needs multiple input buffers
            input_bids = []
            for pid in preds:
                bid = buf_map.get(pid, -1)
                if bid == -1: bid = _find_input_buf(nid, [pid])
                if bid >= 0: input_bids.append(bid)
            axis = node.extra_attrs.get("axis", 1)
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="concat",
                input_bufs=input_bids, output_buf=output_bid,
                params={"axis": axis}))

        elif kv == KernelVariant.CUDAVec4ILIF:
            total = 0
            if node.output_shapes:
                for d in node.output_shapes[0]: total = total * d if total else d
            spatial = total // T if T > 0 else total
            np_ = node.neuron_params
            params = {"total_elems": total, "spatial_elems": spatial,
                      "decay": np_.decay if np_ else 0.25,
                      "max_level": float(np_.max_level if np_ else 4)}
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="ilif_neuron",
                input_bufs=[input_bid], output_buf=output_bid,
                membrane_buf=membrane_bufs.get(nid, -1), params=params))

        elif node.op_type == OpType.Slice:
            # Slice produces a subset of the input — needs own output buffer
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="alias",
                input_bufs=[input_bid], output_buf=output_bid,
                params=node.extra_attrs))

        elif kv == KernelVariant.CUDASoftmax:
            axis = node.extra_attrs.get("axis", -1)
            in_shape = ()
            for bd in bufs:
                if bd.buf_id == input_bid:
                    in_shape = bd.shape; break
            # Compute outer/inner from shape and axis
            ndim = len(in_shape)
            ax = axis if axis >= 0 else ndim + axis
            outer = 1
            inner = 1
            for i, d in enumerate(in_shape):
                if i < ax: outer *= d
                elif i == ax: inner = d
                else: outer *= d  # flatten remaining dims into outer? No — inner is just the axis dim
            # Actually: outer = product of all dims except axis dim
            outer = 1
            for i, d in enumerate(in_shape):
                if i != ax: outer *= d
            inner = in_shape[ax] if 0 <= ax < ndim else 1
            nodes.append(NodeExecPlan(
                nid=nid, kernel_type="softmax",
                input_bufs=[input_bid], output_buf=output_bid,
                params={"outer": outer, "inner": inner}))

        else:
            nodes.append(NodeExecPlan(nid=nid, kernel_type="skip"))

    return ExecutionPlan(
        buffers=bufs,
        nodes=nodes,
        schedule=schedule,
        T=T,
        batch_size=batch_size,
    )


def print_plan(plan: ExecutionPlan):
    """Pretty-print the execution plan."""
    print(f"Execution Plan: {len(plan.buffers)} buffers, {len(plan.nodes)} nodes, "
          f"T={plan.T}, B={plan.batch_size}")

    # Buffer summary by category
    from collections import Counter
    cats = Counter(b.category for b in plan.buffers)
    print(f"  Buffers: {dict(cats)}")

    # Node summary by kernel type
    ktypes = Counter(n.kernel_type for n in plan.nodes)
    print(f"  Nodes: {dict(ktypes)}")

    # Check for unresolved bindings
    n_missing_input = sum(1 for n in plan.nodes if n.kernel_type != "skip" and not n.input_bufs)
    n_missing_output = sum(1 for n in plan.nodes if n.kernel_type != "skip" and n.output_buf == -1)
    n_missing_mem = sum(1 for n in plan.nodes
                        if n.kernel_type in ("tilelang_6", "if_neuron", "lif_neuron")
                        and n.membrane_buf == -1)
    if n_missing_input or n_missing_output or n_missing_mem:
        print(f"  WARNING: {n_missing_input} missing inputs, {n_missing_output} missing outputs, "
              f"{n_missing_mem} missing membranes")
    else:
        print(f"  All bindings resolved ✓")
