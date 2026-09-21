"""Plan-driven executor setup: reads ExecutionPlan and configures C++ executor.

Replaces the 400-line _setup_cpp_executor() with a data-driven approach.
The ExecutionPlan (from BufferPlanner) contains all buffer IDs and kernel
params. This module resolves buffer IDs to GPU pointers and calls the
existing C++ set_*_node() functions.

Usage:
    from sengine_edge.runtime.plan_executor import setup_executor_from_plan
    exe = setup_executor_from_plan(plan, py_engine, kernel_so_map, ir)
"""

from __future__ import annotations

import torch

from sengine_edge.build.buffer_planner import ExecutionPlan, NodeExecPlan, BufferDesc
from sengine_edge.runtime.cpp_executor import CppExecutor
from sengine_edge.ir import EngineIR, OpType
from sengine_edge.logger import logger


def setup_executor_from_plan(
    plan: ExecutionPlan,
    py_engine,          # CUDAGraphEngine with allocated buffers
    kernel_so_map: dict,  # nid → .so path
    ir: EngineIR,
    attn_setup_fn=None,  # optional: callable(exe, nid, node, ...) for fused attention
) -> CppExecutor:
    """Configure C++ executor from a serialized execution plan.

    Resolves buffer IDs to GPU pointers using py_engine's allocated tensors,
    then calls set_*_node() for each node in the plan.

    Returns a ready-to-capture CppExecutor.
    """
    exe = CppExecutor()
    max_nid = max(n.nid for n in plan.nodes) if plan.nodes else 0
    exe.alloc_nodes(max_nid)
    exe.set_schedule(plan.schedule)

    # Set global precision flag
    exe.set_fp32(ir.precision == "fp32")

    # ── Build buffer ID → GPU pointer map ──
    buf_ptrs: dict[int, int] = {}  # buf_id → raw GPU pointer

    for bd in plan.buffers:
        ptr = 0
        if bd.category == "activation":
            t = py_engine.activations.get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        elif bd.category in ("weight", "weight_1x1"):
            if bd.category == "weight_1x1":
                t = py_engine.weights_1x1.get(bd.source_nid)
            else:
                t = py_engine.weights.get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        elif bd.category == "bn_scale":
            t = py_engine.bn_scales.get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        elif bd.category == "bn_bias":
            t = py_engine.bn_biases.get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        elif bd.category == "gemm_bias":
            t = getattr(py_engine, 'gemm_biases', {}).get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        elif bd.category == "membrane":
            t = py_engine.membranes.get(bd.source_nid)
            if t is not None:
                ptr = t.data_ptr()
        buf_ptrs[bd.buf_id] = ptr

    def p(buf_id: int) -> int:
        """Resolve buffer ID to raw GPU pointer."""
        return buf_ptrs.get(buf_id, 0)

    # ── Load TileLang .so files ──
    nid_tl_idx = {}
    for nid, so_path in kernel_so_map.items():
        if isinstance(so_path, tuple):
            continue  # fused attention loaded separately
        tl_idx = exe.load_tilelang(so_path)
        nid_tl_idx[nid] = tl_idx

    # ── Register membranes ──
    for bd in plan.buffers:
        if bd.category == "membrane":
            t = py_engine.membranes.get(bd.source_nid)
            if t is not None:
                exe.add_membrane(t.data_ptr(), t.numel())

    # ── Configure each node from the plan ──
    # Dummy tensors for padding args (Add+LIF 5-arg interface)
    _dummy = torch.zeros(1, dtype=torch.float32, device='cuda')
    _dummy_ptr = _dummy.data_ptr()

    for node_plan in plan.nodes:
        nid = node_plan.nid
        kt = node_plan.kernel_type
        tl_idx = nid_tl_idx.get(nid, -1)

        if kt == "tilelang_5":
            if tl_idx >= 0:
                in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
                out_ptr = p(node_plan.output_buf)
                w_ptr = p(node_plan.weight_buf)
                s_ptr = p(node_plan.scale_buf)
                b_ptr = p(node_plan.bias_buf)
                # For Add+LIF fused: input_bufs has 2 entries, membrane replaces scale
                if node_plan.membrane_buf >= 0 and len(node_plan.input_bufs) >= 2:
                    # 5-arg: input_a, input_b, membrane, dummy, output
                    a_ptr = p(node_plan.input_bufs[0])
                    b2_ptr = p(node_plan.input_bufs[1])
                    m_ptr = p(node_plan.membrane_buf)
                    exe.set_tilelang_5(nid, tl_idx, a_ptr, b2_ptr, m_ptr, _dummy_ptr, out_ptr)
                elif w_ptr and s_ptr and b_ptr:
                    exe.set_tilelang_5(nid, tl_idx, in_ptr, w_ptr, s_ptr, b_ptr, out_ptr)
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "tilelang_6":
            if tl_idx >= 0:
                in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
                w_ptr = p(node_plan.weight_buf)
                m_ptr = p(node_plan.membrane_buf)
                s_ptr = p(node_plan.scale_buf)
                b_ptr = p(node_plan.bias_buf)
                out_ptr = p(node_plan.output_buf)
                # Fused MatMul+LIF: weight comes from ir.weights via ZeroCost chain
                if not w_ptr and node_plan.weight_buf < 0:
                    node = ir.nodes.get(nid)
                    if node:
                        import numpy as np
                        for pid in ir.predecessors(nid):
                            pn = ir.nodes.get(pid)
                            if pn and pn.op_type in (OpType.Transpose, OpType.Reshape):
                                for iname in pn.input_names:
                                    w = ir.weights.get(iname)
                                    if w is not None:
                                        _w_dtype = torch.float32 if ir.precision == "fp32" else torch.float16
                                        wt = torch.from_numpy(w.copy()).to(_w_dtype).cuda() if isinstance(w, np.ndarray) else w.to(_w_dtype).cuda()
                                        perm = pn.extra_attrs.get("perm")
                                        if perm and len(perm) == wt.ndim:
                                            wt = wt.permute(*perm).contiguous()
                                        if not hasattr(py_engine, '_weight_keepalive'):
                                            py_engine._weight_keepalive = []
                                        py_engine._weight_keepalive.append(wt)
                                        w_ptr = wt.data_ptr()
                                        break
                                # Also check deeper: Reshape → Transpose chain
                                if not w_ptr:
                                    for gpid in ir.predecessors(pid):
                                        gpn = ir.nodes.get(gpid)
                                        if gpn:
                                            for iname in gpn.input_names:
                                                w = ir.weights.get(iname)
                                                if w is not None:
                                                    _w_dtype = torch.float32 if ir.precision == "fp32" else torch.float16
                                                    wt = torch.from_numpy(w.copy()).to(_w_dtype).cuda() if isinstance(w, np.ndarray) else w.to(_w_dtype).cuda()
                                                    # Apply grandparent's perm first, then parent's
                                                    gp_perm = gpn.extra_attrs.get("perm")
                                                    if gp_perm and len(gp_perm) == wt.ndim:
                                                        wt = wt.permute(*gp_perm).contiguous()
                                                    perm = pn.extra_attrs.get("perm")
                                                    if perm and len(perm) == wt.ndim:
                                                        wt = wt.permute(*perm).contiguous()
                                                    if not hasattr(py_engine, '_weight_keepalive'):
                                                        py_engine._weight_keepalive = []
                                                    py_engine._weight_keepalive.append(wt)
                                                    w_ptr = wt.data_ptr()
                                                    break
                                if w_ptr:
                                    break
                if w_ptr and m_ptr and s_ptr and b_ptr:
                    exe.set_tilelang_6(nid, tl_idx, in_ptr, w_ptr, m_ptr, s_ptr, b_ptr, out_ptr)
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "tilelang_3":
            if tl_idx >= 0 and len(node_plan.input_bufs) >= 2:
                a_ptr = p(node_plan.input_bufs[0])
                b_ptr = p(node_plan.input_bufs[1])
                out_ptr = p(node_plan.output_buf)
                exe.set_tilelang_3(nid, tl_idx, a_ptr, b_ptr, out_ptr)
            elif tl_idx >= 0 and len(node_plan.input_bufs) == 1:
                # Missing second input — likely a MatMul whose weight comes
                # through a ZeroCost Transpose. Resolve from ir.weights.
                a_ptr = p(node_plan.input_bufs[0])
                out_ptr = p(node_plan.output_buf)
                node = ir.nodes.get(nid)
                b_ptr = 0
                if node:
                    import numpy as np
                    for pid in ir.predecessors(nid):
                        pn = ir.nodes.get(pid)
                        if pn and pn.op_type == OpType.Transpose:
                            perm = pn.extra_attrs.get("perm")
                            for iname in pn.input_names:
                                w = ir.weights.get(iname)
                                if w is not None:
                                    _w_dtype = torch.float32 if ir.precision == "fp32" else torch.float16
                                    wt = torch.from_numpy(w.copy()).to(_w_dtype).cuda() if isinstance(w, np.ndarray) else w.to(_w_dtype).cuda()
                                    if perm and len(perm) == wt.ndim:
                                        wt = wt.permute(*perm).contiguous()
                                    # Store on a keep-alive list (NOT in activations —
                                    # the ZeroCost dispatch would double-transpose it)
                                    if not hasattr(py_engine, '_weight_keepalive'):
                                        py_engine._weight_keepalive = []
                                    py_engine._weight_keepalive.append(wt)
                                    b_ptr = wt.data_ptr()
                                    break
                if b_ptr:
                    exe.set_tilelang_3(nid, tl_idx, a_ptr, b_ptr, out_ptr)
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "if_neuron":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            m_ptr = p(node_plan.membrane_buf)
            params = node_plan.params
            if in_ptr and out_ptr and m_ptr:
                exe.set_if_node(nid, in_ptr, out_ptr, m_ptr,
                                params.get("total_elems", 0),
                                params.get("spatial_elems", 0),
                                params.get("v_threshold", 1.0))
            else:
                exe.set_skip_node(nid)

        elif kt == "lif_neuron":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            m_ptr = p(node_plan.membrane_buf)
            params = node_plan.params
            if in_ptr and out_ptr and m_ptr:
                exe.set_lif_node(nid, in_ptr, out_ptr, m_ptr,
                                 params.get("total_elems", 0),
                                 params.get("spatial_elems", 0),
                                 params.get("v_threshold", 1.0),
                                 params.get("recip_tau", 0.5))
            else:
                exe.set_skip_node(nid)

        elif kt == "add":
            if len(node_plan.input_bufs) >= 2:
                a_ptr = p(node_plan.input_bufs[0])
                b_ptr = p(node_plan.input_bufs[1])
                out_ptr = p(node_plan.output_buf)
                n_elems = node_plan.params.get("n_elems", 0)
                if a_ptr and b_ptr and out_ptr:
                    exe.set_add_node(nid, a_ptr, b_ptr, out_ptr, n_elems)
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "maxpool":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            params = node_plan.params
            node = ir.nodes.get(nid)
            if in_ptr and out_ptr and node and node.output_shapes:
                # Get input shape from buffer manifest (NOT py_engine.activations)
                shape = (0, 0, 0, 0)
                in_bid = node_plan.input_bufs[0] if node_plan.input_bufs else -1
                for bd in plan.buffers:
                    if bd.buf_id == in_bid:
                        shape = bd.shape
                        break
                if len(shape) == 4:
                    exe.set_maxpool_node(nid, in_ptr, out_ptr,
                                         shape[0], shape[1], shape[2], shape[3],
                                         params.get("OH", 0), params.get("OW", 0),
                                         params.get("kernel_size", 3),
                                         params.get("stride", 2),
                                         params.get("padding", 1))
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "global_avgpool":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            for bd in plan.buffers:
                if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                    if len(bd.shape) == 4:
                        exe.set_global_avgpool_node(nid, in_ptr, out_ptr, *bd.shape)
                    else:
                        exe.set_skip_node(nid)
                    break
            else:
                exe.set_skip_node(nid)

        elif kt == "temporal_mean":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            T_val = node_plan.params.get("T", plan.T)
            for bd in plan.buffers:
                if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                    total = 1
                    for d in bd.shape: total *= d
                    spatial = total // T_val if T_val > 0 else total
                    exe.set_temporal_mean_node(nid, in_ptr, out_ptr, T_val, spatial)
                    break
            else:
                exe.set_skip_node(nid)

        elif kt == "gemm":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            w_ptr = p(node_plan.weight_buf)
            out_ptr = p(node_plan.output_buf)
            if in_ptr and w_ptr and out_ptr:
                # Find shapes from buffer manifest
                in_shape = out_shape = w_shape = ()
                for bd in plan.buffers:
                    if bd.buf_id == node_plan.input_bufs[0]: in_shape = bd.shape
                    if bd.buf_id == node_plan.weight_buf: w_shape = bd.shape
                M = in_shape[0] if len(in_shape) >= 1 else 1
                K = in_shape[-1] if len(in_shape) >= 2 else 1
                N = w_shape[0] if len(w_shape) >= 1 else 1
                b_ptr = p(node_plan.bias_buf) if node_plan.bias_buf is not None and node_plan.bias_buf >= 0 else 0
                exe.set_gemm_node(nid, in_ptr, w_ptr, out_ptr, M, K, N, b_ptr)
            else:
                exe.set_skip_node(nid)

        elif kt == "naive_conv":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            w_ptr = p(node_plan.weight_buf)
            s_ptr = p(node_plan.scale_buf)
            b_ptr = p(node_plan.bias_buf)
            out_ptr = p(node_plan.output_buf)
            node = ir.nodes.get(nid)
            if node and node.conv_params and in_ptr and w_ptr and out_ptr and s_ptr and b_ptr:
                cp = node.conv_params
                for bd in plan.buffers:
                    if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                        s = bd.shape; break
                else:
                    s = (0, 0, 0, 0)
                OH = (s[1] + 2*cp.pad_h - cp.kernel_h) // cp.stride_h + 1 if len(s) == 4 else 0
                OW = (s[2] + 2*cp.pad_w - cp.kernel_w) // cp.stride_w + 1 if len(s) == 4 else 0
                exe.set_naive_conv_node(nid, in_ptr, w_ptr, s_ptr, b_ptr, out_ptr,
                                        s[0], s[1], s[2], cp.in_channels, cp.out_channels,
                                        cp.kernel_h, cp.kernel_w, cp.stride_h, cp.pad_h, OH, OW,
                                        cp.groups)
            else:
                exe.set_skip_node(nid)

        elif kt == "layout_transpose":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            direction = node_plan.params.get("direction", 0)
            for bd in plan.buffers:
                if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                    s = bd.shape
                    if len(s) == 4:
                        if direction == 0:
                            exe.set_layout_transpose_node(nid, in_ptr, out_ptr, s[0], s[1], s[2], s[3], direction)
                        else:
                            exe.set_layout_transpose_node(nid, in_ptr, out_ptr, s[0], s[2], s[3], s[1], direction)
                    else:
                        exe.set_skip_node(nid)
                    break
            else:
                exe.set_skip_node(nid)

        elif kt == "alias":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            if in_ptr and out_ptr and in_ptr != out_ptr:
                for bd in plan.buffers:
                    if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                        n = 1
                        for d in bd.shape: n *= d
                        exe.set_alias_node(nid, in_ptr, out_ptr, n)
                        break
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "fused_attn":
            # Fused attention: delegate to the legacy attention setup handler
            if attn_setup_fn is not None:
                try:
                    attn_setup_fn(exe, nid, nid_tl_idx, buf_ptrs, plan)
                except Exception as e:
                    logger.warning("  Plan executor: fused_attn #%d setup failed: %s", nid, e)
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)
                logger.warning("  Plan executor: fused_attn #%d skipped (no handler)", nid)

        elif kt == "cudnn_conv":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            w_ptr = p(node_plan.weight_buf)
            s_ptr = p(node_plan.scale_buf)
            b_ptr = p(node_plan.bias_buf)
            out_ptr = p(node_plan.output_buf)
            node = ir.nodes.get(nid)
            if node and node.conv_params and in_ptr and w_ptr and out_ptr:
                cp = node.conv_params
                for bd in plan.buffers:
                    if bd.buf_id == (node_plan.input_bufs[0] if node_plan.input_bufs else -1):
                        s = bd.shape; break
                else:
                    s = (0, 0, 0, 0)
                OH = (s[1] + 2*cp.pad_h - cp.kernel_h) // cp.stride_h + 1 if len(s) == 4 else 0
                OW = (s[2] + 2*cp.pad_w - cp.kernel_w) // cp.stride_w + 1 if len(s) == 4 else 0
                # cuDNN expects NCHW filter (C_out, C_in/g, K, K).
                # Our weight is NHWC (K, K, C_in/g, C_out). Transpose to NCHW.
                w_tensor = py_engine.weights.get(nid)
                if w_tensor is not None and w_tensor.ndim == 4:
                    # (K, K, C_in/g, C_out) → (C_out, C_in/g, K, K)
                    w_nchw = w_tensor.permute(3, 2, 0, 1).contiguous()
                    if not hasattr(py_engine, '_cudnn_weight_keepalive'):
                        py_engine._cudnn_weight_keepalive = []
                    py_engine._cudnn_weight_keepalive.append(w_nchw)
                    w_ptr = w_nchw.data_ptr()
                exe.set_cudnn_conv_node(nid, in_ptr, w_ptr, s_ptr or 0, b_ptr or 0, out_ptr,
                                         s[0], s[1], s[2], cp.in_channels, cp.out_channels,
                                         cp.kernel_h, cp.kernel_w, cp.stride_h, cp.pad_h,
                                         OH, OW, cp.groups)
            else:
                exe.set_skip_node(nid)

        elif kt == "resize":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            params = node_plan.params
            N = params.get("N", 1); H = params.get("H", 1)
            W = params.get("W", 1); C = params.get("C", 1)
            sh = params.get("scale_h", 2); sw = params.get("scale_w", 2)
            OH = H * sh; OW = W * sw
            if in_ptr and out_ptr:
                exe.set_resize_node(nid, in_ptr, out_ptr, N, H, W, C, OH, OW, sh, sw)
            else:
                exe.set_skip_node(nid)

        elif kt == "concat":
            out_ptr = p(node_plan.output_buf)
            if len(node_plan.input_bufs) >= 2:
                a_ptr = p(node_plan.input_bufs[0])
                b_ptr = p(node_plan.input_bufs[1])
                # Get channel dims from buffer shapes
                Ca = Cb = 0
                for bd in plan.buffers:
                    if bd.buf_id == node_plan.input_bufs[0]:
                        Ca = bd.shape[-1] if bd.shape else 0
                    if bd.buf_id == node_plan.input_bufs[1]:
                        Cb = bd.shape[-1] if bd.shape else 0
                NHW = 1
                for bd in plan.buffers:
                    if bd.buf_id == node_plan.input_bufs[0] and len(bd.shape) >= 2:
                        for d in bd.shape[:-1]: NHW *= d
                        break
                if a_ptr and b_ptr and out_ptr and Ca and Cb:
                    exe.set_concat_node(nid, a_ptr, b_ptr, out_ptr, NHW, Ca, Cb)
                else:
                    exe.set_skip_node(nid)
            else:
                exe.set_skip_node(nid)

        elif kt == "ilif_neuron":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            m_ptr = p(node_plan.membrane_buf)
            params = node_plan.params
            if in_ptr and out_ptr and m_ptr:
                exe.set_ilif_node(nid, in_ptr, out_ptr, m_ptr,
                                  params.get("total_elems", 0),
                                  params.get("spatial_elems", 0),
                                  params.get("decay", 0.25),
                                  params.get("max_level", 4.0))
            else:
                exe.set_skip_node(nid)

        elif kt == "softmax":
            in_ptr = p(node_plan.input_bufs[0]) if node_plan.input_bufs else 0
            out_ptr = p(node_plan.output_buf)
            params = node_plan.params
            if in_ptr and out_ptr:
                exe.set_softmax_node(nid, in_ptr, out_ptr,
                                     params.get("outer", 1), params.get("inner", 1))
            else:
                exe.set_skip_node(nid)

        elif kt == "skip":
            exe.set_skip_node(nid)

        else:
            logger.warning("  Plan executor: unknown kernel_type '%s' for #%d", kt, nid)
            exe.set_skip_node(nid)

    # ── L2 cache persistence for weight buffers (Orin optimization) ──
    # Collect all weight buffer pointers and sizes for L2 pinning.
    weight_ptrs = []
    weight_sizes = []
    for bd in plan.buffers:
        if bd.category in ("weight", "weight_1x1"):
            ptr = buf_ptrs.get(bd.buf_id, 0)
            if ptr:
                nbytes = 1
                for d in bd.shape:
                    nbytes *= d
                nbytes *= 2  # FP16 = 2 bytes (FP32 = 4 bytes if precision is fp32)
                if ir.precision == "fp32":
                    nbytes *= 2
                weight_ptrs.append(ptr)
                weight_sizes.append(nbytes)
    import os as _os
    _on_orin = 'Orin' in torch.cuda.get_device_name(0) or _os.environ.get('SENGINE_EDGE_L2_PERSIST', '0') == '1'
    if weight_ptrs and _on_orin:
        try:
            exe.setup_l2_persistence(weight_ptrs, weight_sizes)
        except Exception as e:
            logger.warning("  L2 persistence setup failed: %s", e)

    return exe
