"""Save and load .sengine binary engine files.

Format:
  Header (JSON): model metadata, T, batch_size, schedule, tuning configs
  Weights (binary): pre-transposed NHWC FP16 weight tensors + BN params

The .sengine file stores everything needed to reconstruct the engine
WITHOUT re-parsing ONNX or re-autotuning. Kernels are recompiled from
cached tile configs on load (~5-10s vs ~60-120s for autotuning).

Usage:
    save_sengine(engine, ir, schedule, "model.sengine")
    engine, ir, schedule = load_sengine("model.sengine")
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from sengine.ir import (
    OpType, KernelVariant, BoundType, TensorLayout, NeuronType,
    ConvParams, NeuronParams, WeightInfo, Node, Edge, FusionGroup, EngineIR,
)
from sengine.logger import logger


# ─── Magic and version ───
MAGIC = b"SENG"
VERSION = 2


def save_sengine(path: str, ir: EngineIR, schedule: list[int],
                 T: int, batch_size: int):
    """Serialize an optimized EngineIR + schedule to a .sengine file.

    The file contains:
    1. JSON header: metadata, node descriptors, edges, schedule, tuning configs
    2. Binary weight blobs: pre-converted numpy arrays

    Args:
        path: Output file path (should end with .sengine)
        ir: Optimized EngineIR with shapes, bound types, BN params, weights
        schedule: BA-MTTS execution order (list of node IDs)
        T: Temporal steps
        batch_size: Batch size
    """
    # ─── Build JSON header ───
    header = {
        "magic": MAGIC.decode(),
        "version": VERSION,
        "T": T,
        "batch_size": batch_size,
        "model_input_shape": list(ir.model_input_shape),
        "model_output_shape": list(ir.model_output_shape),
        "schedule": schedule,
    }

    # Serialize nodes
    nodes_json = []
    for nid in sorted(ir.nodes.keys()):
        node = ir.nodes[nid]
        nd = {
            "id": node.id,
            "name": node.name,
            "op_type": node.op_type.name,
            "is_stateful": node.is_stateful,
            "input_names": node.input_names,
            "output_names": node.output_names,
            "input_shapes": [list(s) for s in node.input_shapes],
            "output_shapes": [list(s) for s in node.output_shapes],
            "assigned_kernel": node.assigned_kernel.name,
            "bound_type": node.bound_type.value,
            "layout": node.layout.name,
            "fusion_group_id": node.fusion_group_id,
            "sparse_weight": node.sparse_weight,
        }
        # Conv params
        if node.conv_params:
            cp = node.conv_params
            nd["conv_params"] = {
                "in_channels": cp.in_channels, "out_channels": cp.out_channels,
                "kernel_h": cp.kernel_h, "kernel_w": cp.kernel_w,
                "stride_h": cp.stride_h, "stride_w": cp.stride_w,
                "pad_h": cp.pad_h, "pad_w": cp.pad_w,
                "dilation_h": cp.dilation_h, "dilation_w": cp.dilation_w,
                "groups": cp.groups,
            }
        # Neuron params
        if node.neuron_params:
            np_ = node.neuron_params
            nd["neuron_params"] = {
                "neuron_type": np_.neuron_type.name, "T": np_.T,
                "tau": np_.tau, "v_threshold": np_.v_threshold,
                "v_reset": np_.v_reset, "hard_reset": np_.hard_reset,
            }
        # Pool params
        if node.pool_params:
            nd["pool_params"] = node.pool_params
        # Gemm params
        if node.gemm_params:
            nd["gemm_params"] = node.gemm_params
        # BN params
        if node.bn_scale is not None:
            nd["bn_scale"] = node.bn_scale
        if node.bn_bias is not None:
            nd["bn_bias"] = node.bn_bias
        # Weight reference
        if node.weight_info:
            nd["weight_name"] = node.weight_info.name
            nd["weight_shape"] = list(node.weight_info.shape)
            nd["weight_sparse"] = node.weight_info.is_sparse
        # Tilelang config
        if node.tilelang_config:
            nd["tilelang_config"] = node.tilelang_config
        nd["est_latency_us"] = node.est_latency_us

        nodes_json.append(nd)

    header["nodes"] = nodes_json

    # Serialize edges
    edges_json = []
    for edge in ir.edges:
        edges_json.append({
            "src_id": edge.src_id, "dst_id": edge.dst_id,
            "tensor_name": edge.tensor_name,
            "tensor_shape": list(edge.tensor_shape),
            "tensor_bytes": edge.tensor_bytes,
        })
    header["edges"] = edges_json

    # Serialize fusion groups
    fg_json = []
    for fg in ir.fusion_groups:
        fg_json.append({
            "group_id": fg.group_id,
            "conv_node_id": fg.conv_node_id,
            "neuron_node_id": fg.neuron_node_id,
        })
    header["fusion_groups"] = fg_json

    # ─── Build weight blobs ───
    weight_names = []
    weight_blobs = []
    for name, arr in ir.weights.items():
        blob = arr.astype(np.float32).tobytes()
        weight_names.append(name)
        weight_blobs.append(blob)

    header["weight_manifest"] = [
        {"name": name, "dtype": "float32", "nbytes": len(blob)}
        for name, blob in zip(weight_names, weight_blobs)
    ]

    # ─── Write file ───
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_bytes)

    with open(path, 'wb') as f:
        # Magic + version + header length
        f.write(MAGIC)
        f.write(struct.pack('<II', VERSION, header_len))
        # JSON header
        f.write(header_bytes)
        # Weight blobs (sequential)
        for blob in weight_blobs:
            f.write(blob)

    total_bytes = 12 + header_len + sum(len(b) for b in weight_blobs)
    logger.phase("SAVE", "Saved .sengine: %d nodes, %d weights, %.1f MB → %s",
                 len(nodes_json), len(weight_names), total_bytes / 1e6, path)


def load_sengine(path: str) -> tuple[EngineIR, list[int], int, int]:
    """Load a .sengine file and reconstruct EngineIR + schedule.

    Returns:
        (ir, schedule, T, batch_size)

    The caller should then:
    1. Create TileLangCompiler(ir, T, batch_size) with tuning configs from ir
    2. Call compile_all() to get kernels (fast, uses cached configs)
    3. Create CUDAGraphEngine and capture graph
    """
    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        assert magic == MAGIC, f"Invalid magic: {magic}"
        version, header_len = struct.unpack('<II', f.read(8))
        assert version == VERSION, f"Unsupported version: {version}"

        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))

        # Read weight blobs
        weight_data: dict[str, np.ndarray] = {}
        for wm in header.get("weight_manifest", []):
            blob = f.read(wm["nbytes"])
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            weight_data[wm["name"]] = arr

    # ─── Reconstruct EngineIR ───
    ir = EngineIR()
    ir.T = header["T"]
    ir.model_input_shape = tuple(header["model_input_shape"])
    ir.model_output_shape = tuple(header["model_output_shape"])
    ir.weights = {}

    # Reconstruct weight arrays with correct shapes
    for nd in header["nodes"]:
        if "weight_name" in nd and nd["weight_name"] in weight_data:
            wname = nd["weight_name"]
            wshape = tuple(nd["weight_shape"])
            if wname not in ir.weights:
                flat = weight_data[wname]
                if flat.size == int(np.prod(wshape)):
                    ir.weights[wname] = flat.reshape(wshape)
                else:
                    ir.weights[wname] = flat

    # Add remaining weights (BN params, folded biases, etc.)
    for wm in header.get("weight_manifest", []):
        if wm["name"] not in ir.weights:
            ir.weights[wm["name"]] = weight_data.get(wm["name"], np.array([]))

    # Reconstruct nodes
    for nd in header["nodes"]:
        node = Node(
            id=nd["id"],
            name=nd["name"],
            op_type=OpType[nd["op_type"]],
            is_stateful=nd.get("is_stateful", False),
            input_names=nd.get("input_names", []),
            output_names=nd.get("output_names", []),
            input_shapes=[tuple(s) for s in nd.get("input_shapes", [])],
            output_shapes=[tuple(s) for s in nd.get("output_shapes", [])],
            assigned_kernel=KernelVariant[nd["assigned_kernel"]],
            bound_type=BoundType(nd["bound_type"]),
            layout=TensorLayout[nd.get("layout", "NCHW")],
            fusion_group_id=nd.get("fusion_group_id", -1),
            sparse_weight=nd.get("sparse_weight", False),
            est_latency_us=nd.get("est_latency_us", 0.0),
        )
        # Conv params
        if "conv_params" in nd:
            cp = nd["conv_params"]
            node.conv_params = ConvParams(**cp)
        # Neuron params
        if "neuron_params" in nd:
            npp = nd["neuron_params"]
            node.neuron_params = NeuronParams(
                neuron_type=NeuronType[npp["neuron_type"]],
                T=npp["T"], tau=npp["tau"],
                v_threshold=npp["v_threshold"], v_reset=npp["v_reset"],
                hard_reset=npp["hard_reset"],
            )
        # Pool/Gemm params
        if "pool_params" in nd:
            node.pool_params = nd["pool_params"]
        if "gemm_params" in nd:
            node.gemm_params = nd["gemm_params"]
        # BN params
        if "bn_scale" in nd:
            node.bn_scale = nd["bn_scale"]
        if "bn_bias" in nd:
            node.bn_bias = nd["bn_bias"]
        # Weight info
        if "weight_name" in nd:
            node.weight_info = WeightInfo(
                name=nd["weight_name"],
                shape=tuple(nd.get("weight_shape", ())),
                is_sparse=nd.get("weight_sparse", False),
            )
        # Tilelang config
        if "tilelang_config" in nd:
            node.tilelang_config = nd["tilelang_config"]

        # Insert with original ID (don't use ir.add_node which auto-assigns)
        original_id = node.id
        ir.nodes[original_id] = node
        ir._adj[original_id] = []
        ir._radj[original_id] = []
        for out_name in node.output_names:
            ir._tensor_producer[out_name] = original_id
        for in_name in node.input_names:
            if in_name not in ir._tensor_consumers:
                ir._tensor_consumers[in_name] = []
            ir._tensor_consumers[in_name].append(original_id)
        ir._next_id = max(ir._next_id, original_id + 1)

    # Reconstruct edges
    for ed in header.get("edges", []):
        edge = Edge(
            src_id=ed["src_id"], dst_id=ed["dst_id"],
            tensor_name=ed.get("tensor_name", ""),
            tensor_shape=tuple(ed.get("tensor_shape", ())),
            tensor_bytes=ed.get("tensor_bytes", 0),
        )
        ir.edges.append(edge)

    # Rebuild adjacency from edges
    ir._adj = {nid: [] for nid in ir.nodes}
    ir._radj = {nid: [] for nid in ir.nodes}
    for edge in ir.edges:
        if edge.src_id in ir.nodes and edge.dst_id in ir.nodes:
            if edge.dst_id not in ir._adj[edge.src_id]:
                ir._adj[edge.src_id].append(edge.dst_id)
            if edge.src_id not in ir._radj[edge.dst_id]:
                ir._radj[edge.dst_id].append(edge.src_id)

    # Reconstruct fusion groups
    for fg_d in header.get("fusion_groups", []):
        fg = FusionGroup(
            group_id=fg_d["group_id"],
            conv_node_id=fg_d["conv_node_id"],
            neuron_node_id=fg_d["neuron_node_id"],
        )
        ir.fusion_groups.append(fg)

    ir.compute_topo_order()

    schedule = header["schedule"]
    T = header["T"]
    batch_size = header["batch_size"]

    logger.phase("LOAD", "Loaded .sengine: %d nodes, %d edges, %d weights from %s",
                 len(ir.nodes), len(ir.edges), len(ir.weights), path)

    return ir, schedule, T, batch_size
