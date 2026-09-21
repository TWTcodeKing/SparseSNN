"""Save and load .sengine-cpu binary engine files.

Format:
  MAGIC (4 bytes): "SENC"
  Header length (4 bytes, uint32 LE)
  Header (JSON): metadata, node descriptors, schedule, kernel paths
  Weight count (4 bytes, uint32 LE)
  For each weight:
    Name length (4 bytes) + Name (bytes)
    Shape ndim (4 bytes) + Shape dims (ndim × 4 bytes each)
    Data length (4 bytes) + Data (raw float32 bytes)

Self-contained — no imports from sengine/.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np

from sengine_cpu.ir import (
    OpType, CPUKernelVariant, BoundType, DataLayout, NeuronType,
    ConvParams, NeuronParams, WeightInfo, Node, FusionGroup, EngineIR,
)
from sengine_cpu.logger import log

MAGIC = b"SENC"
VERSION = 1


def save_sengine_cpu(path: str, ir: EngineIR, schedule: list[int],
                      T: int, batch_size: int,
                      kernel_map: dict[int, str] | None = None):
    """Serialize optimized IR + schedule + weights to .sengine-cpu file."""
    header = {
        "magic": MAGIC.decode(),
        "version": VERSION,
        "T": T,
        "batch_size": batch_size,
        "precision": ir.precision,
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
            "fusion_group_id": node.fusion_group_id,
        }
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
        if node.neuron_params:
            np_ = node.neuron_params
            nd["neuron_params"] = {
                "neuron_type": np_.neuron_type.name, "T": np_.T,
                "tau": np_.tau, "v_threshold": np_.v_threshold,
                "v_reset": np_.v_reset,
            }
        if node.weight_info:
            nd["weight_name"] = node.weight_info.name
        if node.bn_scale is not None:
            nd["bn_scale"] = node.bn_scale
        if node.bn_bias is not None:
            nd["bn_bias"] = node.bn_bias
        if node.pool_params:
            nd["pool_params"] = node.pool_params
        if node.gemm_params:
            nd["gemm_params"] = node.gemm_params
        if node.extra_attrs:
            nd["extra_attrs"] = node.extra_attrs
        if kernel_map and nid in kernel_map:
            nd["so_path"] = kernel_map[nid]
        nodes_json.append(nd)

    header["nodes"] = nodes_json

    # Fusion groups
    header["fusion_groups"] = [
        {"group_id": fg.group_id, "conv_node_id": fg.conv_node_id,
         "neuron_node_id": fg.neuron_node_id,
         "variant": fg.variant.name}
        for fg in ir.fusion_groups
    ]

    # Encode header
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')

    # Write file
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(header_bytes)

        # Write weights
        weight_names = sorted(ir.weights.keys())
        f.write(struct.pack('<I', len(weight_names)))
        for name in weight_names:
            w = ir.weights[name].astype(np.float32)
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', w.ndim))
            for d in w.shape:
                f.write(struct.pack('<I', d))
            data = w.tobytes()
            f.write(struct.pack('<I', len(data)))
            f.write(data)

    log.info("Saved %s (%.1f MB, %d weights, %d nodes)",
             path, Path(path).stat().st_size / 1e6,
             len(weight_names), len(nodes_json))


def load_sengine_cpu(path: str) -> tuple[EngineIR, list[int], int, int]:
    """Load a .sengine-cpu file. Returns (ir, schedule, T, batch_size)."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Invalid magic: {magic}"
        header_len = struct.unpack('<I', f.read(4))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))

        # Load weights
        n_weights = struct.unpack('<I', f.read(4))[0]
        weights = {}
        for _ in range(n_weights):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            ndim = struct.unpack('<I', f.read(4))[0]
            shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
            data_len = struct.unpack('<I', f.read(4))[0]
            data = np.frombuffer(f.read(data_len), dtype=np.float32).reshape(shape)
            weights[name] = data

    # Reconstruct IR
    ir = EngineIR()
    ir.T = header["T"]
    ir.precision = header.get("precision", "fp32")
    ir.model_input_shape = tuple(header["model_input_shape"])
    ir.model_output_shape = tuple(header["model_output_shape"])
    ir.weights = weights

    kv_map = {v.name: v for v in CPUKernelVariant}
    nt_map = {v.name: v for v in NeuronType}

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
            assigned_kernel=kv_map.get(nd.get("assigned_kernel", "ZeroCost"),
                                       CPUKernelVariant.ZeroCost),
            bound_type=BoundType(nd.get("bound_type", "Z")),
            fusion_group_id=nd.get("fusion_group_id", -1),
        )
        if "conv_params" in nd:
            node.conv_params = ConvParams(**nd["conv_params"])
        if "neuron_params" in nd:
            np_d = nd["neuron_params"]
            node.neuron_params = NeuronParams(
                neuron_type=nt_map.get(np_d.get("neuron_type", "IF"), NeuronType.IF),
                T=np_d.get("T", 4),
                tau=np_d.get("tau", 1.0),
                v_threshold=np_d.get("v_threshold", 1.0),
                v_reset=np_d.get("v_reset", 0.0),
            )
        if "weight_name" in nd:
            node.weight_info = WeightInfo(name=nd["weight_name"])
        if "bn_scale" in nd:
            node.bn_scale = nd["bn_scale"]
        if "bn_bias" in nd:
            node.bn_bias = nd["bn_bias"]
        if "pool_params" in nd:
            node.pool_params = nd["pool_params"]
        if "gemm_params" in nd:
            node.gemm_params = nd["gemm_params"]
        if "extra_attrs" in nd:
            node.extra_attrs = nd["extra_attrs"]
        ir.add_node(node)

    # Fusion groups
    for fg_d in header.get("fusion_groups", []):
        ir.fusion_groups.append(FusionGroup(
            group_id=fg_d["group_id"],
            conv_node_id=fg_d["conv_node_id"],
            neuron_node_id=fg_d.get("neuron_node_id", -1),
            variant=kv_map.get(fg_d.get("variant", "TVMConvBNIF"),
                               CPUKernelVariant.TVMConvBNIF),
        ))

    ir.build_edges()
    ir.compute_topo_order()

    schedule = header["schedule"]
    T = header["T"]
    batch_size = header["batch_size"]

    log.info("Loaded %s: %d nodes, %d weights, T=%d, B=%d",
             path, len(ir.nodes), len(weights), T, batch_size)
    return ir, schedule, T, batch_size
