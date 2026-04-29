"""ONNX Parser for SEngine.

Parses plugin-mode ONNX files (containing FusedIFNeuron/FusedLIFNeuron/FusedMSNeuron
custom ops) into EngineIR.

Supported ops:
  Conv, BatchNormalization, FusedIFNeuron/FusedLIFNeuron/FusedMSNeuron,
  Add, MaxPool, GlobalAveragePool, Flatten, Gemm, Tile, ReduceMean,
  MatMul (Linear or attention), Transpose, Reshape, Mul/Scale,
  and shape-manipulation ops (Shape, Gather, Div, Cast, Unsqueeze, Concat).
"""

from __future__ import annotations
import numpy as np
import onnx
from onnx import numpy_helper

from sengine.ir import (
    OpType, NeuronType, KernelVariant, TensorLayout,
    ConvParams, NeuronParams, Node, Edge, WeightInfo, EngineIR,
)


# Map ONNX custom op names to our neuron types
_NEURON_OP_MAP = {
    "FusedIFNeuron": NeuronType.IF,
    "FusedLIFNeuron": NeuronType.LIF,
    "FusedMSNeuron": NeuronType.MS,
}

# ONNX ops that are part of the temporal-mean tail and can be collapsed
_TEMPORAL_MEAN_TAIL_OPS = {
    "Shape", "Gather", "Div", "Cast", "Unsqueeze", "Concat", "Reshape",
    "Constant", "ConstantOfShape", "Expand",
}


def _get_attr(onnx_node, name, default=None):
    """Extract attribute from ONNX node by name."""
    for attr in onnx_node.attribute:
        if attr.name == name:
            if attr.type == 1:   # FLOAT
                return attr.f
            elif attr.type == 2:  # INT
                return attr.i
            elif attr.type == 7:  # INTS
                return list(attr.ints)
            elif attr.type == 6:  # FLOATS
                return list(attr.floats)
            elif attr.type == 4:  # TENSOR
                return numpy_helper.to_array(attr.t)
    return default


def _get_attrs(onnx_node) -> dict:
    """Extract all attributes from ONNX node."""
    attrs = {}
    for attr in onnx_node.attribute:
        if attr.type == 1:
            attrs[attr.name] = attr.f
        elif attr.type == 2:
            attrs[attr.name] = attr.i
        elif attr.type == 7:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == 6:
            attrs[attr.name] = list(attr.floats)
    return attrs


class ONNXParser:
    """Parse plugin-mode ONNX into EngineIR."""

    def __init__(self, onnx_path: str):
        self.onnx_model = onnx.load(onnx_path)
        self.graph = self.onnx_model.graph

        # Build initializer lookup (name → numpy array)
        self._initializers: dict[str, np.ndarray] = {}
        for init in self.graph.initializer:
            self._initializers[init.name] = numpy_helper.to_array(init)

        # Also extract constants from Identity/Constant nodes that produce
        # BN running stats (not always in graph.initializer)
        self._resolve_constant_identity_nodes()

        # Build tensor shape map from value_info + graph inputs
        self._tensor_shapes: dict[str, tuple] = {}
        for vi in list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output):
            if vi.type.tensor_type.HasField("shape"):
                dims = tuple(
                    d.dim_value if d.dim_value > 0 else 0
                    for d in vi.type.tensor_type.shape.dim
                )
                self._tensor_shapes[vi.name] = dims

    def parse(self) -> EngineIR:
        """Parse the ONNX model into an EngineIR."""
        ir = EngineIR()

        # Detect T from Tile node
        ir.T = self._detect_T()

        # Model input/output shapes
        if self.graph.input:
            in_vi = self.graph.input[0]
            ir.model_input_shape = tuple(
                d.dim_value for d in in_vi.type.tensor_type.shape.dim
            )
        if self.graph.output:
            out_vi = self.graph.output[0]
            dims = out_vi.type.tensor_type.shape.dim
            ir.model_output_shape = tuple(d.dim_value for d in dims)

        # Pre-scan: identify nodes in the temporal-mean tail (Gemm→...→ReduceMean)
        # Only these specific nodes should be skipped, not ALL Reshape/Shape ops
        temporal_tail_names = self._find_temporal_mean_tail()

        # Parse nodes
        for onnx_node in self.graph.node:
            op_type = onnx_node.op_type
            node_name = onnx_node.name or onnx_node.output[0] if onnx_node.output else ""

            # Skip temporal-mean tail nodes (between classifier Gemm and ReduceMean)
            if node_name in temporal_tail_names:
                continue

            if op_type == "ReduceMean":
                gemm_out = getattr(self, '_gemm_output_name', None)
                input_names = [gemm_out] if gemm_out else [n for n in onnx_node.input if n]
                node = Node(
                    id=-1,
                    name=node_name or "temporal_mean",
                    op_type=OpType.TemporalMean,
                    input_names=input_names,
                    output_names=list(onnx_node.output),
                    assigned_kernel=KernelVariant.TemporalMean,
                )
                ir.add_node(node)
                continue

            if op_type == "Conv":
                self._parse_conv(onnx_node, ir)
            elif op_type == "BatchNormalization":
                self._parse_bn(onnx_node, ir)
            elif op_type in _NEURON_OP_MAP:
                self._parse_neuron(onnx_node, ir)
            elif op_type == "Add":
                self._parse_add(onnx_node, ir)
            elif op_type == "MaxPool":
                self._parse_maxpool(onnx_node, ir)
            elif op_type == "GlobalAveragePool":
                self._parse_globalavgpool(onnx_node, ir)
            elif op_type == "Flatten":
                self._parse_flatten(onnx_node, ir)
            elif op_type == "Gemm":
                self._parse_gemm(onnx_node, ir)
                self._gemm_output_name = onnx_node.output[0]
            elif op_type == "Tile":
                self._parse_tile(onnx_node, ir)
            elif op_type == "MatMul":
                self._parse_matmul(onnx_node, ir)
            elif op_type == "Transpose":
                self._parse_transpose(onnx_node, ir)
            elif op_type == "Reshape":
                self._parse_reshape(onnx_node, ir)
            elif op_type == "Mul":
                self._parse_mul(onnx_node, ir)
            elif op_type == "Split":
                self._parse_split(onnx_node, ir)
            elif op_type in _TEMPORAL_MEAN_TAIL_OPS:
                # Shape/Gather/Div/Cast/etc. that don't belong to temporal tail
                # and aren't structural (Reshape/Transpose/Mul handled above) — skip
                continue
            else:
                node = Node(
                    id=-1,
                    name=node_name or f"unknown_{op_type}",
                    op_type=OpType.Identity,
                    input_names=[n for n in onnx_node.input if n],
                    output_names=list(onnx_node.output),
                    extra_attrs={"original_op": op_type},
                )
                ir.add_node(node)

        # Load weights into IR
        ir.weights = self._initializers

        # Build edges and compute topo order
        ir.build_edges()
        ir.compute_topo_order()

        return ir

    def _detect_T(self) -> int:
        """Detect T from FusedIFNeuron/FusedLIFNeuron attributes or Tile repeat."""
        for node in self.graph.node:
            if node.op_type in _NEURON_OP_MAP:
                T = _get_attr(node, "T", None) or _get_attr(node, "T_i", None)
                if T and T > 0:
                    return int(T)
        # Fallback: look for Tile repeats constant
        return 4  # default

    def _parse_conv(self, onnx_node, ir: EngineIR):
        attrs = _get_attrs(onnx_node)
        kernel_shape = attrs.get("kernel_shape", [1, 1])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        dilations = attrs.get("dilations", [1, 1])
        groups = attrs.get("group", 1)

        # Get weight shape from initializer
        weight_name = onnx_node.input[1] if len(onnx_node.input) > 1 else ""
        weight_shape = ()
        if weight_name in self._initializers:
            weight_shape = self._initializers[weight_name].shape

        out_channels = int(weight_shape[0]) if weight_shape else 0
        in_channels = int(weight_shape[1]) * groups if weight_shape else 0

        conv_params = ConvParams(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_h=kernel_shape[0],
            kernel_w=kernel_shape[1] if len(kernel_shape) > 1 else kernel_shape[0],
            stride_h=strides[0],
            stride_w=strides[1] if len(strides) > 1 else strides[0],
            pad_h=pads[0],
            pad_w=pads[1] if len(pads) > 1 else pads[0],
            dilation_h=dilations[0],
            dilation_w=dilations[1] if len(dilations) > 1 else dilations[0],
            groups=groups,
        )

        # Output shape from value_info
        out_name = onnx_node.output[0]
        out_shape = self._tensor_shapes.get(out_name, ())

        weight_info = WeightInfo(
            name=weight_name,
            shape=weight_shape,
            onnx_idx=-1,
        )

        bias_info = None
        if len(onnx_node.input) > 2 and onnx_node.input[2]:
            bias_info = WeightInfo(
                name=onnx_node.input[2],
                shape=self._initializers.get(onnx_node.input[2], np.array([])).shape,
            )

        node = Node(
            id=-1,
            name=onnx_node.name or f"conv_{weight_name}",
            op_type=OpType.Conv2d,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            output_shapes=[out_shape] if out_shape else [],
            conv_params=conv_params,
            weight_info=weight_info,
            bias_info=bias_info,
            assigned_kernel=KernelVariant.CuDNNConv,
        )
        ir.add_node(node)

    def _parse_bn(self, onnx_node, ir: EngineIR):
        # BN has inputs: X, scale, bias, running_mean, running_var
        node = Node(
            id=-1,
            name=onnx_node.name or f"bn_{onnx_node.output[0]}",
            op_type=OpType.BatchNorm,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs=_get_attrs(onnx_node),
        )
        # Store BN parameter names for later folding
        if len(onnx_node.input) >= 5:
            node.extra_attrs["bn_scale_name"] = onnx_node.input[1]
            node.extra_attrs["bn_bias_name"] = onnx_node.input[2]
            node.extra_attrs["bn_mean_name"] = onnx_node.input[3]
            node.extra_attrs["bn_var_name"] = onnx_node.input[4]
        ir.add_node(node)

    def _parse_neuron(self, onnx_node, ir: EngineIR):
        neuron_type = _NEURON_OP_MAP[onnx_node.op_type]
        T = _get_attr(onnx_node, "T") or _get_attr(onnx_node, "T_i") or 4

        params = NeuronParams(neuron_type=neuron_type, T=int(T))

        if neuron_type == NeuronType.IF:
            params.tau = float('inf')
            params.v_threshold = _get_attr(onnx_node, "v_threshold_f",
                                           _get_attr(onnx_node, "v_threshold", 1.0))
            params.v_reset = _get_attr(onnx_node, "v_reset_f",
                                       _get_attr(onnx_node, "v_reset", 0.0))
            params.hard_reset = bool(_get_attr(onnx_node, "hard_reset_i",
                                               _get_attr(onnx_node, "hard_reset", 1)))
        elif neuron_type == NeuronType.LIF:
            params.tau = _get_attr(onnx_node, "tau_f",
                                   _get_attr(onnx_node, "tau", 2.0))
            params.v_threshold = _get_attr(onnx_node, "v_threshold_f",
                                           _get_attr(onnx_node, "v_threshold", 1.0))
            params.v_reset = _get_attr(onnx_node, "v_reset_f",
                                       _get_attr(onnx_node, "v_reset", 0.0))
            params.hard_reset = bool(_get_attr(onnx_node, "hard_reset_i",
                                               _get_attr(onnx_node, "hard_reset", 1)))
        elif neuron_type == NeuronType.MS:
            params.tau = _get_attr(onnx_node, "decay_f",
                                   _get_attr(onnx_node, "decay", 0.25))
            params.v_threshold = _get_attr(onnx_node, "thresh_f",
                                           _get_attr(onnx_node, "thresh", 0.5))

        node = Node(
            id=-1,
            name=onnx_node.name or f"neuron_{onnx_node.output[0]}",
            op_type=OpType.IF if neuron_type == NeuronType.IF
                    else OpType.LIF if neuron_type == NeuronType.LIF
                    else OpType.MS,
            is_stateful=True,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            neuron_params=params,
            assigned_kernel=KernelVariant.StandaloneLIF,
        )
        ir.add_node(node)

    def _parse_add(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1,
            name=onnx_node.name or f"add_{onnx_node.output[0]}",
            op_type=OpType.Add,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=KernelVariant.Elementwise,
        )
        ir.add_node(node)

    def _parse_maxpool(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1,
            name=onnx_node.name or "maxpool",
            op_type=OpType.MaxPool,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            pool_params=_get_attrs(onnx_node),
            assigned_kernel=KernelVariant.CuDNNPool,
        )
        ir.add_node(node)

    def _parse_globalavgpool(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1,
            name=onnx_node.name or "globalavgpool",
            op_type=OpType.GlobalAvgPool,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=KernelVariant.CuDNNPool,
        )
        ir.add_node(node)

    def _parse_flatten(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1,
            name=onnx_node.name or "flatten",
            op_type=OpType.Flatten,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=KernelVariant.ZeroCost,
            extra_attrs=_get_attrs(onnx_node),
        )
        ir.add_node(node)

    def _parse_gemm(self, onnx_node, ir: EngineIR):
        weight_name = onnx_node.input[1] if len(onnx_node.input) > 1 else ""
        bias_name = onnx_node.input[2] if len(onnx_node.input) > 2 else ""
        weight_shape = ()
        if weight_name in self._initializers:
            weight_shape = self._initializers[weight_name].shape

        node = Node(
            id=-1,
            name=onnx_node.name or "gemm_classifier",
            op_type=OpType.Gemm,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            gemm_params=_get_attrs(onnx_node),
            weight_info=WeightInfo(name=weight_name, shape=weight_shape),
            bias_info=WeightInfo(name=bias_name) if bias_name else None,
            assigned_kernel=KernelVariant.CuBLASGemm,
        )
        ir.add_node(node)

    def _parse_tile(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1,
            name=onnx_node.name or "tile_repeat",
            op_type=OpType.Tile,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=KernelVariant.TileRepeat,
        )
        ir.add_node(node)

    def _parse_matmul(self, onnx_node, ir: EngineIR):
        """Parse MatMul: either Linear (one input is weight) or attention matmul."""
        input_a = onnx_node.input[0] if len(onnx_node.input) > 0 else ""
        input_b = onnx_node.input[1] if len(onnx_node.input) > 1 else ""

        # Check if either input is a weight (initializer) → treat as Linear layer
        if input_b in self._initializers:
            weight_shape = self._initializers[input_b].shape
            node = Node(
                id=-1,
                name=onnx_node.name or f"linear_{onnx_node.output[0]}",
                op_type=OpType.Linear,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                weight_info=WeightInfo(name=input_b, shape=weight_shape),
                gemm_params={"N": int(weight_shape[-1]) if weight_shape else 0},
            )
        elif input_a in self._initializers:
            weight_shape = self._initializers[input_a].shape
            node = Node(
                id=-1,
                name=onnx_node.name or f"linear_{onnx_node.output[0]}",
                op_type=OpType.Linear,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                weight_info=WeightInfo(name=input_a, shape=weight_shape),
                gemm_params={"N": int(weight_shape[-1]) if weight_shape else 0},
            )
        else:
            # Both inputs are dynamic → attention matmul (QK^T or attn@V)
            node = Node(
                id=-1,
                name=onnx_node.name or f"matmul_{onnx_node.output[0]}",
                op_type=OpType.MatMul,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
            )
        ir.add_node(node)

    def _parse_transpose(self, onnx_node, ir: EngineIR):
        perm = _get_attr(onnx_node, "perm", None)
        node = Node(
            id=-1,
            name=onnx_node.name or f"transpose_{onnx_node.output[0]}",
            op_type=OpType.Transpose,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"perm": list(perm) if perm else []},
            assigned_kernel=KernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_reshape(self, onnx_node, ir: EngineIR):
        # Get target shape from input[1] (constant tensor)
        shape_name = onnx_node.input[1] if len(onnx_node.input) > 1 else ""
        target_shape = None
        if shape_name in self._initializers:
            target_shape = self._initializers[shape_name].astype(int).tolist()

        node = Node(
            id=-1,
            name=onnx_node.name or f"reshape_{onnx_node.output[0]}",
            op_type=OpType.Reshape,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"target_shape": target_shape},
            assigned_kernel=KernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_mul(self, onnx_node, ir: EngineIR):
        """Parse Mul: scalar constant → Scale, otherwise → element-wise Mul."""
        for inp_name in onnx_node.input:
            if inp_name in self._initializers:
                val = self._initializers[inp_name]
                if val.size == 1:
                    node = Node(
                        id=-1,
                        name=onnx_node.name or f"scale_{onnx_node.output[0]}",
                        op_type=OpType.Scale,
                        input_names=[n for n in onnx_node.input if n and n != inp_name],
                        output_names=list(onnx_node.output),
                        extra_attrs={"scale_value": float(val.flat[0])},
                        assigned_kernel=KernelVariant.Elementwise,
                    )
                    ir.add_node(node)
                    return
        # General element-wise multiply
        node = Node(
            id=-1,
            name=onnx_node.name or f"mul_{onnx_node.output[0]}",
            op_type=OpType.Mul,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=KernelVariant.Elementwise,
        )
        ir.add_node(node)

    def _parse_split(self, onnx_node, ir: EngineIR):
        """Parse Split op — creates multiple outputs."""
        node = Node(
            id=-1,
            name=onnx_node.name or f"split_{onnx_node.output[0]}",
            op_type=OpType.Identity,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"original_op": "Split", **_get_attrs(onnx_node)},
            assigned_kernel=KernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _resolve_constant_identity_nodes(self):
        """Resolve Identity nodes that just forward constants/initializers.

        In many ONNX exports, BN running_mean/running_var are stored as
        graph inputs → Identity nodes. This traces the chain and adds
        the resolved tensors to _initializers.
        """
        # Map: output_name → source initializer name
        identity_map: dict[str, str] = {}
        for node in self.graph.node:
            if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
                identity_map[node.output[0]] = node.input[0]
            elif node.op_type == "Constant" and len(node.output) == 1:
                # Extract constant value
                val = _get_attr(node, "value", None)
                if val is not None:
                    self._initializers[node.output[0]] = val

        # Resolve chains: Identity → Identity → ... → initializer
        for out_name, src_name in identity_map.items():
            # Follow the chain
            visited = set()
            current = src_name
            while current in identity_map and current not in visited:
                visited.add(current)
                current = identity_map[current]
            # If the final source is an initializer, add the output name too
            if current in self._initializers and out_name not in self._initializers:
                self._initializers[out_name] = self._initializers[current]

    def _find_temporal_mean_tail(self) -> set[str]:
        """Pre-scan to identify nodes in the classifier tail (Gemm→...→ReduceMean).

        Only these specific nodes are skipped during parsing, NOT all Reshape/Shape ops
        (which would break transformer attention blocks).
        """
        tail_names = set()

        # Find the last Gemm node (classifier) and trace forward to ReduceMean
        gemm_outputs = set()
        for node in reversed(self.graph.node):
            if node.op_type == "Gemm":
                gemm_outputs.update(node.output)
                break

        if not gemm_outputs:
            return tail_names

        # BFS forward from Gemm output to find all intermediate nodes until ReduceMean
        visited_tensors = set(gemm_outputs)
        for node in self.graph.node:
            if node.op_type in ("Gemm", "ReduceMean"):
                continue
            # Check if any input comes from the tail
            if any(inp in visited_tensors for inp in node.input):
                if node.op_type in _TEMPORAL_MEAN_TAIL_OPS:
                    name = node.name or node.output[0] if node.output else ""
                    tail_names.add(name)
                    visited_tensors.update(node.output)

        return tail_names
