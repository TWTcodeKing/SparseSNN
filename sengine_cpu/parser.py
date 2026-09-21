"""ONNX Parser for sengine_cpu.

Parses ONNX files (with FusedIFNeuron/FusedLIFNeuron/FusedMSNeuron/FusedILIFNeuron
custom ops) into EngineIR. Self-contained — no imports from sengine/.

Supported ops:
  Conv, BatchNormalization, FusedIFNeuron/FusedLIFNeuron/FusedMSNeuron/FusedILIFNeuron,
  Add, MaxPool, GlobalAveragePool, Flatten, Gemm, Tile, ReduceMean,
  MatMul (Linear or attention), Transpose, Reshape, Mul/Scale, Split,
  Resize, Concat, Softmax, Slice,
  and shape-manipulation ops (Shape, Gather, Div, Cast, Unsqueeze, Concat).
"""

from __future__ import annotations
import numpy as np
import onnx
from onnx import numpy_helper

from sengine_cpu.ir import (
    OpType, NeuronType, CPUKernelVariant, DataLayout,
    ConvParams, NeuronParams, AttentionParams, Node, Edge, WeightInfo, EngineIR,
)

_NEURON_OP_MAP = {
    "FusedIFNeuron": NeuronType.IF,
    "FusedLIFNeuron": NeuronType.LIF,
    "FusedMSNeuron": NeuronType.MS,
    "FusedILIFNeuron": NeuronType.ILIF,
}

_ATTENTION_OP_MAP = {
    "FusedSpikformerAttention": "spikformer",
    "FusedMaxformerAttention": "maxformer",
    "FusedDSSAAttention": "dssa",
    "FusedTokenQKAttention": "token_qk",
}

_TEMPORAL_MEAN_TAIL_OPS = {
    "Shape", "Gather", "Div", "Cast", "Unsqueeze", "Reshape",
    "Constant", "ConstantOfShape", "Expand",
}


def _get_attr(onnx_node, name, default=None):
    for attr in onnx_node.attribute:
        if attr.name == name:
            if attr.type == 1:   return attr.f
            elif attr.type == 2: return attr.i
            elif attr.type == 7: return list(attr.ints)
            elif attr.type == 6: return list(attr.floats)
            elif attr.type == 4: return numpy_helper.to_array(attr.t)
    return default


def _get_attrs(onnx_node) -> dict:
    attrs = {}
    for attr in onnx_node.attribute:
        if attr.type == 1:   attrs[attr.name] = attr.f
        elif attr.type == 2: attrs[attr.name] = attr.i
        elif attr.type == 7: attrs[attr.name] = list(attr.ints)
        elif attr.type == 6: attrs[attr.name] = list(attr.floats)
    return attrs


class ONNXParser:
    """Parse ONNX into sengine_cpu EngineIR."""

    def __init__(self, onnx_path: str):
        self.onnx_model = onnx.load(onnx_path)
        try:
            self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        except Exception:
            pass
        self.graph = self.onnx_model.graph

        self._initializers: dict[str, np.ndarray] = {}
        for init in self.graph.initializer:
            self._initializers[init.name] = numpy_helper.to_array(init)

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

        # Constants + 5-pass propagation
        self._constants: dict[str, np.ndarray] = dict(self._initializers)
        for node in self.graph.node:
            if node.op_type == "Constant" and node.output:
                for attr in node.attribute:
                    if attr.name == "value":
                        self._constants[node.output[0]] = numpy_helper.to_array(attr.t)

        self._propagate_constants_and_shapes()

    def _propagate_constants_and_shapes(self):
        """5-pass constant + shape propagation."""
        for _pass in range(5):
            for node in self.graph.node:
                out = node.output[0] if node.output else ''
                op = node.op_type

                # --- Constant propagation ---
                if op in ("Unsqueeze", "Squeeze", "Cast", "ConstantOfShape", "Expand"):
                    if node.input and node.input[0] in self._constants and out:
                        self._constants[out] = self._constants[node.input[0]]
                elif op == "Reshape" and len(node.input) >= 2:
                    if node.input[0] in self._constants and out:
                        self._constants[out] = self._constants[node.input[0]]
                elif op == "Shape" and out:
                    if node.input and node.input[0] in self._tensor_shapes:
                        shape = self._tensor_shapes[node.input[0]]
                        if all(d > 0 for d in shape):
                            self._constants[out] = np.array(shape, dtype=np.int64)
                elif op == "Gather" and out:
                    if (len(node.input) >= 2
                            and node.input[0] in self._constants
                            and node.input[1] in self._constants):
                        data = self._constants[node.input[0]].flatten()
                        idx = int(self._constants[node.input[1]].flatten()[0])
                        if 0 <= idx < len(data):
                            self._constants[out] = np.array([data[idx]], dtype=data.dtype)
                elif op == "Slice" and out:
                    if (len(node.input) >= 3
                            and all(node.input[i] in self._constants
                                    for i in range(min(3, len(node.input))))):
                        data = self._constants[node.input[0]].flatten()
                        starts = int(self._constants[node.input[1]].flatten()[0])
                        ends = int(self._constants[node.input[2]].flatten()[0])
                        self._constants[out] = data[starts:min(ends, len(data))]
                elif op == "Mul" and out:
                    if (len(node.input) >= 2
                            and node.input[0] in self._constants
                            and node.input[1] in self._constants):
                        a = self._constants[node.input[0]].flatten()
                        b = self._constants[node.input[1]].flatten()
                        self._constants[out] = (a.astype(np.int64) * b.astype(np.int64))
                elif op == "Div" and out:
                    if (len(node.input) >= 2
                            and node.input[0] in self._constants
                            and node.input[1] in self._constants):
                        a = self._constants[node.input[0]].flatten().astype(float)
                        b = self._constants[node.input[1]].flatten().astype(float)
                        b = np.where(b == 0, 1, b)
                        self._constants[out] = (a / b).astype(np.int64)
                elif op == "Concat" and out:
                    parts = []
                    all_const = True
                    for inp in node.input:
                        if inp in self._constants:
                            parts.append(self._constants[inp].flatten())
                        else:
                            all_const = False; break
                    if all_const and parts:
                        self._constants[out] = np.concatenate(parts)

                # --- Tensor shape propagation ---
                if out and out not in self._tensor_shapes:
                    in0 = node.input[0] if node.input else ''
                    in0_shape = self._tensor_shapes.get(in0)
                    if in0_shape is not None:
                        if op in ("BatchNormalization", "Relu", "Clip", "Round",
                                  "FusedIFNeuron", "FusedLIFNeuron", "FusedMSNeuron",
                                  "FusedILIFNeuron", "Add", "Mul", "Sub", "Div",
                                  "Identity", "Cast", "Expand", "Softmax"):
                            self._tensor_shapes[out] = in0_shape
                        elif op == "Conv" and node.input[1] in self._initializers:
                            w = self._initializers[node.input[1]]
                            attrs = {a.name: (list(a.ints) if a.ints else [a.i])
                                     for a in node.attribute}
                            if len(in0_shape) == 4 and len(w.shape) == 4:
                                N, C, H, W_ = in0_shape
                                kh = attrs.get('kernel_shape', [w.shape[2]])[0]
                                s = attrs.get('strides', [1])[0]
                                p = attrs.get('pads', [0])[0]
                                OH = (H + 2*p - kh) // s + 1
                                OW = (W_ + 2*p - kh) // s + 1
                                self._tensor_shapes[out] = (N, w.shape[0], OH, OW)
                        elif op == "MaxPool" and len(in0_shape) == 4:
                            attrs = {a.name: (list(a.ints) if a.ints else [a.i])
                                     for a in node.attribute}
                            N, C, H, W_ = in0_shape
                            kh = attrs.get('kernel_shape', [3])[0]
                            s = attrs.get('strides', [2])[0]
                            p = attrs.get('pads', [0])[0]
                            OH = (H + 2*p - kh) // s + 1
                            OW = (W_ + 2*p - kh) // s + 1
                            self._tensor_shapes[out] = (N, C, OH, OW)
                        elif op == "Resize":
                            sizes_name = node.input[3] if len(node.input) > 3 and node.input[3] else ''
                            if sizes_name in self._constants:
                                sizes = self._constants[sizes_name]
                                self._tensor_shapes[out] = tuple(int(s) for s in sizes.flatten())
                            else:
                                self._tensor_shapes[out] = in0_shape
                        elif op == "Tile" and len(node.input) >= 2 and node.input[1] in self._constants:
                            repeats = self._constants[node.input[1]].flatten()
                            if len(repeats) == len(in0_shape):
                                self._tensor_shapes[out] = tuple(
                                    int(d * r) for d, r in zip(in0_shape, repeats))
                        elif op == "Transpose":
                            perm = None
                            for a in node.attribute:
                                if a.name == 'perm': perm = list(a.ints)
                            if perm and len(perm) == len(in0_shape):
                                self._tensor_shapes[out] = tuple(in0_shape[p] for p in perm)
                        elif op == "Reshape" and len(node.input) >= 2 and node.input[1] in self._constants:
                            target = self._constants[node.input[1]].flatten().tolist()
                            total = 1
                            for d in in0_shape: total *= d
                            resolved = list(target)
                            neg_idx = -1; known = 1
                            for i, d in enumerate(resolved):
                                if d == 0 and i < len(in0_shape): resolved[i] = in0_shape[i]
                                if d == -1: neg_idx = i
                                elif resolved[i] > 0: known *= resolved[i]
                            if neg_idx >= 0 and known > 0:
                                resolved[neg_idx] = total // known
                            if all(d > 0 for d in resolved):
                                self._tensor_shapes[out] = tuple(int(d) for d in resolved)
                        elif op == "Slice":
                            if (len(node.input) >= 4
                                    and node.input[1] in self._constants
                                    and node.input[2] in self._constants):
                                starts = self._constants[node.input[1]].flatten()
                                ends = self._constants[node.input[2]].flatten()
                                axes = self._constants[node.input[3]].flatten() if (
                                    len(node.input) > 3 and node.input[3] in self._constants
                                ) else list(range(len(starts)))
                                out_s = list(in0_shape)
                                for a, s, e in zip(axes, starts, ends):
                                    a = int(a)
                                    if 0 <= a < len(out_s):
                                        dim = out_s[a]
                                        e_c = min(int(e), dim) if int(e) >= 0 else max(0, dim + int(e))
                                        s_c = max(0, int(s)) if int(s) >= 0 else max(0, dim + int(s))
                                        out_s[a] = e_c - s_c
                                self._tensor_shapes[out] = tuple(out_s)
                        elif op == "Flatten":
                            if len(in0_shape) >= 2:
                                flat = 1
                                for d in in0_shape[1:]: flat *= d
                                self._tensor_shapes[out] = (in0_shape[0], flat)
                        elif op == "Concat":
                            attrs = {a.name: a.i for a in node.attribute}
                            axis = attrs.get('axis', 0)
                            all_shapes = [self._tensor_shapes.get(inp) for inp in node.input]
                            if all(s is not None for s in all_shapes) and all_shapes:
                                ndim = len(all_shapes[0])
                                ax = axis if axis >= 0 else ndim + axis
                                out_s = list(all_shapes[0])
                                if 0 <= ax < ndim:
                                    out_s[ax] = sum(s[ax] for s in all_shapes)
                                self._tensor_shapes[out] = tuple(out_s)
                        elif op == "GlobalAveragePool" and len(in0_shape) == 4:
                            self._tensor_shapes[out] = (in0_shape[0], in0_shape[1], 1, 1)
                        elif op == "Gemm" and len(node.input) >= 2 and node.input[1] in self._initializers:
                            w = self._initializers[node.input[1]]
                            transB = 0
                            for a in node.attribute:
                                if a.name == 'transB': transB = a.i
                            N_out = w.shape[0] if transB else w.shape[1]
                            self._tensor_shapes[out] = (in0_shape[0], N_out)
                        elif op == "MatMul" and len(node.input) >= 2:
                            in1_shape = self._tensor_shapes.get(node.input[1])
                            if in1_shape and len(in0_shape) >= 2 and len(in1_shape) >= 2:
                                self._tensor_shapes[out] = in0_shape[:-1] + (in1_shape[-1],)
                        elif op == "ReduceMean":
                            axes = None; keepdims = 1
                            for a in node.attribute:
                                if a.name == 'axes': axes = list(a.ints)
                                if a.name == 'keepdims': keepdims = a.i
                            if axes and keepdims:
                                out_s = list(in0_shape)
                                for ax in axes:
                                    if 0 <= ax < len(out_s): out_s[ax] = 1
                                self._tensor_shapes[out] = tuple(out_s)
                            elif axes and not keepdims:
                                out_s = [d for i, d in enumerate(in0_shape) if i not in axes]
                                self._tensor_shapes[out] = tuple(out_s)
                            else:
                                self._tensor_shapes[out] = in0_shape

    def parse(self) -> EngineIR:
        ir = EngineIR()
        ir.T = self._detect_T()

        if self.graph.input:
            in_vi = self.graph.input[0]
            ir.model_input_shape = tuple(
                d.dim_value for d in in_vi.type.tensor_type.shape.dim)
        if self.graph.output:
            out_vi = self.graph.output[0]
            ir.model_output_shape = tuple(
                d.dim_value for d in out_vi.type.tensor_type.shape.dim)

        temporal_tail_names = self._find_temporal_mean_tail()

        for onnx_node in self.graph.node:
            op_type = onnx_node.op_type
            node_name = onnx_node.name or (onnx_node.output[0] if onnx_node.output else "")

            if node_name in temporal_tail_names:
                continue

            if op_type == "ReduceMean":
                gemm_out = getattr(self, '_gemm_output_name', None)
                input_names = [gemm_out] if gemm_out else [n for n in onnx_node.input if n]
                node = Node(
                    id=-1, name=node_name or "temporal_mean",
                    op_type=OpType.TemporalMean,
                    input_names=input_names,
                    output_names=list(onnx_node.output),
                    assigned_kernel=CPUKernelVariant.NativeTemporalMean,
                )
                ir.add_node(node); continue

            if op_type == "Conv":           self._parse_conv(onnx_node, ir)
            elif op_type == "BatchNormalization": self._parse_bn(onnx_node, ir)
            elif op_type in _NEURON_OP_MAP: self._parse_neuron(onnx_node, ir)
            elif op_type in _ATTENTION_OP_MAP: self._parse_fused_attention(onnx_node, ir)
            elif op_type == "Add":          self._parse_add(onnx_node, ir)
            elif op_type == "MaxPool":      self._parse_maxpool(onnx_node, ir)
            elif op_type == "GlobalAveragePool": self._parse_globalavgpool(onnx_node, ir)
            elif op_type == "Flatten":      self._parse_flatten(onnx_node, ir)
            elif op_type == "Gemm":
                self._parse_gemm(onnx_node, ir)
                self._gemm_output_name = onnx_node.output[0]
            elif op_type == "Tile":         self._parse_tile(onnx_node, ir)
            elif op_type == "MatMul":       self._parse_matmul(onnx_node, ir)
            elif op_type == "Transpose":    self._parse_transpose(onnx_node, ir)
            elif op_type == "Reshape":      self._parse_reshape(onnx_node, ir)
            elif op_type == "Mul":          self._parse_mul(onnx_node, ir)
            elif op_type == "Split":        self._parse_split(onnx_node, ir)
            elif op_type == "Resize":       self._parse_resize(onnx_node, ir)
            elif op_type == "Concat":       self._parse_concat(onnx_node, ir)
            elif op_type == "Softmax":      self._parse_softmax(onnx_node, ir)
            elif op_type == "Slice":        self._parse_slice(onnx_node, ir)
            elif op_type in _TEMPORAL_MEAN_TAIL_OPS:
                continue
            else:
                node = Node(
                    id=-1, name=node_name or f"unknown_{op_type}",
                    op_type=OpType.Identity,
                    input_names=[n for n in onnx_node.input if n],
                    output_names=list(onnx_node.output),
                    extra_attrs={"original_op": op_type},
                )
                ir.add_node(node)

        ir.weights = self._initializers
        ir.build_edges()
        ir.compute_topo_order()
        return ir

    # ─── Op handlers ──────────────────────────────────────────────

    def _detect_T(self) -> int:
        for node in self.graph.node:
            if node.op_type in _NEURON_OP_MAP:
                T = _get_attr(node, "T", None) or _get_attr(node, "T_i", None)
                if T and T > 0:
                    return int(T)
        return 4

    def _parse_conv(self, onnx_node, ir: EngineIR):
        attrs = _get_attrs(onnx_node)
        kernel_shape = attrs.get("kernel_shape", [1, 1])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        dilations = attrs.get("dilations", [1, 1])
        groups = attrs.get("group", 1)

        weight_name = onnx_node.input[1] if len(onnx_node.input) > 1 else ""
        weight_shape = ()
        if weight_name in self._initializers:
            weight_shape = self._initializers[weight_name].shape

        out_channels = int(weight_shape[0]) if weight_shape else 0
        in_channels = int(weight_shape[1]) * groups if weight_shape else 0

        conv_params = ConvParams(
            in_channels=in_channels, out_channels=out_channels,
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

        out_shape = self._tensor_shapes.get(onnx_node.output[0], ())
        weight_info = WeightInfo(name=weight_name, shape=weight_shape)
        bias_info = None
        if len(onnx_node.input) > 2 and onnx_node.input[2]:
            bias_info = WeightInfo(
                name=onnx_node.input[2],
                shape=self._initializers.get(onnx_node.input[2], np.array([])).shape,
            )

        node = Node(
            id=-1, name=onnx_node.name or f"conv_{weight_name}",
            op_type=OpType.Conv2d,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            output_shapes=[out_shape] if out_shape else [],
            conv_params=conv_params,
            weight_info=weight_info, bias_info=bias_info,
            assigned_kernel=CPUKernelVariant.TVMConvBN,
        )
        ir.add_node(node)

    def _parse_bn(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or f"bn_{onnx_node.output[0]}",
            op_type=OpType.BatchNorm,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs=_get_attrs(onnx_node),
        )
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
        elif neuron_type == NeuronType.LIF:
            params.tau = _get_attr(onnx_node, "tau_f",
                                   _get_attr(onnx_node, "tau", 2.0))
            params.v_threshold = _get_attr(onnx_node, "v_threshold_f",
                                           _get_attr(onnx_node, "v_threshold", 1.0))
            params.v_reset = _get_attr(onnx_node, "v_reset_f",
                                       _get_attr(onnx_node, "v_reset", 0.0))
        elif neuron_type == NeuronType.MS:
            params.tau = _get_attr(onnx_node, "decay_f",
                                   _get_attr(onnx_node, "decay", 0.25))
            params.v_threshold = _get_attr(onnx_node, "thresh_f",
                                           _get_attr(onnx_node, "thresh", 0.5))
        elif neuron_type == NeuronType.ILIF:
            params.decay = _get_attr(onnx_node, "decay_f",
                                     _get_attr(onnx_node, "decay", 0.25))
            params.max_level = int(_get_attr(onnx_node, "max_level_i",
                                             _get_attr(onnx_node, "max_level", 4)))

        _NEURON_OPTYPE = {
            NeuronType.IF: OpType.IF, NeuronType.LIF: OpType.LIF,
            NeuronType.MS: OpType.MS, NeuronType.ILIF: OpType.ILIF,
        }
        # CPU kernel: standalone native neuron (will be fused by optimizer if possible)
        _NEURON_KERNEL = {
            NeuronType.IF: CPUKernelVariant.NativeIF,
            NeuronType.LIF: CPUKernelVariant.NativeLIF,
            NeuronType.MS: CPUKernelVariant.NativeLIF,
            NeuronType.ILIF: CPUKernelVariant.NativeILIF,
        }
        node = Node(
            id=-1, name=onnx_node.name or f"neuron_{onnx_node.output[0]}",
            op_type=_NEURON_OPTYPE.get(neuron_type, OpType.LIF),
            is_stateful=True,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            neuron_params=params,
            assigned_kernel=_NEURON_KERNEL.get(neuron_type, CPUKernelVariant.NativeLIF),
        )
        ir.add_node(node)

    def _parse_fused_attention(self, onnx_node, ir: EngineIR):
        variant = _ATTENTION_OP_MAP[onnx_node.op_type]

        def _ga(name, default=None):
            val = _get_attr(onnx_node, name)
            if val is not None: return val
            base = name[:-2] if name.endswith(('_i', '_f')) else name
            val = _get_attr(onnx_node, base)
            return val if val is not None else default

        params = AttentionParams(
            variant=variant,
            num_heads=int(_ga("num_heads_i", 1) or 1),
            head_dim=int(_ga("head_dim_i", 64) or 64),
            scale=float(_ga("scale_f", 1.0) or 1.0),
            H=int(_ga("H_i", 0) or _ga("H_in_i", 0) or _ga("H_in", 0) or 0),
            W=int(_ga("W_i", 0) or _ga("W_in_i", 0) or _ga("W_in", 0) or 0),
            attn_lif_tau=float(_ga("attn_lif_tau_f", 2.0) or 2.0),
            attn_lif_v_threshold=float(_ga("attn_lif_v_threshold_f", 1.0) or 1.0),
        )
        extra = {}
        if variant == "dssa" and len(onnx_node.input) >= 4:
            extra["scale1_name"] = onnx_node.input[2]
            extra["scale2_name"] = onnx_node.input[3]

        # On CPU, attention is decomposed — no fused attention kernel
        node = Node(
            id=-1, name=onnx_node.name or f"fused_attn_{onnx_node.output[0]}",
            op_type=OpType.FusedAttention,
            is_stateful=True,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            attention_params=params,
            assigned_kernel=CPUKernelVariant.ZeroCost,  # decomposed later
            extra_attrs=extra,
        )
        ir.add_node(node)

    def _parse_add(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or f"add_{onnx_node.output[0]}",
            op_type=OpType.Add,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=CPUKernelVariant.NativeAdd,
        )
        ir.add_node(node)

    def _parse_maxpool(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or "maxpool",
            op_type=OpType.MaxPool,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            pool_params=_get_attrs(onnx_node),
            assigned_kernel=CPUKernelVariant.NativeMaxPool,
        )
        ir.add_node(node)

    def _parse_globalavgpool(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or "globalavgpool",
            op_type=OpType.GlobalAvgPool,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=CPUKernelVariant.NativeGlobalAvgPool,
        )
        ir.add_node(node)

    def _parse_flatten(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or "flatten",
            op_type=OpType.Flatten,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=CPUKernelVariant.ZeroCost,
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
            id=-1, name=onnx_node.name or "gemm_classifier",
            op_type=OpType.Gemm,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            gemm_params=_get_attrs(onnx_node),
            weight_info=WeightInfo(name=weight_name, shape=weight_shape),
            bias_info=WeightInfo(name=bias_name) if bias_name else None,
            assigned_kernel=CPUKernelVariant.NativeGemm,
        )
        ir.add_node(node)

    def _parse_tile(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or "tile_repeat",
            op_type=OpType.Tile,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=CPUKernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_matmul(self, onnx_node, ir: EngineIR):
        input_a = onnx_node.input[0] if len(onnx_node.input) > 0 else ""
        input_b = onnx_node.input[1] if len(onnx_node.input) > 1 else ""

        if input_b in self._initializers:
            weight_shape = self._initializers[input_b].shape
            node = Node(
                id=-1, name=onnx_node.name or f"linear_{onnx_node.output[0]}",
                op_type=OpType.Linear,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                weight_info=WeightInfo(name=input_b, shape=weight_shape),
                gemm_params={"N": int(weight_shape[-1]) if weight_shape else 0},
            )
        elif input_a in self._initializers:
            weight_shape = self._initializers[input_a].shape
            node = Node(
                id=-1, name=onnx_node.name or f"linear_{onnx_node.output[0]}",
                op_type=OpType.Linear,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                weight_info=WeightInfo(name=input_a, shape=weight_shape),
                gemm_params={"N": int(weight_shape[-1]) if weight_shape else 0},
            )
        else:
            node = Node(
                id=-1, name=onnx_node.name or f"matmul_{onnx_node.output[0]}",
                op_type=OpType.MatMul,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                assigned_kernel=CPUKernelVariant.TVMMatMul,
            )
        ir.add_node(node)

    def _parse_transpose(self, onnx_node, ir: EngineIR):
        perm = _get_attr(onnx_node, "perm", None)
        node = Node(
            id=-1, name=onnx_node.name or f"transpose_{onnx_node.output[0]}",
            op_type=OpType.Transpose,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"perm": list(perm) if perm else []},
            assigned_kernel=CPUKernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_reshape(self, onnx_node, ir: EngineIR):
        shape_name = onnx_node.input[1] if len(onnx_node.input) > 1 else ""
        target_shape = None
        if shape_name in self._constants:
            target_shape = self._constants[shape_name].flatten().astype(int).tolist()
        elif shape_name in self._initializers:
            target_shape = self._initializers[shape_name].astype(int).tolist()

        node = Node(
            id=-1, name=onnx_node.name or f"reshape_{onnx_node.output[0]}",
            op_type=OpType.Reshape,
            input_names=[onnx_node.input[0]] if onnx_node.input else [],
            output_names=list(onnx_node.output),
            extra_attrs={"target_shape": target_shape},
            assigned_kernel=CPUKernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_mul(self, onnx_node, ir: EngineIR):
        for inp_name in onnx_node.input:
            if inp_name in self._initializers:
                val = self._initializers[inp_name]
                if val.size == 1:
                    node = Node(
                        id=-1, name=onnx_node.name or f"scale_{onnx_node.output[0]}",
                        op_type=OpType.Scale,
                        input_names=[n for n in onnx_node.input if n and n != inp_name],
                        output_names=list(onnx_node.output),
                        extra_attrs={"scale_value": float(val.flat[0])},
                        assigned_kernel=CPUKernelVariant.NativeAdd,  # reuse add for scale
                    )
                    ir.add_node(node); return
        node = Node(
            id=-1, name=onnx_node.name or f"mul_{onnx_node.output[0]}",
            op_type=OpType.Mul,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            assigned_kernel=CPUKernelVariant.NativeAdd,
        )
        ir.add_node(node)

    def _parse_split(self, onnx_node, ir: EngineIR):
        node = Node(
            id=-1, name=onnx_node.name or f"split_{onnx_node.output[0]}",
            op_type=OpType.Identity,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"original_op": "Split", **_get_attrs(onnx_node)},
            assigned_kernel=CPUKernelVariant.ZeroCost,
        )
        ir.add_node(node)

    def _parse_resize(self, onnx_node, ir: EngineIR):
        scales = sizes = None
        if len(onnx_node.input) > 2 and onnx_node.input[2]:
            scales = self._constants.get(onnx_node.input[2])
        if len(onnx_node.input) > 3 and onnx_node.input[3]:
            sizes = self._constants.get(onnx_node.input[3])

        mode = "nearest"
        for attr in onnx_node.attribute:
            if attr.name == "mode":
                mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s

        extra = {"mode": mode}
        if scales is not None:
            extra["scales"] = [float(s) for s in scales.flatten().tolist()]
        if sizes is not None:
            extra["sizes"] = [int(s) for s in sizes.flatten().tolist()]

        data_input = [onnx_node.input[0]] if onnx_node.input else []
        node = Node(
            id=-1, name=onnx_node.name or f"resize_{onnx_node.output[0]}",
            op_type=OpType.Resize,
            input_names=data_input,
            output_names=list(onnx_node.output),
            extra_attrs=extra,
            assigned_kernel=CPUKernelVariant.NativeResize,
        )
        ir.add_node(node)

    def _parse_concat(self, onnx_node, ir: EngineIR):
        axis = 1
        for attr in onnx_node.attribute:
            if attr.name == "axis": axis = int(attr.i)

        data_inputs = [n for n in onnx_node.input if n and n not in self._constants]
        if not data_inputs:
            node = Node(
                id=-1, name=onnx_node.name or f"concat_{onnx_node.output[0]}",
                op_type=OpType.Identity,
                input_names=[n for n in onnx_node.input if n],
                output_names=list(onnx_node.output),
                assigned_kernel=CPUKernelVariant.ZeroCost,
            )
            ir.add_node(node); return

        node = Node(
            id=-1, name=onnx_node.name or f"concat_{onnx_node.output[0]}",
            op_type=OpType.Concat,
            input_names=data_inputs,
            output_names=list(onnx_node.output),
            extra_attrs={"axis": axis},
            assigned_kernel=CPUKernelVariant.NativeConcat,
        )
        ir.add_node(node)

    def _parse_softmax(self, onnx_node, ir: EngineIR):
        axis = -1
        for attr in onnx_node.attribute:
            if attr.name == "axis": axis = int(attr.i)
        node = Node(
            id=-1, name=onnx_node.name or f"softmax_{onnx_node.output[0]}",
            op_type=OpType.Softmax,
            input_names=[n for n in onnx_node.input if n],
            output_names=list(onnx_node.output),
            extra_attrs={"axis": axis},
            assigned_kernel=CPUKernelVariant.NativeSoftmax,
        )
        ir.add_node(node)

    def _parse_slice(self, onnx_node, ir: EngineIR):
        extra = {}
        for idx, key in [(1, "starts"), (2, "ends"), (3, "axes")]:
            if idx < len(onnx_node.input):
                val = self._constants.get(onnx_node.input[idx])
                if val is not None:
                    extra[key] = [int(v) for v in val.flatten().tolist()]
        node = Node(
            id=-1, name=onnx_node.name or f"slice_{onnx_node.output[0]}",
            op_type=OpType.Slice,
            input_names=[onnx_node.input[0]] if onnx_node.input else [],
            output_names=list(onnx_node.output),
            extra_attrs=extra,
            assigned_kernel=CPUKernelVariant.ZeroCost,
        )
        ir.add_node(node)

    # ─── Helpers ──────────────────────────────────────────────────

    def _resolve_constant_identity_nodes(self):
        identity_map: dict[str, str] = {}
        for node in self.graph.node:
            if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
                identity_map[node.output[0]] = node.input[0]
            elif node.op_type == "Constant" and len(node.output) == 1:
                val = _get_attr(node, "value", None)
                if val is not None:
                    self._initializers[node.output[0]] = val

        for out_name, src_name in identity_map.items():
            visited = set()
            current = src_name
            while current in identity_map and current not in visited:
                visited.add(current)
                current = identity_map[current]
            if current in self._initializers and out_name not in self._initializers:
                self._initializers[out_name] = self._initializers[current]

    def _find_temporal_mean_tail(self) -> set[str]:
        tail_names = set()
        gemm_outputs = set()
        for node in reversed(self.graph.node):
            if node.op_type == "Gemm":
                gemm_outputs.update(node.output)
                break

        if not gemm_outputs:
            return tail_names

        visited_tensors = set(gemm_outputs)
        for node in self.graph.node:
            if node.op_type in ("Gemm", "ReduceMean"):
                continue
            if any(inp in visited_tensors for inp in node.input):
                if node.op_type in _TEMPORAL_MEAN_TAIL_OPS:
                    name = node.name or node.output[0] if node.output else ""
                    tail_names.add(name)
                    visited_tensors.update(node.output)

        return tail_names
