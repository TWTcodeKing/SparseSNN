"""Operator DAG data structures and extraction for SNN models.

Provides OpNode, Edge, OperatorDAG, and extract_dag() to build a flat
operator graph from a PyTorch SNN model. Foundation for temporal
unrolling (Phase 2) and pipeline partitioning.

Architecture-specific extractors live in model_dag/ subpackage.
This module contains the shared data structures, shape helpers,
builder helpers, and the top-level extract_dag() dispatcher.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from sengine.tdl.analysis import collect_neuron_params


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class OpNode:
    """Single leaf operator in the DAG."""
    id: int
    name: str          # module path, e.g. "layer1.0.conv1.module.0"
    op_type: str       # "conv2d" | "bn2d" | "if_neuron" | "add" | ...
    is_stateful: bool  # True only for spiking neurons
    params: dict       # op-specific parameters
    input_shape: tuple  # per-timestep (B, C, H, W) or (B, F)
    output_shape: tuple


@dataclass
class Edge:
    """Directed activation edge between two OpNodes."""
    src_id: int
    dst_id: int
    edge_type: str = "data"   # "data" or "state" (state added in Phase 2)
    tensor_shape: tuple = ()
    tensor_bytes: int = 0


class OperatorDAG:
    """Directed acyclic graph of leaf operators in an SNN model."""

    def __init__(self):
        self.nodes: dict[int, OpNode] = {}
        self.edges: list[Edge] = []
        self._next_id = 0
        self._adj: dict[int, list[int]] = {}   # node -> successors
        self._radj: dict[int, list[int]] = {}  # node -> predecessors

    def add_node(self, name: str, op_type: str, is_stateful: bool,
                 params: dict, input_shape: tuple,
                 output_shape: tuple) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = OpNode(nid, name, op_type, is_stateful, params,
                                 input_shape, output_shape)
        self._adj[nid] = []
        self._radj[nid] = []
        return nid

    def add_edge(self, src_id: int, dst_id: int, edge_type: str = "data",
                 dtype_bytes: int = 2):
        shape = self.nodes[src_id].output_shape
        nbytes = dtype_bytes
        for d in shape:
            nbytes *= d
        e = Edge(src_id, dst_id, edge_type, shape, nbytes)
        self.edges.append(e)
        self._adj[src_id].append(dst_id)
        self._radj[dst_id].append(src_id)

    def topological_order(self) -> list[int]:
        """Kahn's algorithm — returns list of node IDs in topological order."""
        in_deg = {nid: len(preds) for nid, preds in self._radj.items()}
        queue = deque(nid for nid, d in in_deg.items() if d == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._adj[nid]:
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)
        if len(order) != len(self.nodes):
            raise ValueError(
                f"Graph has cycles: ordered {len(order)}/{len(self.nodes)} nodes")
        return order

    def predecessors(self, nid: int) -> list[int]:
        return list(self._radj[nid])

    def successors(self, nid: int) -> list[int]:
        return list(self._adj[nid])

    def subgraph(self, node_ids: set[int]) -> OperatorDAG:
        """Extract a subgraph containing only the given node IDs."""
        sub = OperatorDAG()
        id_map = {}
        for nid in sorted(node_ids):
            n = self.nodes[nid]
            new_id = sub.add_node(n.name, n.op_type, n.is_stateful,
                                  n.params, n.input_shape, n.output_shape)
            id_map[nid] = new_id
        for e in self.edges:
            if e.src_id in node_ids and e.dst_id in node_ids:
                sub.add_edge(id_map[e.src_id], id_map[e.dst_id], e.edge_type)
        return sub

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def stateful_nodes(self) -> list[int]:
        """Return IDs of all stateful (neuron) nodes."""
        return [nid for nid, n in self.nodes.items() if n.is_stateful]

    def __repr__(self):
        s = sum(1 for n in self.nodes.values() if n.is_stateful)
        return f"OperatorDAG(nodes={self.num_nodes}, edges={self.num_edges}, stateful={s})"

    def summary(self) -> str:
        """Human-readable summary grouped by op type."""
        from collections import Counter
        counts = Counter(n.op_type for n in self.nodes.values())
        lines = [repr(self)]
        for op, cnt in sorted(counts.items()):
            lines.append(f"  {op}: {cnt}")
        return "\n".join(lines)


# ===================================================================
# Shape computation helpers
# ===================================================================

def _intify(x):
    """Extract int from a tuple-or-int (assumes square for spatial dims)."""
    return x[0] if isinstance(x, tuple) else x


def _conv2d_params(conv: nn.Conv2d) -> dict:
    """Extract Conv2d parameters as a flat dict."""
    return {
        'in_channels': conv.in_channels,
        'out_channels': conv.out_channels,
        'kernel_size': _intify(conv.kernel_size),
        'stride': _intify(conv.stride),
        'padding': _intify(conv.padding),
        'dilation': _intify(conv.dilation),
        'groups': conv.groups,
    }


def _conv2d_out_shape(in_shape: tuple, out_channels: int, kernel_size: int,
                      stride: int, padding: int, dilation: int = 1) -> tuple:
    """Compute Conv2d output shape from input (B, C, H, W)."""
    B, C, H, W = in_shape
    H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return (B, out_channels, H_out, W_out)


def _pool_out_shape(in_shape: tuple, kernel_size: int, stride: int,
                    padding: int) -> tuple:
    """Compute pooling output shape from input (B, C, H, W)."""
    B, C, H, W = in_shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    return (B, C, H_out, W_out)


# ===================================================================
# Building block helpers (shared across architectures)
# ===================================================================

def _add_conv_bn(dag: OperatorDAG, conv: nn.Conv2d, bn: nn.Module,
                 conv_name: str, bn_name: str, in_shape: tuple,
                 in_node_id: Optional[int] = None) -> tuple[int, tuple]:
    """Add Conv2d + BN pair. Returns (bn_node_id, output_shape)."""
    p = _conv2d_params(conv)
    out = _conv2d_out_shape(in_shape, p['out_channels'], p['kernel_size'],
                            p['stride'], p['padding'], p['dilation'])
    conv_id = dag.add_node(conv_name, 'conv2d', False, p, in_shape, out)
    bn_id = dag.add_node(bn_name, 'bn2d', False,
                         {'num_features': bn.num_features}, out, out)
    if in_node_id is not None:
        dag.add_edge(in_node_id, conv_id)
    dag.add_edge(conv_id, bn_id)
    return bn_id, out


def _add_neuron(dag: OperatorDAG, name: str, shape: tuple,
                neuron_params: dict, in_node_id: int) -> int:
    """Add a spiking neuron node. Returns node ID."""
    nparams = neuron_params.get(name, {})
    ntype = nparams.get('type', 'IF').lower() + '_neuron'
    nid = dag.add_node(name, ntype, True, nparams, shape, shape)
    dag.add_edge(in_node_id, nid)
    return nid


def _add_residual(dag: OperatorDAG, name: str, connect_f: str,
                  shape: tuple, main_id: int, shortcut_id: int) -> int:
    """Add a residual connection (add/mul/iand) node. Returns node ID."""
    op = {'ADD': 'add', 'AND': 'mul', 'IAND': 'iand'}[connect_f]
    res_id = dag.add_node(name, op, False, {}, shape, shape)
    dag.add_edge(main_id, res_id)
    dag.add_edge(shortcut_id, res_id)
    return res_id


def _add_bn(dag, bn, bn_name, shape, in_node_id):
    """Add a bare BN node (works for BN1d, BN2d, BN3d). Returns node_id."""
    nid = dag.add_node(bn_name, 'bn2d', False,
                       {'num_features': bn.num_features}, shape, shape)
    dag.add_edge(in_node_id, nid)
    return nid


def _add_linear(dag, linear, name, in_shape, in_node_id):
    """Add a Linear node. Returns (node_id, output_shape)."""
    out_f = linear.out_features
    out_shape = in_shape[:-1] + (out_f,)
    nid = dag.add_node(name, 'linear', False,
                       {'in_features': linear.in_features,
                        'out_features': out_f},
                       in_shape, out_shape)
    dag.add_edge(in_node_id, nid)
    return nid, out_shape


def _add_matmul(dag, name, shape_a, shape_out, in_a, in_b):
    """Add a MatMul node with two inputs. Returns node_id."""
    nid = dag.add_node(name, 'matmul', False, {}, shape_a, shape_out)
    dag.add_edge(in_a, nid)
    dag.add_edge(in_b, nid)
    return nid


def _add_pool(dag, pool, name, in_shape, in_node_id):
    """Add a MaxPool2d node. Returns (node_id, output_shape)."""
    ks = _intify(pool.kernel_size)
    st = _intify(pool.stride)
    pa = _intify(pool.padding)
    out = _pool_out_shape(in_shape, ks, st, pa)
    nid = dag.add_node(name, 'maxpool2d', False,
                       {'kernel_size': ks, 'stride': st, 'padding': pa},
                       in_shape, out)
    dag.add_edge(in_node_id, nid)
    return nid, out


# ===================================================================
# Top-level dispatcher
# ===================================================================

def extract_dag(model: nn.Module, input_shape: tuple = (1, 3, 224, 224),
                ) -> OperatorDAG:
    """Extract an OperatorDAG from a PyTorch SNN model.

    Architecture detection mirrors transforms.py:_patch_model_forward().
    Delegates to architecture-specific extractors in model_dag/.

    Args:
        model:       PyTorch SNN model (eval mode).
        input_shape: Per-timestep input (B, C, H, W).

    Returns:
        OperatorDAG with leaf operators and data edges.
    """
    from sengine.tdl.model_dag import (
        extract_sewresnet_dag, extract_sewresnet_cifar_dag,
        extract_msresnet18_dag, extract_msresnet_cifar_dag, extract_msresnet104_dag,
        extract_spikformer_dag, extract_spikingresformer_dag,
        extract_metaformer_dag, extract_qkformer_dag, extract_maxformer_dag,
    )

    from models.sewresnet import SEWResNet, SEWResNetCifar
    from models.msresnet import MSResNet18, MSResNet104, MSResNetCifar
    from models.spikformer import Spikformer
    from models.spikingresformer import SpikingResformer
    from models.metaformer import SpikeDrivenTransformerV2
    from models.qkformer import QKFormer
    from models.maxformer import MaxFormer, MS_QKFormer

    try:
        from models.metaformer import SpikeDrivenTransformerV2Cifar
    except ImportError:
        SpikeDrivenTransformerV2Cifar = None
    try:
        from models.maxformer import MaxFormerCifar, MS_QKFormerCifar, MaxFormerDVS
    except ImportError:
        MaxFormerCifar = MS_QKFormerCifar = MaxFormerDVS = None

    # --- ResNet family ---
    if isinstance(model, SEWResNet):
        return extract_sewresnet_dag(model, input_shape)
    if isinstance(model, SEWResNetCifar):
        return extract_sewresnet_cifar_dag(model, input_shape)
    if isinstance(model, MSResNet18):
        return extract_msresnet18_dag(model, input_shape)
    if isinstance(model, MSResNet104):
        return extract_msresnet104_dag(model, input_shape)
    if isinstance(model, MSResNetCifar):
        return extract_msresnet_cifar_dag(model, input_shape)

    # --- Transformer family ---
    if isinstance(model, Spikformer):
        return extract_spikformer_dag(model, input_shape)
    if isinstance(model, SpikingResformer):
        return extract_spikingresformer_dag(model, input_shape)
    if SpikeDrivenTransformerV2Cifar and isinstance(model, SpikeDrivenTransformerV2Cifar):
        return extract_metaformer_dag(model, input_shape)
    if isinstance(model, SpikeDrivenTransformerV2):
        return extract_metaformer_dag(model, input_shape)
    if isinstance(model, QKFormer):
        return extract_qkformer_dag(model, input_shape)
    if MS_QKFormerCifar and isinstance(model, MS_QKFormerCifar):
        return extract_maxformer_dag(model, input_shape)
    if isinstance(model, MS_QKFormer):
        return extract_maxformer_dag(model, input_shape)
    if MaxFormerCifar and isinstance(model, MaxFormerCifar):
        return extract_maxformer_dag(model, input_shape)
    if MaxFormerDVS and isinstance(model, MaxFormerDVS):
        return extract_maxformer_dag(model, input_shape)
    if isinstance(model, MaxFormer):
        return extract_maxformer_dag(model, input_shape)

    # Fallback: attribute-based detection
    if (hasattr(model, 'conv1') and hasattr(model, 'bn1')
            and hasattr(model, 'sn1')):
        if hasattr(model, 'layer4'):
            return extract_sewresnet_dag(model, input_shape)
        else:
            return extract_sewresnet_cifar_dag(model, input_shape)

    if hasattr(model, 'conv1') and hasattr(model, 'conv2_x'):
        if hasattr(model.conv1, 'module') and isinstance(model.conv1.module, nn.Sequential):
            seq = model.conv1.module
            if len(seq) > 2 and isinstance(seq[0], nn.Conv2d):
                return extract_msresnet104_dag(model, input_shape)
        return extract_msresnet18_dag(model, input_shape)

    if hasattr(model, 'conv1') and hasattr(model, 'layer1') and hasattr(model, 'sn_out'):
        return extract_msresnet_cifar_dag(model, input_shape)

    if hasattr(model, 'patch_embed') and hasattr(model, 'block'):
        return extract_spikformer_dag(model, input_shape)

    if hasattr(model, 'prologue') and hasattr(model, 'layers'):
        return extract_spikingresformer_dag(model, input_shape)

    if hasattr(model, 'downsample1_1') and hasattr(model, 'block3'):
        return extract_metaformer_dag(model, input_shape)

    raise ValueError(
        f"Unknown model architecture: {type(model).__name__}. "
        "Supported: SEWResNet, SEWResNetCifar, MSResNet, Spikformer, "
        "SpikingResformer, MetaFormer, QKFormer, MaxFormer, MS_QKFormer.")
