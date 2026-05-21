"""Graph IR data structures for sengine-cpu.

Self-contained CPU inference engine IR. No imports from sengine/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from collections import deque


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class OpType(Enum):
    Conv2d = auto()
    Linear = auto()
    MatMul = auto()
    LIF = auto()
    IF = auto()
    MS = auto()
    Add = auto()
    Mul = auto()
    Sub = auto()
    Scale = auto()
    MaxPool = auto()
    GlobalAvgPool = auto()
    Flatten = auto()
    Reshape = auto()
    Transpose = auto()
    Tile = auto()
    TemporalMean = auto()
    InputConvert = auto()
    OutputConvert = auto()
    Identity = auto()
    BatchNorm = auto()
    Gemm = auto()
    Concat = auto()
    ReduceMean = auto()
    FusedAttention = auto()
    Resize = auto()
    Softmax = auto()
    ILIF = auto()
    Slice = auto()


class NeuronType(Enum):
    LIF = auto()
    IF = auto()
    MS = auto()
    ILIF = auto()
    NoNeuron = auto()


class CPUKernelVariant(Enum):
    """Kernel implementations available on CPU."""
    # TVM-compiled fused kernels (.so via dlopen)
    TVMConvBNIF = auto()            # Conv+BN+IF, cache-tiled T-loop
    TVMConvBNLIF = auto()           # Conv+BN+LIF, cache-tiled T-loop
    TVMConvBNAddLIF = auto()        # Conv+BN+Add+LIF with residual
    TVMConv1x1BNIF = auto()        # 1x1 Conv+BN+IF
    TVMConv1x1BNLIF = auto()       # 1x1 Conv+BN+LIF
    TVMMatMulBNIF = auto()         # MatMul+BN+IF
    TVMMatMulBNLIF = auto()        # MatMul+BN+LIF
    TVMAddLIF = auto()             # Fused Add+LIF
    TVMPoolLIF = auto()            # Fused MaxPool+LIF
    # TVM-compiled decomposed kernels (.so)
    TVMConvBN = auto()             # Conv+BN (no neuron)
    TVMConv1x1BN = auto()          # 1x1 Conv+BN
    TVMLinearBN = auto()           # Linear+BN
    TVMMatMul = auto()             # Pure MatMul
    # Native C kernels (hand-optimized, SIMD)
    NativeIF = auto()              # IF neuron (OpenMP + SIMD)
    NativeLIF = auto()             # LIF neuron
    NativeILIF = auto()            # Integer LIF
    NativeAdd = auto()             # Element-wise add
    NativeMaxPool = auto()         # MaxPool2d
    NativeGlobalAvgPool = auto()   # Global average pool
    NativeTemporalMean = auto()    # Temporal mean (reduce over T)
    NativeGemm = auto()            # BLAS-backed GEMM (classifier)
    NativeSoftmax = auto()         # Softmax
    NativeResize = auto()          # Nearest-neighbor / bilinear resize
    NativeConcat = auto()          # Channel concat
    # Zero-cost (no computation)
    ZeroCost = auto()              # Reshape, Flatten, Identity, Transpose
    Skip = auto()                  # Absorbed into fused kernel


class BoundType(Enum):
    """Resource binding for BA-MTTS scheduling.

    On CPU, COMPUTE ops are ALU/FMA-bound (Conv, MatMul) while MEMORY ops
    are bandwidth-bound (neurons, Add, Pool). Scheduling C->M transitions
    helps exploit CPU out-of-order execution and prefetch overlap.
    """
    COMPUTE = "C"   # ALU/FMA-bound (Conv, Linear, MatMul)
    MEMORY = "M"    # bandwidth-bound (neurons, Add, Pool)
    ZERO = "Z"      # negligible cost (Reshape, Flatten, Identity)


class DataLayout(Enum):
    NCHW = auto()   # Channel-first (PyTorch default)
    NHWC = auto()   # Channel-last (SIMD-friendly on CPU)
    ND = auto()      # Non-spatial (2D, 3D, or flat)


# ═══════════════════════════════════════════════════════════════════
# Op parameter dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConvParams:
    in_channels: int = 0
    out_channels: int = 0
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1


@dataclass
class NeuronParams:
    neuron_type: NeuronType = NeuronType.NoNeuron
    T: int = 4
    tau: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    hard_reset: bool = True
    decay: float = 0.25      # I-LIF specific
    max_level: int = 4       # I-LIF specific


@dataclass
class AttentionParams:
    variant: str = ""        # "spikformer", "maxformer", "dssa", "tokenqk"
    num_heads: int = 1
    head_dim: int = 64
    scale: float = 1.0
    H: int = 0
    W: int = 0
    attn_lif_tau: float = 2.0
    attn_lif_v_threshold: float = 1.0


@dataclass
class WeightInfo:
    name: str = ""
    shape: tuple = ()
    onnx_idx: int = -1


# ═══════════════════════════════════════════════════════════════════
# Graph nodes and edges
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Node:
    id: int
    name: str
    op_type: OpType
    is_stateful: bool = False

    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    input_shapes: list[tuple] = field(default_factory=list)
    output_shapes: list[tuple] = field(default_factory=list)

    # Op-specific params
    conv_params: Optional[ConvParams] = None
    neuron_params: Optional[NeuronParams] = None
    attention_params: Optional[AttentionParams] = None
    pool_params: Optional[dict] = None
    gemm_params: Optional[dict] = None
    extra_attrs: dict = field(default_factory=dict)

    # Engine annotations (set by optimizer)
    assigned_kernel: CPUKernelVariant = CPUKernelVariant.ZeroCost
    layout: DataLayout = DataLayout.NCHW
    fusion_group_id: int = -1
    output_mem_slot: int = -1

    # Weight references
    weight_info: Optional[WeightInfo] = None
    bias_info: Optional[WeightInfo] = None
    bn_scale: Optional[list] = None
    bn_bias: Optional[list] = None

    # BA-MTTS scheduling
    bound_type: BoundType = BoundType.ZERO
    est_latency_us: float = 0.0

    # CPU-specific: tile config for TVM cache-tiled kernels
    cpu_tile_config: Optional[dict] = None   # {tile_M, tile_N, tile_K, ...}


@dataclass
class Edge:
    src_id: int
    dst_id: int
    tensor_name: str = ""
    tensor_shape: tuple = ()
    tensor_bytes: int = 0
    layout: DataLayout = DataLayout.NCHW


@dataclass
class FusionGroup:
    group_id: int
    conv_node_id: int
    bn_node_id: int = -1
    neuron_node_id: int = -1
    variant: CPUKernelVariant = CPUKernelVariant.TVMConvBNIF
    bn_scale: Optional[list] = None
    bn_bias: Optional[list] = None


# ═══════════════════════════════════════════════════════════════════
# Engine IR — the main graph container
# ═══════════════════════════════════════════════════════════════════

class EngineIR:
    """Directed acyclic graph of operators for sengine-cpu."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []
        self.fusion_groups: list[FusionGroup] = []
        self.T: int = 4
        self.precision: str = "fp32"   # always FP32 on CPU
        self.model_input_shape: tuple = ()
        self.model_output_shape: tuple = ()

        # Tensor name → producer node id
        self._tensor_producer: dict[str, int] = {}
        # Tensor name → list of consumer node ids
        self._tensor_consumers: dict[str, list[int]] = {}
        # Adjacency
        self._adj: dict[int, list[int]] = {}
        self._radj: dict[int, list[int]] = {}
        self._topo_order: list[int] = []
        self._next_id: int = 0

        # Weight data (name → numpy array, loaded by parser)
        self.weights: dict = {}  # str → numpy.ndarray

    def add_node(self, node: Node) -> int:
        nid = self._next_id
        self._next_id += 1
        node.id = nid
        self.nodes[nid] = node
        self._adj[nid] = []
        self._radj[nid] = []
        for out_name in node.output_names:
            self._tensor_producer[out_name] = nid
        for in_name in node.input_names:
            if in_name not in self._tensor_consumers:
                self._tensor_consumers[in_name] = []
            self._tensor_consumers[in_name].append(nid)
        return nid

    def build_edges(self):
        """Build edges from tensor producer/consumer relationships."""
        self.edges.clear()
        self._adj = {nid: [] for nid in self.nodes}
        self._radj = {nid: [] for nid in self.nodes}

        for tensor_name, consumers in self._tensor_consumers.items():
            if tensor_name not in self._tensor_producer:
                continue
            src_id = self._tensor_producer[tensor_name]
            if src_id not in self.nodes:
                continue
            src_node = self.nodes[src_id]
            shape = src_node.output_shapes[0] if src_node.output_shapes else ()
            nbytes = 4  # FP32 on CPU
            for d in shape:
                nbytes *= d

            for dst_id in consumers:
                if dst_id not in self.nodes:
                    continue
                edge = Edge(src_id=src_id, dst_id=dst_id,
                            tensor_name=tensor_name,
                            tensor_shape=shape, tensor_bytes=nbytes)
                self.edges.append(edge)
                if dst_id not in self._adj[src_id]:
                    self._adj[src_id].append(dst_id)
                if src_id not in self._radj[dst_id]:
                    self._radj[dst_id].append(src_id)

    def compute_topo_order(self):
        """Kahn's algorithm for topological sort."""
        in_degree = {nid: len(self._radj.get(nid, [])) for nid in self.nodes}
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._adj.get(nid, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        assert len(order) == len(self.nodes), \
            f"Topo sort incomplete: {len(order)}/{len(self.nodes)} nodes (cycle?)"
        self._topo_order = order

    @property
    def topo_order(self) -> list[int]:
        if not self._topo_order:
            self.compute_topo_order()
        return self._topo_order

    def predecessors(self, nid: int) -> list[int]:
        return self._radj.get(nid, [])

    def successors(self, nid: int) -> list[int]:
        return self._adj.get(nid, [])

    def node(self, nid: int) -> Node:
        return self.nodes[nid]

    def get_edge(self, src_id: int, dst_id: int) -> Optional[Edge]:
        for e in self.edges:
            if e.src_id == src_id and e.dst_id == dst_id:
                return e
        return None

    def remove_node(self, nid: int):
        """Remove a node and relink edges around it.

        Used for absorbing BN nodes: consumers of BN output are rewired
        to consume BN's data input instead.
        """
        node = self.nodes[nid]
        if node.input_names and node.output_names:
            data_input = node.input_names[0]
            for out_name in node.output_names:
                if out_name in self._tensor_consumers:
                    for consumer_id in self._tensor_consumers[out_name]:
                        if consumer_id not in self.nodes:
                            continue
                        cons_node = self.nodes[consumer_id]
                        cons_node.input_names = [
                            data_input if n == out_name else n
                            for n in cons_node.input_names
                        ]
                    if data_input not in self._tensor_consumers:
                        self._tensor_consumers[data_input] = []
                    self._tensor_consumers[data_input].extend(
                        [c for c in self._tensor_consumers[out_name] if c != nid]
                    )
                if data_input in self._tensor_producer:
                    self._tensor_producer[out_name] = self._tensor_producer[data_input]

        for in_name in node.input_names:
            if in_name in self._tensor_consumers:
                self._tensor_consumers[in_name] = [
                    c for c in self._tensor_consumers[in_name] if c != nid
                ]

        del self.nodes[nid]
        self._adj.pop(nid, None)
        self._radj.pop(nid, None)
        self._topo_order = []  # invalidate

    def summary(self) -> str:
        from collections import Counter
        lines = [f"EngineIR(cpu): {len(self.nodes)} nodes, {len(self.edges)} edges, T={self.T}"]
        lines.append(f"  Input: {self.model_input_shape}  Output: {self.model_output_shape}")
        op_counts = Counter(n.op_type.name for n in self.nodes.values())
        lines.append(f"  Ops: {dict(op_counts.most_common())}")
        lines.append(f"  Fusion groups: {len(self.fusion_groups)}")
        kv_counts = Counter(n.assigned_kernel.name for n in self.nodes.values()
                            if n.assigned_kernel != CPUKernelVariant.ZeroCost)
        if kv_counts:
            lines.append(f"  Kernels: {dict(kv_counts.most_common())}")
        return "\n".join(lines)
