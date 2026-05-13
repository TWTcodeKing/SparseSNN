"""Graph IR data structures for SEngine.

Extended from SpikeEngine with STAKG co-scheduling group support.
"""
from __future__ import annotations

# Module-level ref to the current EngineIR during shape propagation.
# Set by propagate_shapes(), read by _compute_output_shape() for Slice resolution.
_ir_ref = None
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from collections import deque


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
    BatchNorm = auto()       # standalone BN (before folding)
    Gemm = auto()            # ONNX Gemm (classifier FC)
    Concat = auto()
    ReduceMean = auto()
    FusedAttention = auto()      # Fused attention core (all variants)
    Resize = auto()              # Spatial upsample (nearest/bilinear) — FPN necks
    Softmax = auto()             # Softmax normalization — detection DFL head
    ILIF = auto()                # Integer LIF neuron (multi-level spikes)
    Slice = auto()               # Channel/spatial slice (C2fSpike split)


class NeuronType(Enum):
    LIF = auto()
    IF = auto()
    MS = auto()
    ILIF = auto()
    NoNeuron = auto()


class KernelVariant(Enum):
    CuDNNConv = auto()
    FusedSparseConvBNLIF = auto()
    FusedSparseConvBNLIF_TileSkip = auto()
    CuSPARSELtLinear = auto()
    CuBLASGemm = auto()
    StandaloneLIF = auto()
    Elementwise = auto()
    CuDNNPool = auto()
    TemporalMean = auto()
    LayoutTranspose = auto()
    TileRepeat = auto()
    ZeroCost = auto()
    # WaveFuse co-scheduled kernels (dispatched via dlopen'd .so)
    WaveFuseInterleaved = auto()   # Fused Conv+LIF (warp interleaving)
    WaveFuseConvBN = auto()        # Conv+BN only from wavefuse .so
    WaveFuseLIF = auto()           # LIF only from wavefuse .so
    # TileLang decomposed kernels (for BA-MTTS compute↔memory scheduling)
    TileLangConvBN = auto()        # 3×3 Conv+BN, compute-bound (no neuron)
    TileLangConv1x1BN = auto()     # 1×1 Conv+BN, compute-bound
    TileLangStemConvBN = auto()    # 7×7 stem Conv+BN (padded C_in=3→16)
    TileLangLinearBN = auto()      # Linear+BN (transformer projections)
    TileLangMatMul = auto()        # Pure matmul (attention QK^T, attn@V)
    TileLangLinearBNLIF = auto()   # Fused Linear+BN+LIF (MLP sequential chains)
    CUDAVec4IF = auto()            # Fast CUDA vec4 IF neuron (memory-bound)
    CUDAVec4LIF = auto()           # Fast CUDA vec4 LIF neuron (memory-bound)
    CUDAVec4ILIF = auto()          # Fast CUDA vec4 I-LIF neuron (integer, multi-level spikes)
    # Detection model ops (FPN/PANet neck, detection head)
    CUDAResize = auto()            # Nearest-neighbor upsample (memory-bound)
    CUDAConcat = auto()            # Channel concat (memory-bound)
    CUDASoftmax = auto()           # Softmax normalization (DFL detection head)
    # Per-timestep fused Conv+BN+IF (T=1 per launch, correct, for large batch)
    TileLangFusedConvBNIF = auto() # Fused Conv+BN+IF epilogue (T_steps=1)
    TileLangFusedConv1x1BNIF = auto()  # Fused 1×1 Conv+BN+IF (T_steps=1)
    # Depthwise conv kernels (for MaxFormer, QKFormer)
    TileLangDWConvBN = auto()          # DW Conv+BN (groups=C_in)
    TileLangFusedDWConvBNIF = auto()   # Fused DW Conv+BN+IF (T_steps=1)
    # Grouped Conv kernel (for SpikingResFormer GWFFN)
    TileLangGroupedConvBN = auto()     # Grouped Conv+BN (groups > 1, groups != C_in)
    # Fused memory-bound op + LIF (single kernel, per-CTA T-loop)
    TileLangFusedAddLIF = auto()       # Add(a,b)+LIF → single launch
    TileLangFusedPoolLIF = auto()      # MaxPool+LIF → single launch
    # Attention matmul kernels (for SpikFormer/MaxFormer attention)
    TileLangMatMulScale = auto()       # MatMul + scale epilogue, COMPUTE-bound
    TileLangFusedMatMulLIF = auto()    # Fused MatMul + LIF epilogue (T=1/launch)
    # Fused attention cores (single op replacing Reshape→MatMul→Scale→LIF chains)
    FusedSpikformerAttn = auto()       # SpikFormer: Q@K^T*scale→@V→merge→LIF
    FusedMaxformerAttn = auto()        # MaxFormer:  K^T@V→Q@result*scale→merge→LIF
    FusedDSSAAttn = auto()             # DSSA: split K/V→K^T@Q*s1→LIF→V@attn*s2→reshape
    FusedTokenQKAttn = auto()          # MS_QKFormer: sum(Q)→LIF→mul(attn,K)→merge


class BoundType(Enum):
    """Hardware resource binding for BA-MTTS scheduling.

    The GPU naturally overlaps consecutive compute-bound and memory-bound
    kernels on the same stream (tensor cores || load/store units). BA-MTTS
    maximizes C↔M transitions in the execution order.
    """
    COMPUTE = "C"   # tensor-core bound (Conv, Linear, MatMul GEMMs)
    MEMORY = "M"    # bandwidth bound (neurons, Add, Pool)
    ZERO = "Z"      # negligible cost (Reshape, Flatten, Identity)


class STAKGPattern(Enum):
    """Fusion patterns from STAKG partitioning (mirrors wavefuse FusionPattern)."""
    SOLO = "solo"
    EPILOGUE_ABSORB = "epilogue_absorb"
    INTERLEAVE_1_1 = "interleave_1_1"
    INTERLEAVE_1_N = "interleave_1_n"
    HORIZONTAL_CONV = "horizontal_conv"
    SEQUENTIAL_LIF = "sequential_lif"


class TensorLayout(Enum):
    NCHW = auto()    # 4D channel-first (N, C, H, W)
    NHWC = auto()    # 4D channel-last (N, H, W, C)
    ND = auto()      # non-spatial (2D, 3D, or any dim without H/W semantics)


@dataclass
class KernelLayoutContract:
    """Declares the expected input/output data layouts for a kernel variant."""
    input_layout: TensorLayout
    output_layout: TensorLayout


# Maps each KernelVariant to its layout contract.
# None means "inherits from predecessor" (layout-transparent).
KERNEL_CONTRACTS: dict = {
    # TileLang Conv: compiled for NHWC
    KernelVariant.TileLangConvBN:            KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangConv1x1BN:         KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangStemConvBN:         KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangFusedConvBNIF:      KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangFusedConv1x1BNIF:   KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangDWConvBN:           KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangFusedDWConvBNIF:    KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.TileLangGroupedConvBN:     KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    # cuDNN Conv/Pool: dispatch wrapper handles NHWC→NCHW internally
    KernelVariant.CuDNNConv:                  KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.CuDNNPool:                  KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    # Linear/MatMul/Neuron: layout-agnostic (2D flattened)
    KernelVariant.TileLangLinearBN:           KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.TileLangLinearBNLIF:        KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.TileLangMatMul:             KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.TileLangMatMulScale:        KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.TileLangFusedMatMulLIF:     KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.CuBLASGemm:                KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    # Fused attention: handle NHWC↔NCHW internally, no external reformats needed
    KernelVariant.FusedSpikformerAttn:       KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.FusedMaxformerAttn:        KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.FusedDSSAAttn:             KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.FusedTokenQKAttn:          KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    # Neuron kernels: flatten internally but preserve predecessor layout
    KernelVariant.CUDAVec4IF:                None,
    KernelVariant.CUDAVec4LIF:               None,
    KernelVariant.CUDAVec4ILIF:              None,
    # Detection ops: NHWC spatial ops
    KernelVariant.CUDAResize:                KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.CUDAConcat:                KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.CUDASoftmax:               None,
    # Layout-transparent (inherit from predecessor)
    KernelVariant.Elementwise:               None,
    KernelVariant.ZeroCost:                  None,
    KernelVariant.TemporalMean:              None,
    KernelVariant.TileRepeat:                None,
    KernelVariant.LayoutTranspose:           None,  # special: reformat node
    KernelVariant.StandaloneLIF:             None,
    KernelVariant.FusedSparseConvBNLIF:      KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.FusedSparseConvBNLIF_TileSkip: KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.CuSPARSELtLinear:          KernelLayoutContract(TensorLayout.ND, TensorLayout.ND),
    KernelVariant.WaveFuseInterleaved:       KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.WaveFuseConvBN:            KernelLayoutContract(TensorLayout.NHWC, TensorLayout.NHWC),
    KernelVariant.WaveFuseLIF:               None,
}


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
    # I-LIF specific (integer LIF with multi-level spikes)
    decay: float = 0.25
    max_level: int = 4


@dataclass
class AttentionParams:
    """Parameters for a fused attention custom op."""
    variant: str = ""               # "spikformer", "maxformer", "dssa"
    num_heads: int = 1
    head_dim: int = 64
    scale: float = 1.0
    H: int = 0                      # spatial dims for reshape-back (maxformer/dssa)
    W: int = 0
    # Attention LIF neuron params (embedded in the fused op)
    attn_lif_tau: float = 2.0
    attn_lif_v_threshold: float = 1.0


@dataclass
class WeightInfo:
    """Metadata about a weight tensor (actual data loaded separately)."""
    name: str = ""             # ONNX initializer name
    shape: tuple = ()
    is_sparse: bool = False    # has valid 2:4 pattern
    onnx_idx: int = -1         # index in ONNX initializer list


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
    assigned_kernel: KernelVariant = KernelVariant.ZeroCost
    layout: TensorLayout = TensorLayout.NCHW
    fusion_group_id: int = -1
    sparse_weight: bool = False
    tile_skip_enabled: bool = False
    stakg_group_id: int = -1      # which STAKG group this node belongs to
    output_mem_slot: int = -1

    # Weight references
    weight_info: Optional[WeightInfo] = None
    bias_info: Optional[WeightInfo] = None
    # BN params (populated during BN folding)
    bn_scale: Optional[list] = None
    bn_bias: Optional[list] = None

    # Timestep overrides (populated by profiler)
    timestep_overrides: dict[int, KernelVariant] = field(default_factory=dict)

    # BA-MTTS scheduling annotations (set by classify_bound pass)
    bound_type: BoundType = BoundType.ZERO
    tilelang_config: Optional[dict] = None   # autotuned {block_M, block_N, block_K, ...}
    est_latency_us: float = 0.0              # measured latency from autotuning


@dataclass
class Edge:
    src_id: int
    dst_id: int
    tensor_name: str = ""
    tensor_shape: tuple = ()
    tensor_bytes: int = 0
    layout: TensorLayout = TensorLayout.NCHW


@dataclass
class FusionGroup:
    group_id: int
    conv_node_id: int
    bn_node_id: int = -1          # -1 if BN already folded
    neuron_node_id: int = -1      # -1 if no neuron fusion
    variant: KernelVariant = KernelVariant.FusedSparseConvBNLIF
    bn_scale: Optional[list] = None
    bn_bias: Optional[list] = None


@dataclass
class STAKGGroup:
    """A kernel group from STAKG partitioning.

    Encodes which nodes are co-scheduled in a single kernel launch.
    For INTERLEAVE_1_1: Conv warps 0-3 + LIF warps 4-7 in one thread block.
    For SOLO: single node dispatched via standard cuDNN/cuBLAS path.
    """
    group_id: int
    pattern: STAKGPattern = STAKGPattern.SOLO
    compute_node_ids: list[int] = field(default_factory=list)
    epilogue_node_ids: list[int] = field(default_factory=list)
    memory_node_ids: list[int] = field(default_factory=list)
    add_node_ids: list[int] = field(default_factory=list)  # absorbed residual Add
    so_path: str = ""             # compiled .so path (for interleaved kernels)
    so_hash: str = ""             # content hash for cache invalidation
    n_sync: int = 0               # __syncthreads count for template
    smem_bytes: int = 0
    compute_warps: int = 4
    memory_warps: int = 4
    estimated_latency_us: float = 0.0
    has_skip_add: bool = False    # whether fused kernel includes residual Add

    @property
    def all_node_ids(self) -> list[int]:
        return (self.compute_node_ids + self.epilogue_node_ids +
                self.memory_node_ids + self.add_node_ids)

    @property
    def is_interleaved(self) -> bool:
        return self.pattern in (STAKGPattern.INTERLEAVE_1_1,
                                STAKGPattern.INTERLEAVE_1_N)

    @property
    def is_epilogue(self) -> bool:
        return self.pattern == STAKGPattern.EPILOGUE_ABSORB

    @property
    def is_fused(self) -> bool:
        """True if this group uses a compiled fused kernel (interleave or epilogue)."""
        return self.is_interleaved or self.is_epilogue


class EngineIR:
    """Directed acyclic graph of operators for SEngine."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []
        self.fusion_groups: list[FusionGroup] = []
        self.T: int = 4
        self.precision: str = "fp16"   # "fp16" or "fp32", global engine precision
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
        self.weights: dict[str, 'numpy.ndarray'] = {}

        # STAKG co-scheduling (populated by stakg_partition_ir)
        self.stakg_groups: list[STAKGGroup] = []
        self.stakg_schedule: list[int] = []  # linearized group execution order

    def add_node(self, node: Node) -> int:
        nid = self._next_id
        self._next_id += 1
        node.id = nid
        self.nodes[nid] = node
        self._adj[nid] = []
        self._radj[nid] = []
        # Register tensor producers/consumers
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
        # Reset adjacency for all living nodes
        self._adj = {nid: [] for nid in self.nodes}
        self._radj = {nid: [] for nid in self.nodes}

        for tensor_name, consumers in self._tensor_consumers.items():
            if tensor_name not in self._tensor_producer:
                continue  # graph input or initializer — no producer node
            src_id = self._tensor_producer[tensor_name]
            if src_id not in self.nodes:
                continue  # producer was removed
            src_node = self.nodes[src_id]
            shape = src_node.output_shapes[0] if src_node.output_shapes else ()
            nbytes = 2  # FP16
            for d in shape:
                nbytes *= d

            for dst_id in consumers:
                if dst_id not in self.nodes:
                    continue  # consumer was removed
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
        """Find edge between two specific nodes."""
        for e in self.edges:
            if e.src_id == src_id and e.dst_id == dst_id:
                return e
        return None

    def output_edge_layout(self, nid: int) -> TensorLayout:
        """Get the layout of a node's outgoing edges (all share the same layout)."""
        for e in self.edges:
            if e.src_id == nid:
                return e.layout
        return TensorLayout.NCHW  # default

    def remove_node(self, nid: int):
        """Remove a node (for absorbed BN nodes). Relinks edges around it.

        If the node has one input and one output, all consumers of the node's
        output are rewired to consume the node's input instead.
        """
        node = self.nodes[nid]
        # For BN removal: BN has one data input (from Conv) and one output (to Neuron).
        # Relink: consumers of BN output now consume BN's data input instead.
        if node.input_names and node.output_names:
            # Use the first data input (skip weight/bias inputs)
            data_input = node.input_names[0]
            for out_name in node.output_names:
                # All consumers of out_name now consume data_input
                if out_name in self._tensor_consumers:
                    for consumer_id in self._tensor_consumers[out_name]:
                        if consumer_id not in self.nodes:
                            continue
                        cons_node = self.nodes[consumer_id]
                        cons_node.input_names = [
                            data_input if n == out_name else n
                            for n in cons_node.input_names
                        ]
                    # Move consumers to the data_input's consumer list
                    if data_input not in self._tensor_consumers:
                        self._tensor_consumers[data_input] = []
                    self._tensor_consumers[data_input].extend(
                        [c for c in self._tensor_consumers[out_name] if c != nid]
                    )
                # Update producer: out_name's effective producer is data_input's producer
                if data_input in self._tensor_producer:
                    self._tensor_producer[out_name] = self._tensor_producer[data_input]

        # Remove from tracking
        for out_name in node.output_names:
            if out_name in self._tensor_producer and self._tensor_producer[out_name] == nid:
                pass  # keep remapped producer
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
        lines = [f"EngineIR: {len(self.nodes)} nodes, {len(self.edges)} edges, T={self.T}"]
        lines.append(f"  Input: {self.model_input_shape}  Output: {self.model_output_shape}")
        from collections import Counter
        op_counts = Counter(n.op_type.name for n in self.nodes.values())
        lines.append(f"  Ops: {dict(op_counts.most_common())}")
        lines.append(f"  Fusion groups: {len(self.fusion_groups)}")
        if self.stakg_groups:
            from collections import Counter
            pat_counts = Counter(g.pattern.value for g in self.stakg_groups)
            lines.append(f"  STAKG groups: {len(self.stakg_groups)} "
                         f"({dict(pat_counts.most_common())})")
            n_interleaved = sum(1 for g in self.stakg_groups if g.is_interleaved)
            lines.append(f"  Interleaved: {n_interleaved} groups")
        return "\n".join(lines)
