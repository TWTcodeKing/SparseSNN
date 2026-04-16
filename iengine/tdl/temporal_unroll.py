"""Temporal DAG: unroll the single-timestep OperatorDAG across T timesteps.

Creates L*T nodes with two edge types:
  - data edges: (layer_i, t) → (layer_j, t)  for each original edge, per timestep
  - state edges: (neuron_n, t-1) → (neuron_n, t)  for stateful neurons across timesteps

The temporal DAG is the input to the SliceGraph DP partitioner (slicegraph.py)
and encodes the TAIL diagonal wavefront schedule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from iengine.tdl.graph_ir import OperatorDAG, OpNode
from iengine.tdl.cost_model import CostModel


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class TemporalNode:
    """One operator at one timestep in the L×T temporal DAG."""
    uid: int
    layer_id: int       # index in topological order of single-timestep DAG (0..L-1)
    timestep: int       # 0..T-1
    op_node: OpNode     # reference to the original single-timestep OpNode
    cost_us: float      # estimated latency from cost model


@dataclass
class TemporalEdge:
    """Edge in the temporal DAG."""
    src_uid: int
    dst_uid: int
    edge_type: str      # "data" or "state"


class TemporalDAG:
    """L×T unrolled DAG with data and state dependencies.

    Nodes are indexed by uid. Provides lookup by (layer_id, timestep)
    and wavefront ordering for diagonal pipeline scheduling.
    """

    def __init__(self, L: int, T: int):
        self.L = L
        self.T = T
        self.nodes: dict[int, TemporalNode] = {}
        self.edges: list[TemporalEdge] = []
        self._adj: dict[int, list[int]] = {}
        self._radj: dict[int, list[int]] = {}
        # (layer_id, timestep) → uid lookup
        self._grid: dict[tuple[int, int], int] = {}

    def add_node(self, node: TemporalNode):
        self.nodes[node.uid] = node
        self._adj[node.uid] = []
        self._radj[node.uid] = []
        self._grid[(node.layer_id, node.timestep)] = node.uid

    def add_edge(self, src_uid: int, dst_uid: int, edge_type: str = "data"):
        self.edges.append(TemporalEdge(src_uid, dst_uid, edge_type))
        self._adj[src_uid].append(dst_uid)
        self._radj[dst_uid].append(src_uid)

    def uid_at(self, layer_id: int, timestep: int) -> int:
        """Get uid for a given (layer, timestep) coordinate."""
        return self._grid[(layer_id, timestep)]

    def node_at(self, layer_id: int, timestep: int) -> TemporalNode:
        return self.nodes[self.uid_at(layer_id, timestep)]

    def predecessors(self, uid: int) -> list[int]:
        return list(self._radj[uid])

    def successors(self, uid: int) -> list[int]:
        return list(self._adj[uid])

    def wavefront_order(self) -> list[list[int]]:
        """Anti-diagonal wavefronts: wavefront w = {(i, t) : i + t == w}.

        Returns list of wavefronts, each a list of uids.
        Wavefront 0 has only (0, 0).
        Wavefront L+T-2 has only (L-1, T-1).
        Nodes within a wavefront are independent and can execute in parallel.
        """
        num_wavefronts = self.L + self.T - 1
        wavefronts: list[list[int]] = [[] for _ in range(num_wavefronts)]
        for (layer_id, timestep), uid in self._grid.items():
            w = layer_id + timestep
            wavefronts[w].append(uid)
        return wavefronts

    def data_edges(self) -> list[TemporalEdge]:
        return [e for e in self.edges if e.edge_type == "data"]

    def state_edges(self) -> list[TemporalEdge]:
        return [e for e in self.edges if e.edge_type == "state"]

    def layer_cost(self, layer_id: int) -> float:
        """Total cost across all timesteps for one layer (same per timestep)."""
        return self.nodes[self._grid[(layer_id, 0)]].cost_us * self.T

    def stage_cost_per_timestep(self, layer_ids: list[int]) -> float:
        """Cost of executing a set of layers for one timestep."""
        return sum(self.nodes[self._grid[(lid, 0)]].cost_us for lid in layer_ids)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def __repr__(self):
        nd = sum(1 for e in self.edges if e.edge_type == "data")
        ns = sum(1 for e in self.edges if e.edge_type == "state")
        return (f"TemporalDAG(L={self.L}, T={self.T}, nodes={self.num_nodes}, "
                f"data_edges={nd}, state_edges={ns})")


# ===================================================================
# Builder
# ===================================================================

def build_temporal_dag(dag: OperatorDAG, T: int,
                       cost_model: CostModel) -> TemporalDAG:
    """Unroll a single-timestep OperatorDAG into an L×T TemporalDAG.

    Args:
        dag:        Single-timestep operator DAG from extract_dag().
        T:          Number of SNN timesteps.
        cost_model: For per-node latency estimation.

    Returns:
        TemporalDAG with L*T nodes, data edges per-timestep,
        and state edges across timesteps for stateful neurons.
    """
    topo = dag.topological_order()
    L = len(topo)
    tdag = TemporalDAG(L, T)

    # Map original node id → layer index in topological order
    nid_to_layer = {nid: idx for idx, nid in enumerate(topo)}

    # Pre-compute per-node costs (same for every timestep)
    costs = cost_model.estimate_all(dag)

    # 1. Create L*T nodes
    uid = 0
    for t in range(T):
        for layer_id, nid in enumerate(topo):
            op_node = dag.nodes[nid]
            tnode = TemporalNode(
                uid=uid,
                layer_id=layer_id,
                timestep=t,
                op_node=op_node,
                cost_us=costs[nid],
            )
            tdag.add_node(tnode)
            uid += 1

    # 2. Data edges: for each original edge (i→j), replicate at every timestep
    for edge in dag.edges:
        src_layer = nid_to_layer[edge.src_id]
        dst_layer = nid_to_layer[edge.dst_id]
        for t in range(T):
            src_uid = tdag.uid_at(src_layer, t)
            dst_uid = tdag.uid_at(dst_layer, t)
            tdag.add_edge(src_uid, dst_uid, "data")

    # 3. State edges: for each stateful (neuron) node, add (n, t-1) → (n, t)
    stateful_layers = []
    for nid in topo:
        if dag.nodes[nid].is_stateful:
            stateful_layers.append(nid_to_layer[nid])

    for layer_id in stateful_layers:
        for t in range(1, T):
            src_uid = tdag.uid_at(layer_id, t - 1)
            dst_uid = tdag.uid_at(layer_id, t)
            tdag.add_edge(src_uid, dst_uid, "state")

    return tdag
