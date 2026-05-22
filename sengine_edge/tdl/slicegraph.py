"""SliceGraph DP partitioner: split L layers into k balanced pipeline stages.

Uses min-max dynamic programming to find the partition that minimizes
the bottleneck stage cost, then selects the optimal number of stages k*
by minimizing total pipeline makespan = (T + k - 1) * bottleneck_cost.

The partitioner operates on the topological order of the single-timestep
OperatorDAG — layers are contiguous groups of operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sengine_edge.tdl.graph_ir import OperatorDAG
from sengine_edge.tdl.cost_model import CostModel, HardwareSpec
from sengine_edge.tdl.temporal_unroll import TemporalDAG


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class Partition:
    """Result of partitioning L layers into k stages."""
    k: int                              # number of stages
    boundaries: list[int]               # k+1 boundary indices [0, b1, b2, ..., L]
    stage_costs_us: list[float]         # per-stage cost (one timestep)
    bottleneck_us: float                # max stage cost
    makespan_us: float                  # (T + k - 1) * bottleneck

    @property
    def stages(self) -> list[tuple[int, int]]:
        """List of (start, end) layer index ranges per stage (end exclusive)."""
        return [(self.boundaries[i], self.boundaries[i + 1])
                for i in range(self.k)]

    def __repr__(self):
        stages_str = ', '.join(f'[{s}:{e})' for s, e in self.stages)
        return (f"Partition(k={self.k}, stages=[{stages_str}], "
                f"bottleneck={self.bottleneck_us:.1f}us, "
                f"makespan={self.makespan_us:.1f}us)")


# ===================================================================
# DP partitioner
# ===================================================================

class SliceGraphDP:
    """Min-max DP partitioner for SNN pipeline stages.

    Partitions L sequential layers into k contiguous groups minimizing
    the maximum group cost (bottleneck stage).

    Usage:
        dp = SliceGraphDP(tdag)
        part = dp.partition(k=4)
        k_star, best = dp.find_optimal_k(T=4, max_k=8)
    """

    def __init__(self, tdag: TemporalDAG):
        self.tdag = tdag
        self.L = tdag.L
        self.T = tdag.T
        # Per-layer cost (one timestep) — indexed by layer_id 0..L-1
        self._layer_costs = [
            tdag.nodes[tdag.uid_at(i, 0)].cost_us for i in range(self.L)
        ]
        # Prefix sums for O(1) range cost queries
        self._prefix = [0.0] * (self.L + 1)
        for i in range(self.L):
            self._prefix[i + 1] = self._prefix[i] + self._layer_costs[i]

    def _range_cost(self, start: int, end: int) -> float:
        """Cost of layers [start, end) for one timestep."""
        return self._prefix[end] - self._prefix[start]

    def partition(self, k: int) -> Partition:
        """Find optimal k-way contiguous partition minimizing bottleneck.

        Uses O(L^2 * k) DP with backtracking.
        """
        L = self.L
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > L:
            k = L  # can't have more stages than layers

        INF = float('inf')

        # dp[j][i] = min bottleneck partitioning layers [0..i) into j stages
        dp = [[INF] * (L + 1) for _ in range(k + 1)]
        # split[j][i] = optimal split point for backtracking
        split = [[0] * (L + 1) for _ in range(k + 1)]

        # Base case: 1 stage covering [0..i)
        for i in range(1, L + 1):
            dp[1][i] = self._range_cost(0, i)
            split[1][i] = 0

        # Fill DP table
        for j in range(2, k + 1):
            for i in range(j, L + 1):
                for s in range(j - 1, i):
                    # Split: first j-1 stages cover [0..s), last stage covers [s..i)
                    cost = max(dp[j - 1][s], self._range_cost(s, i))
                    if cost < dp[j][i]:
                        dp[j][i] = cost
                        split[j][i] = s

        # Backtrack to recover boundaries
        boundaries = [L]
        pos = L
        for j in range(k, 1, -1):
            pos = split[j][pos]
            boundaries.append(pos)
        boundaries.append(0)
        boundaries.reverse()

        # Compute per-stage costs
        stage_costs = []
        for idx in range(k):
            stage_costs.append(self._range_cost(boundaries[idx], boundaries[idx + 1]))

        bottleneck = max(stage_costs)
        makespan = (self.T + k - 1) * bottleneck

        return Partition(
            k=k,
            boundaries=boundaries,
            stage_costs_us=stage_costs,
            bottleneck_us=bottleneck,
            makespan_us=makespan,
        )

    def find_optimal_k(self, max_k: int = 16) -> tuple[int, Partition]:
        """Find k* minimizing pipeline makespan = (T + k - 1) * bottleneck.

        Returns (k_star, best_partition).
        k*=1 when the GPU is already saturated (pipelining adds overhead
        without reducing bottleneck).
        """
        best_k = 1
        best_part = self.partition(1)

        for k in range(2, min(max_k + 1, self.L + 1)):
            part = self.partition(k)
            if part.makespan_us < best_part.makespan_us:
                best_k = k
                best_part = part

        return best_k, best_part

    def quotient_graph(self, partition: Partition) -> list[list[int]]:
        """Compute stage dependency adjacency list from the partition.

        Returns adj[stage_i] = list of stage indices that stage_i depends on.
        A stage depends on another if any original edge crosses the boundary.
        """
        k = partition.k
        stages = partition.stages
        # Map layer_id → stage index
        layer_to_stage = {}
        for si, (start, end) in enumerate(stages):
            for lid in range(start, end):
                layer_to_stage[lid] = si

        # Find cross-stage dependencies from data edges in timestep 0
        deps: list[set[int]] = [set() for _ in range(k)]
        for edge in self.tdag.data_edges():
            if edge.src_uid >= self.L:
                break  # only need timestep 0 edges
            src_node = self.tdag.nodes[edge.src_uid]
            dst_node = self.tdag.nodes[edge.dst_uid]
            if src_node.timestep != 0 or dst_node.timestep != 0:
                continue
            src_stage = layer_to_stage[src_node.layer_id]
            dst_stage = layer_to_stage[dst_node.layer_id]
            if src_stage != dst_stage:
                deps[dst_stage].add(src_stage)

        return [sorted(d) for d in deps]
