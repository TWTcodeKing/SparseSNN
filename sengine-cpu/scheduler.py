"""BA-MTTS scheduler for sengine-cpu.

Bound-Aware Maximum-Transition Topological Sort: greedy topo sort that
maximizes COMPUTE↔MEMORY transitions to exploit CPU prefetch overlap
and out-of-order execution.

Self-contained — no imports from sengine/.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque

from sengine_cpu.ir import EngineIR, BoundType, OpType
from sengine_cpu.logger import log


class Bound:
    COMPUTE = "C"
    MEMORY = "M"


@dataclass
class Op:
    id: int
    name: str
    bound: str       # "C" or "M"
    deps: list[int]  # predecessor op ids
    est_us: float = 0.0


class BoundAwareScheduler:
    """BA-MTTS scheduler adapted for CPU overlap model."""

    def __init__(self):
        self.ops: dict[int, Op] = {}
        self.adj: dict[int, list[int]] = {}   # successors
        self.radj: dict[int, list[int]] = {}  # predecessors
        # CPU overlap efficiency (prefetch + OoO). Lower than GPU (0.64).
        self.overlap_eff = 0.4

    def add_op(self, op: Op):
        self.ops[op.id] = op
        self.adj.setdefault(op.id, [])
        self.radj.setdefault(op.id, [])
        for dep in op.deps:
            self.adj.setdefault(dep, []).append(op.id)
            self.radj.setdefault(op.id, []).append(dep)

    def schedule(self) -> list[int]:
        """Greedy topo sort maximizing C↔M transitions."""
        in_deg = {oid: len(self.radj.get(oid, [])) for oid in self.ops}

        ready_c = [oid for oid, d in in_deg.items()
                    if d == 0 and self.ops[oid].bound == Bound.COMPUTE]
        ready_m = [oid for oid, d in in_deg.items()
                    if d == 0 and self.ops[oid].bound == Bound.MEMORY]

        order = []
        last_bound = None
        transitions = 0

        while ready_c or ready_m:
            chosen = None
            if last_bound == Bound.COMPUTE:
                if ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)
                elif ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
            elif last_bound == Bound.MEMORY:
                if ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
                elif ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)
            else:
                if ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
                elif ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)

            if chosen is None:
                break

            op = self.ops[chosen]
            if last_bound is not None and op.bound != last_bound:
                transitions += 1

            order.append(chosen)
            last_bound = op.bound

            for succ in self.adj.get(chosen, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    if self.ops[succ].bound == Bound.COMPUTE:
                        ready_c.append(succ)
                    else:
                        ready_m.append(succ)

        max_possible = max(len(order) - 1, 1)
        log.info("BA-MTTS: %d/%d transitions (%.0f%%), %d ops scheduled",
                 transitions, max_possible, 100*transitions/max_possible, len(order))
        return order

    def _pick_best(self, candidates: list[int], my_bound: str) -> int:
        """Pick op that unblocks most opposite-type successors."""
        opposite = Bound.MEMORY if my_bound == Bound.COMPUTE else Bound.COMPUTE
        best_id = candidates[0]
        best_score = -1
        for oid in candidates:
            score = sum(1 for s in self.adj.get(oid, [])
                        if self.ops.get(s) and self.ops[s].bound == opposite)
            if score > best_score:
                best_score = score
                best_id = oid
        return best_id

    def baseline_schedule(self) -> list[int]:
        """Standard topological sort (no optimization)."""
        in_deg = {oid: len(self.radj.get(oid, [])) for oid in self.ops}
        queue = deque(oid for oid, d in in_deg.items() if d == 0)
        order = []
        while queue:
            oid = queue.popleft()
            order.append(oid)
            for succ in self.adj.get(oid, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)
        return order


def build_schedule_from_ir(ir: EngineIR) -> list[int]:
    """Build a BA-MTTS schedule from an optimized EngineIR."""
    scheduler = BoundAwareScheduler()

    for nid in ir.topo_order:
        node = ir.nodes.get(nid)
        if node is None:
            continue
        if node.bound_type == BoundType.ZERO:
            continue  # skip zero-cost nodes

        bound = Bound.COMPUTE if node.bound_type == BoundType.COMPUTE else Bound.MEMORY
        deps = [p for p in ir.predecessors(nid)
                if ir.nodes.get(p) and ir.nodes[p].bound_type != BoundType.ZERO]

        scheduler.add_op(Op(
            id=nid, name=node.name, bound=bound,
            deps=deps, est_us=node.est_latency_us,
        ))

    return scheduler.schedule()
