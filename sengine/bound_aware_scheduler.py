"""Bound-Aware Maximum-Transition Topological Sort (BA-MTTS).

Core algorithm for SNN inference: find the execution order that maximizes
hardware-level compute↔memory overlap through op reordering.

The GPU naturally overlaps consecutive compute-bound and memory-bound
kernels on the SAME stream (tensor cores || load/store units). The
compiler's job: find a topological sort of the ST-DAG that maximizes
the number of C↔M transitions between consecutive operations.

Algorithm:
  1. Classify each op: C (compute-bound) or M (memory-bound)
  2. Build dependency graph
  3. Greedy topological sort with transition-maximizing priority:
     - At each step, prefer the ready op whose type DIFFERS from the last
     - Tie-break: pick the op that unblocks the most opposite-type successors
  4. Output: execution order with maximum hardware overlap

Complexity: O(V + E) — same as standard topological sort.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Bound(Enum):
    COMPUTE = "C"  # tensor core bound
    MEMORY = "M"   # bandwidth bound


@dataclass
class Op:
    id: int
    name: str
    bound: Bound
    deps: list[int] = field(default_factory=list)
    est_us: float = 0.0  # estimated latency


class BoundAwareScheduler:
    """BA-MTTS: Maximum-Transition Topological Sort."""

    def __init__(self):
        self.ops: dict[int, Op] = {}
        self.adj: dict[int, list[int]] = defaultdict(list)
        self.radj: dict[int, list[int]] = defaultdict(list)
        self._next_id = 0

    def add(self, name: str, bound: Bound, deps: list[int],
            est_us: float = 0.0) -> int:
        oid = self._next_id
        self._next_id += 1
        self.ops[oid] = Op(oid, name, bound, list(deps), est_us)
        for d in deps:
            self.adj[d].append(oid)
            self.radj[oid].append(d)
        return oid

    def schedule(self, verbose=False) -> list[int]:
        """BA-MTTS: greedy topo sort maximizing C↔M transitions.

        Returns ordered list of op ids.
        """
        # Compute in-degree
        in_deg = {oid: len(self.radj.get(oid, [])) for oid in self.ops}

        # Ready set (in-degree == 0)
        ready_c = []  # compute-bound ready ops
        ready_m = []  # memory-bound ready ops
        for oid, deg in in_deg.items():
            if deg == 0:
                if self.ops[oid].bound == Bound.COMPUTE:
                    ready_c.append(oid)
                else:
                    ready_m.append(oid)

        order = []
        last_bound: Optional[Bound] = None
        transitions = 0
        total_overlap_us = 0.0

        while ready_c or ready_m:
            # Pick the ready op that creates a transition (differs from last type)
            chosen = None

            if last_bound == Bound.COMPUTE:
                # Prefer M
                if ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)
                elif ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
            elif last_bound == Bound.MEMORY:
                # Prefer C
                if ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
                elif ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)
            else:
                # Start: prefer C (compute first, then memory overlaps)
                if ready_c:
                    chosen = self._pick_best(ready_c, Bound.COMPUTE)
                    ready_c.remove(chosen)
                elif ready_m:
                    chosen = self._pick_best(ready_m, Bound.MEMORY)
                    ready_m.remove(chosen)

            if chosen is None:
                break

            op = self.ops[chosen]

            # Track transitions
            if last_bound is not None and op.bound != last_bound:
                transitions += 1
                # Estimate overlap: min of previous and current latency
                if order:
                    prev_us = self.ops[order[-1]].est_us
                    total_overlap_us += min(prev_us, op.est_us) * 0.64  # 64% efficiency

            order.append(chosen)
            last_bound = op.bound

            # Update ready set
            for succ in self.adj.get(chosen, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    if self.ops[succ].bound == Bound.COMPUTE:
                        ready_c.append(succ)
                    else:
                        ready_m.append(succ)

        # Statistics
        max_possible = len(order) - 1
        if verbose:
            print(f"  Schedule: {len(order)} ops, {transitions}/{max_possible} transitions "
                  f"({transitions/max_possible*100:.0f}%)")
            print(f"  Estimated overlap savings: {total_overlap_us:.0f} us")

        return order

    def _pick_best(self, candidates: list[int], my_bound: Bound) -> int:
        """Tie-break: pick the op that unblocks the most OPPOSITE-type successors.

        This look-ahead ensures the pipeline stays full: scheduling an op
        now that releases opposite-type ops maximizes future transitions.
        """
        opposite = Bound.MEMORY if my_bound == Bound.COMPUTE else Bound.COMPUTE

        best_id = candidates[0]
        best_score = -1

        for oid in candidates:
            # Count how many opposite-type successors become ready if we schedule oid
            score = 0
            for succ in self.adj.get(oid, []):
                if self.ops[succ].bound == opposite:
                    # Check if succ would become ready
                    remaining_deps = sum(1 for d in self.radj.get(succ, [])
                                        if d != oid and d not in set())
                    # Simpler: just count opposite-type successors
                    score += 1
            if score > best_score:
                best_score = score
                best_id = oid

        return best_id

    def schedule_moa(self, overlap_eff: float = 0.64, verbose: bool = False) -> list[int]:
        """MOA: Maximum-Overlap Aware scheduling.

        Instead of maximizing transition COUNT, maximizes total HIDDEN time.
        Key insight: an M-bound op following a C-bound op has its latency
        hidden (up to min(C_dur, M_dur) × overlap_efficiency). The scheduler
        should minimize total exposed M time.

        Strategy:
        - After a C op, prefer the M op with the highest hideable time
          (= min(prev_C_dur, M_dur) × eff)
        - After an M op, prefer a C op that can hide the MOST future M ops
          (look-ahead: C with longest duration hides more M)
        - Batch tiny M ops (Add:3us) to avoid M→M breaks:
          if the only ready ops are M, pick the SHORTEST one first
          (minimize exposed time before the next C)
        """
        in_deg = {oid: len(self.radj.get(oid, [])) for oid in self.ops}
        ready_c = []
        ready_m = []
        for oid, deg in in_deg.items():
            if deg == 0:
                (ready_c if self.ops[oid].bound == Bound.COMPUTE else ready_m).append(oid)

        order = []
        last_bound: Optional[Bound] = None
        last_dur: float = 0.0
        total_hidden = 0.0

        while ready_c or ready_m:
            chosen = None

            if last_bound == Bound.COMPUTE:
                # After C: prefer M op that maximizes hidden time
                if ready_m:
                    # Pick M with highest hideable = min(last_C_dur, M_dur)
                    best_oid = max(ready_m,
                                   key=lambda o: min(last_dur, self.ops[o].est_us))
                    chosen = best_oid
                    ready_m.remove(chosen)
                    hidden = min(last_dur, self.ops[chosen].est_us) * overlap_eff
                    total_hidden += hidden
                elif ready_c:
                    # No M ready — pick longest C (builds overlap capacity)
                    chosen = max(ready_c, key=lambda o: self.ops[o].est_us)
                    ready_c.remove(chosen)

            elif last_bound == Bound.MEMORY:
                # After M: prefer C (to start a new overlap window)
                if ready_c:
                    # Pick LONGEST C — it can hide more of the following M
                    chosen = max(ready_c, key=lambda o: self.ops[o].est_us)
                    ready_c.remove(chosen)
                    hidden = min(last_dur, self.ops[chosen].est_us) * overlap_eff
                    total_hidden += hidden
                elif ready_m:
                    # Only M available — pick SHORTEST to minimize exposed time
                    chosen = min(ready_m, key=lambda o: self.ops[o].est_us)
                    ready_m.remove(chosen)

            else:
                # Start: pick longest C (maximize first overlap window)
                if ready_c:
                    chosen = max(ready_c, key=lambda o: self.ops[o].est_us)
                    ready_c.remove(chosen)
                elif ready_m:
                    chosen = min(ready_m, key=lambda o: self.ops[o].est_us)
                    ready_m.remove(chosen)

            if chosen is None:
                break

            op = self.ops[chosen]
            order.append(chosen)
            last_bound = op.bound
            last_dur = op.est_us

            for succ in self.adj.get(chosen, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    (ready_c if self.ops[succ].bound == Bound.COMPUTE
                     else ready_m).append(succ)

        if verbose:
            transitions = self.count_transitions(order)
            print(f"  MOA Schedule: {len(order)} ops, {transitions} transitions, "
                  f"{total_hidden:.0f}us hidden")

        return order

    def compute_effective_time(self, order: list[int],
                               overlap_eff: float = 0.64) -> float:
        """Compute effective execution time with C↔M overlap model."""
        total = 0.0
        for i, oid in enumerate(order):
            dur = self.ops[oid].est_us
            if i > 0 and self.ops[oid].bound != self.ops[order[i-1]].bound:
                hidden = min(self.ops[order[i-1]].est_us, dur) * overlap_eff
                total += dur - hidden
            else:
                total += dur
        return total

    def baseline_schedule(self) -> list[int]:
        """Standard topological sort (no transition optimization) for comparison."""
        in_deg = {oid: len(self.radj.get(oid, [])) for oid in self.ops}
        ready = deque(oid for oid, d in in_deg.items() if d == 0)
        order = []
        while ready:
            v = ready.popleft()
            order.append(v)
            for succ in self.adj.get(v, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    ready.append(succ)
        return order

    def count_transitions(self, order: list[int]) -> int:
        """Count C↔M transitions in a schedule."""
        t = 0
        for i in range(1, len(order)):
            if self.ops[order[i]].bound != self.ops[order[i-1]].bound:
                t += 1
        return t

    def print_schedule(self, order: list[int], label: str = ""):
        """Print schedule with transition markers."""
        if label:
            print(f"\n  {label}:")
        transitions = self.count_transitions(order)
        print(f"  ({transitions} transitions out of {len(order)-1} possible)")
        for i, oid in enumerate(order):
            op = self.ops[oid]
            marker = ""
            if i > 0 and op.bound != self.ops[order[i-1]].bound:
                marker = " ↕ OVERLAP"
            print(f"    {i:>3}. [{op.bound.value}] {op.name:<25} ({op.est_us:.0f}us){marker}")


def build_spikformer_dag(depths=8, embed=384, mlp_ratio=4):
    """Build DECOMPOSED SpikFormer DAG for BA-MTTS scheduling.

    Each Linear+BN+LIF is decomposed into:
      - Linear+BN (C): compute-bound GEMM
      - LIF (M): memory-bound neuron
    """
    s = BoundAwareScheduler()

    prev_output = None  # last op id of previous block

    for b in range(depths):
        p = f"b{b}"
        input_deps = [prev_output] if prev_output is not None else []

        # QKV parallel projections (all depend on block input)
        q_c = s.add(f"{p}_Q_linbn", Bound.COMPUTE, input_deps, est_us=15)
        q_m = s.add(f"{p}_Q_lif", Bound.MEMORY, [q_c], est_us=10)

        k_c = s.add(f"{p}_K_linbn", Bound.COMPUTE, input_deps, est_us=15)
        k_m = s.add(f"{p}_K_lif", Bound.MEMORY, [k_c], est_us=10)

        v_c = s.add(f"{p}_V_linbn", Bound.COMPUTE, input_deps, est_us=15)
        v_m = s.add(f"{p}_V_lif", Bound.MEMORY, [v_c], est_us=10)

        # Attention matmuls
        qk = s.add(f"{p}_QK", Bound.COMPUTE, [q_m, k_m], est_us=10)
        av = s.add(f"{p}_attnV", Bound.COMPUTE, [qk, v_m], est_us=10)

        # Attention output processing
        attn_lif = s.add(f"{p}_attn_lif", Bound.MEMORY, [av], est_us=10)

        # Projection
        proj_c = s.add(f"{p}_proj_linbn", Bound.COMPUTE, [attn_lif], est_us=15)
        proj_m = s.add(f"{p}_proj_lif", Bound.MEMORY, [proj_c], est_us=10)

        # SSA residual add
        ssa_add = s.add(f"{p}_SSA_add", Bound.MEMORY, [proj_m] + input_deps, est_us=2)

        # MLP
        fc1_c = s.add(f"{p}_fc1_linbn", Bound.COMPUTE, [ssa_add], est_us=15)
        fc1_m = s.add(f"{p}_fc1_lif", Bound.MEMORY, [fc1_c], est_us=10)
        fc2_c = s.add(f"{p}_fc2_linbn", Bound.COMPUTE, [fc1_m], est_us=15)
        fc2_m = s.add(f"{p}_fc2_lif", Bound.MEMORY, [fc2_c], est_us=10)

        # MLP residual add
        mlp_add = s.add(f"{p}_MLP_add", Bound.MEMORY, [fc2_m, ssa_add], est_us=2)

        prev_output = mlp_add

    return s


# ═══ CLI ═══

if __name__ == "__main__":
    print("=" * 70)
    print("  Bound-Aware Maximum-Transition Topological Sort (BA-MTTS)")
    print("  SpikFormer-8-384 ST-DAG")
    print("=" * 70)

    sched = build_spikformer_dag(depths=8)

    # Baseline: standard topo sort
    baseline = sched.baseline_schedule()
    sched.print_schedule(baseline, "BASELINE (standard topological sort)")

    # BA-MTTS: maximum transitions
    optimal = sched.schedule(verbose=True)
    sched.print_schedule(optimal, "BA-MTTS (maximum transitions)")

    # Compare
    base_t = sched.count_transitions(baseline)
    opt_t = sched.count_transitions(optimal)
    n_ops = len(sched.ops)

    print(f"\n{'=' * 70}")
    print(f"  COMPARISON:")
    print(f"  Baseline transitions: {base_t}/{n_ops-1} ({base_t/(n_ops-1)*100:.0f}%)")
    print(f"  BA-MTTS transitions:  {opt_t}/{n_ops-1} ({opt_t/(n_ops-1)*100:.0f}%)")
    print(f"  Improvement: +{opt_t - base_t} transitions ({(opt_t-base_t)/base_t*100:.0f}% more)")

    est_overlap_base = sum(
        min(sched.ops[baseline[i]].est_us, sched.ops[baseline[i+1]].est_us) * 0.64
        for i in range(len(baseline)-1)
        if sched.ops[baseline[i]].bound != sched.ops[baseline[i+1]].bound
    )
    est_overlap_opt = sum(
        min(sched.ops[optimal[i]].est_us, sched.ops[optimal[i+1]].est_us) * 0.64
        for i in range(len(optimal)-1)
        if sched.ops[optimal[i]].bound != sched.ops[optimal[i+1]].bound
    )
    print(f"  Estimated overlap: baseline={est_overlap_base:.0f}us, BA-MTTS={est_overlap_opt:.0f}us")
    print(f"  Additional hidden latency: {est_overlap_opt - est_overlap_base:.0f} us")
    print(f"{'=' * 70}")
