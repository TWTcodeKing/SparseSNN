"""Graph Slicer: compute-anchored deep fusion for sengine_cpu.

Partitions the graph into FusionSlices: each anchored by a COMPUTE-bound op
(Conv, Linear, MatMul) absorbing reachable MEMORY-bound successors (IF, LIF,
Add, Pool) via BFS. Reverse topo order prevents upstream mega-chains.

Self-contained — no imports from sengine/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from sengine_cpu.ir import EngineIR, BoundType, OpType


@dataclass
class FusionSlice:
    """A group of ops fused into a single kernel."""
    anchor_nid: int
    absorbed_nids: list[int] = field(default_factory=list)
    pattern: str = ""
    residual_nid: Optional[int] = None


def slice_graph(ir: EngineIR) -> list[FusionSlice]:
    """Partition graph into compute-anchored fusion slices."""
    _ABSORBABLE_OPS = {OpType.IF, OpType.LIF, OpType.MS, OpType.Add,
                       OpType.MaxPool, OpType.GlobalAvgPool, OpType.Scale,
                       OpType.Mul, OpType.Sub}
    _ZERO_OPS = {OpType.Reshape, OpType.Transpose, OpType.Identity,
                 OpType.Flatten, OpType.Tile, OpType.Concat, OpType.ReduceMean}

    claimed = set()
    slices = []

    for nid in reversed(ir.topo_order):
        node = ir.nodes.get(nid)
        if node is None or nid in claimed or node.bound_type != BoundType.COMPUTE:
            continue

        anchor = nid
        absorbed = []
        claimed.add(anchor)

        frontier = list(ir.successors(anchor))
        visited = {anchor}

        while frontier:
            cand_nid = frontier.pop(0)
            if cand_nid in visited or cand_nid in claimed:
                continue
            visited.add(cand_nid)

            cand = ir.nodes.get(cand_nid)
            if cand is None:
                continue

            # Zero-cost: pass through
            if cand.op_type in _ZERO_OPS or cand.bound_type == BoundType.ZERO:
                for s in ir.successors(cand_nid):
                    if s not in visited:
                        frontier.append(s)
                continue

            # Compute: stop (it anchors its own slice)
            if cand.bound_type == BoundType.COMPUTE:
                continue

            # Must be absorbable
            if cand.op_type not in _ABSORBABLE_OPS:
                continue

            # Check predecessors: all must be in-slice, zero-cost, or 1 external (Add residual)
            preds = ir.predecessors(cand_nid)
            can_absorb = True
            n_external = 0
            absorbed_set = set(absorbed)
            for pred_nid in preds:
                if pred_nid == anchor or pred_nid in absorbed_set:
                    continue
                pred = ir.nodes.get(pred_nid)
                if pred and (pred.op_type in _ZERO_OPS or pred.bound_type == BoundType.ZERO):
                    continue
                if cand.op_type == OpType.Add and n_external == 0:
                    n_external += 1
                else:
                    can_absorb = False; break

            if not can_absorb:
                continue

            # Add constraint: anchor or absorbed must be direct predecessor
            if cand.op_type == OpType.Add:
                anchor_is_direct = anchor in ir.predecessors(cand_nid)
                absorbed_is_direct = any(ab in ir.predecessors(cand_nid) for ab in absorbed)
                if not anchor_is_direct and not absorbed_is_direct:
                    continue

            absorbed.append(cand_nid)
            claimed.add(cand_nid)
            for s in ir.successors(cand_nid):
                if s not in visited:
                    frontier.append(s)

        # Build pattern
        ops_in_slice = [ir.nodes[anchor].op_type.name]
        for ab_nid in absorbed:
            ab = ir.nodes.get(ab_nid)
            if ab:
                ops_in_slice.append(ab.op_type.name)

        # Find residual source
        res_nid = None
        for ab_nid in absorbed:
            ab = ir.nodes.get(ab_nid)
            if ab and ab.op_type == OpType.Add:
                for pred_nid in ir.predecessors(ab_nid):
                    if pred_nid != anchor and pred_nid not in claimed:
                        res_nid = pred_nid; break
                    if pred_nid not in [anchor] + absorbed:
                        pn = ir.nodes.get(pred_nid)
                        if pn and pn.op_type not in _ZERO_OPS and pn.bound_type != BoundType.ZERO:
                            res_nid = pred_nid

        slices.append(FusionSlice(
            anchor_nid=anchor, absorbed_nids=absorbed,
            pattern="+".join(ops_in_slice), residual_nid=res_nid))

    return slices
