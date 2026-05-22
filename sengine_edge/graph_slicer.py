"""Graph Slicer: greedy compute-anchored deep fusion.

Given an optimized EngineIR, partitions the graph into FusionSlices:
each slice is anchored by a COMPUTE-bound op (Conv, Linear, MatMul)
and greedily absorbs all reachable MEMORY-bound successors (IF, LIF,
Add, Pool) until hitting another COMPUTE-bound op or a multi-consumer
fan-out that can't be absorbed.

Algorithm:
  1. Walk topo order. For each COMPUTE node not yet claimed:
     a. Start a new slice with this node as anchor.
     b. BFS forward through successors:
        - MEMORY/ZERO node with single consumer → absorb into slice
        - MEMORY node that feeds into ANOTHER slice's anchor → stop
        - COMPUTE node → stop (it starts its own slice)
        - Multi-consumer MEMORY node → absorb only if ALL consumers
          are already in this slice or are the next COMPUTE anchor
     c. Record the slice: (anchor_nid, [absorbed_mem_nids], pattern_str)

  2. Each slice becomes a single fused kernel at code generation time.

Output: list of FusionSlice objects describing what each kernel must compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sengine_edge.ir import EngineIR, BoundType, OpType


@dataclass
class FusionSlice:
    """A group of ops fused into a single kernel."""
    anchor_nid: int                          # Compute-bound anchor (Conv, Linear, MatMul)
    absorbed_nids: list[int] = field(default_factory=list)  # Memory-bound ops absorbed
    pattern: str = ""                        # Human-readable pattern (e.g. "Conv+BN+IF+Add+LIF")
    # Residual input: nid of the skip-connection tensor feeding the Add
    residual_nid: Optional[int] = None


def slice_graph(ir: EngineIR) -> list[FusionSlice]:
    """Partition the graph into compute-anchored fusion slices.

    Returns a list of FusionSlice, one per compute-bound node.
    Memory-bound nodes not absorbed by any slice remain standalone.
    """
    _COMPUTE = {BoundType.COMPUTE}
    _MEMORY = {BoundType.MEMORY}
    # MaxPool/GlobalAvgPool excluded: they change spatial dimensions and have
    # no compatible fused kernel template with Conv anchors. They run standalone.
    _ABSORBABLE_OPS = {OpType.IF, OpType.LIF, OpType.MS, OpType.Add,
                       OpType.Scale, OpType.Mul, OpType.Sub}
    _ZERO_OPS = {OpType.Reshape, OpType.Transpose, OpType.Identity,
                 OpType.Flatten, OpType.Tile, OpType.Concat, OpType.ReduceMean}

    claimed = set()  # nids already absorbed into a slice
    slices = []

    # Process in REVERSE topo order: downstream compute ops claim their
    # immediate memory-bound successors FIRST, preventing upstream mega-chains
    # from greedily absorbing Adds that belong to closer compute anchors.
    for nid in reversed(ir.topo_order):
        node = ir.nodes.get(nid)
        if node is None:
            continue
        if nid in claimed:
            continue
        if node.bound_type not in _COMPUTE:
            continue

        # Start a new slice anchored at this compute node
        anchor = nid
        absorbed = []
        claimed.add(anchor)

        # BFS forward: absorb memory-bound successors
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

            # Skip zero-cost transparent nodes (they don't need fusion)
            if cand.op_type in _ZERO_OPS or cand.bound_type == BoundType.ZERO:
                # Pass through to their successors
                for s in ir.successors(cand_nid):
                    if s not in visited:
                        frontier.append(s)
                continue

            # Stop at compute-bound nodes (they anchor their own slice)
            if cand.bound_type in _COMPUTE:
                continue

            # Memory-bound candidate: check if we can absorb it
            if cand.op_type not in _ABSORBABLE_OPS:
                continue

            # Check: all predecessors of this candidate must be either
            # (a) the anchor, (b) already absorbed, (c) zero-cost, or
            # (d) an external input — allowed for Add (residual skip connection)
            #     This enables Conv+Add+LIF deep fusion where Add takes
            #     the conv output + an external skip connection.
            preds = ir.predecessors(cand_nid)
            can_absorb = True
            residual_src = None
            n_external = 0
            absorbed_set = set(absorbed)
            for pred_nid in preds:
                if pred_nid == anchor or pred_nid in absorbed_set:
                    continue  # in current slice
                pred = ir.nodes.get(pred_nid)
                if pred and (pred.op_type in _ZERO_OPS or pred.bound_type == BoundType.ZERO):
                    continue  # transparent
                # External predecessor (from another slice or unclaimed)
                # Add can accept ONE external input as residual/skip connection
                if cand.op_type == OpType.Add and n_external == 0:
                    residual_src = pred_nid
                    n_external += 1
                else:
                    can_absorb = False
                    break

            if not can_absorb:
                continue

            # Constraint: for Add nodes, only absorb if the anchor is a
            # DIRECT predecessor. This prevents mega-chains from greedily
            # absorbing Adds that should belong to a closer compute op.
            if cand.op_type == OpType.Add:
                anchor_is_direct = anchor in ir.predecessors(cand_nid)
                absorbed_is_direct = any(ab in ir.predecessors(cand_nid) for ab in absorbed)
                if not anchor_is_direct and not absorbed_is_direct:
                    continue

            absorbed.append(cand_nid)
            claimed.add(cand_nid)

            # Continue BFS through this absorbed node's successors
            for s in ir.successors(cand_nid):
                if s not in visited:
                    frontier.append(s)

        # Build pattern string
        ops_in_slice = [ir.nodes[anchor].op_type.name]
        for ab_nid in absorbed:
            ab = ir.nodes.get(ab_nid)
            if ab:
                ops_in_slice.append(ab.op_type.name)

        # Find residual source for Add nodes
        res_nid = None
        for ab_nid in absorbed:
            ab = ir.nodes.get(ab_nid)
            if ab and ab.op_type == OpType.Add:
                for pred_nid in ir.predecessors(ab_nid):
                    if pred_nid != anchor and pred_nid not in claimed:
                        res_nid = pred_nid
                        break
                    # Check absorbed list
                    if pred_nid not in [anchor] + absorbed:
                        pn = ir.nodes.get(pred_nid)
                        if pn and pn.op_type not in _ZERO_OPS and pn.bound_type != BoundType.ZERO:
                            res_nid = pred_nid

        sl = FusionSlice(
            anchor_nid=anchor,
            absorbed_nids=absorbed,
            pattern="+".join(ops_in_slice),
            residual_nid=res_nid,
        )
        slices.append(sl)

    return slices


def print_slices(ir: EngineIR, slices: list[FusionSlice]):
    """Pretty-print fusion slices."""
    # Count patterns
    from collections import Counter
    patterns = Counter(s.pattern for s in slices)

    standalone_mem = set()
    claimed = set()
    for s in slices:
        claimed.add(s.anchor_nid)
        claimed.update(s.absorbed_nids)
    for nid in ir.topo_order:
        n = ir.nodes.get(nid)
        if n and n.bound_type == BoundType.MEMORY and nid not in claimed:
            standalone_mem.add(nid)

    print(f"Graph Slicing Result: {len(slices)} compute-anchored slices")
    print(f"  Standalone memory-bound ops: {len(standalone_mem)}")
    print()

    print(f"Pattern distribution:")
    for pattern, count in patterns.most_common():
        print(f"  {pattern:<50} ×{count}")
    print()

    print(f"{'#':<4} {'Anchor':<30} {'Absorbed':<6} {'Pattern':<50} {'Residual'}")
    print(f"{'-'*4} {'-'*30} {'-'*6} {'-'*50} {'-'*10}")
    for i, s in enumerate(slices):
        anchor = ir.nodes[s.anchor_nid]
        cp = anchor.conv_params
        shape = anchor.output_shapes[0] if anchor.output_shapes else ()
        shape_s = f"{shape}" if shape else "?"
        name = anchor.name[:28] if anchor.name else f"node_{s.anchor_nid}"
        res = f"#{s.residual_nid}" if s.residual_nid else ""
        print(f"{i:<4} {name:<30} {len(s.absorbed_nids):<6} {s.pattern:<50} {res}")

    if standalone_mem:
        print(f"\nStandalone memory-bound (not absorbed):")
        for nid in sorted(standalone_mem):
            n = ir.nodes[nid]
            print(f"  #{nid} {n.op_type.name} ({n.assigned_kernel.name})")
