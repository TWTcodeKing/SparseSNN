"""Memory planning for SpikeEngine.

Computes tensor lifetimes from topological execution order and assigns
memory pool offsets via greedy first-fit with 256-byte alignment.
"""

from __future__ import annotations
from dataclasses import dataclass

from sengine_edge.ir import EngineIR, OpType


@dataclass
class MemSlot:
    """A memory slot in the pool."""
    slot_id: int
    offset: int       # byte offset into pool
    size: int         # allocated bytes
    tensor_name: str  # which tensor uses this slot
    birth: int        # topo step when produced
    death: int        # topo step when last consumed


@dataclass
class MemoryPlan:
    """Output of memory planning."""
    pool_bytes: int           # total pool size
    slots: list[MemSlot]      # per-tensor slot assignments
    activation_bytes: int     # activation-only memory
    state_bytes: int          # neuron membrane state memory
    weight_bytes: int         # compressed weight memory (separate)


def plan_memory(ir: EngineIR, dtype_bytes: int = 2, alignment: int = 256,
                execution_order: list[int] | None = None) -> MemoryPlan:
    """Compute memory pool layout from the EngineIR.

    Args:
        ir: Optimized EngineIR with topo_order computed.
        dtype_bytes: bytes per element for activations (2 for FP16).
        alignment: byte alignment for each slot (256 for cuSPARSELt compat).
        execution_order: optional STAKG-linearized node execution order.
            If provided, tensor lifetimes use this order instead of topo_order.
            Within interleaved groups, co-scheduled nodes share the same step.

    Returns:
        MemoryPlan with pool size and slot assignments.
    """
    topo = execution_order if execution_order is not None else ir.topo_order

    # Step 1: Compute tensor lifetimes
    # tensor_name → (birth_step, death_step, size_bytes)
    tensor_life: dict[str, tuple[int, int, int]] = {}

    # Map topo step for each node
    node_step = {nid: step for step, nid in enumerate(topo)}

    for step, nid in enumerate(topo):
        node = ir.nodes[nid]
        # Output tensors are born at this step
        for out_name in node.output_names:
            # Estimate size from output shapes or from edge info
            size = _estimate_tensor_bytes(ir, out_name, nid, dtype_bytes)
            tensor_life[out_name] = (step, step, size)  # birth = death initially

    # Update death step based on consumers
    for step, nid in enumerate(topo):
        node = ir.nodes[nid]
        for in_name in node.input_names:
            if in_name in tensor_life:
                birth, death, size = tensor_life[in_name]
                tensor_life[in_name] = (birth, max(death, step), size)

    # Step 2: Separate activation tensors from state tensors
    # State tensors (neuron membrane) persist for the entire execution
    state_tensors = set()
    for node in ir.nodes.values():
        if node.is_stateful and node.neuron_params:
            # Membrane state: same shape as output but FP32 (4 bytes/element)
            for out_name in node.output_names:
                state_name = f"__state_{out_name}"
                if out_name in tensor_life:
                    _, _, act_size = tensor_life[out_name]
                    state_size = act_size * 2  # FP32 = 2x FP16
                    tensor_life[state_name] = (0, len(topo) - 1, state_size)
                    state_tensors.add(state_name)

    # Filter out zero-size or weight tensors (weights are stored separately)
    activation_tensors = {
        name: (birth, death, size)
        for name, (birth, death, size) in tensor_life.items()
        if size > 0 and name not in ir.weights
    }

    # Step 3: Greedy first-fit allocation
    # Sort by birth step, then by size (large first for better packing)
    sorted_tensors = sorted(
        activation_tensors.items(),
        key=lambda x: (x[1][0], -x[1][2])
    )

    slots: list[MemSlot] = []
    # Track free regions: list of (offset, size, available_after_step)
    free_regions: list[tuple[int, int, int]] = []
    pool_high_water = 0

    for name, (birth, death, size) in sorted_tensors:
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        if aligned_size == 0:
            aligned_size = alignment  # minimum 1 block

        # Reclaim regions that are freed by this step
        available = [(off, sz) for off, sz, free_after in free_regions if free_after <= birth]
        free_regions = [(off, sz, free_after) for off, sz, free_after in free_regions
                        if free_after > birth]

        # Try first-fit in reclaimed regions
        placed = False
        for off, sz in sorted(available, key=lambda x: x[0]):
            if sz >= aligned_size:
                slot = MemSlot(
                    slot_id=len(slots),
                    offset=off,
                    size=aligned_size,
                    tensor_name=name,
                    birth=birth,
                    death=death,
                )
                slots.append(slot)
                # If there's leftover space, return it to free regions
                leftover = sz - aligned_size
                if leftover > 0:
                    free_regions.append((off + aligned_size, leftover, death + 1))
                placed = True
                break
            else:
                # Region too small, return to free regions
                free_regions.append((off, sz, 0))

        if not placed:
            # Allocate at the end
            offset = pool_high_water
            aligned_offset = ((offset + alignment - 1) // alignment) * alignment
            slot = MemSlot(
                slot_id=len(slots),
                offset=aligned_offset,
                size=aligned_size,
                tensor_name=name,
                birth=birth,
                death=death,
            )
            slots.append(slot)
            pool_high_water = aligned_offset + aligned_size

        # Mark region as free after death
        if placed:
            pass  # already handled
        else:
            free_regions.append((slots[-1].offset, aligned_size, death + 1))

    # Compute totals
    activation_bytes = sum(s.size for s in slots if s.tensor_name not in state_tensors)
    state_bytes = sum(s.size for s in slots if s.tensor_name in state_tensors)
    weight_bytes = sum(
        w.nbytes for w in ir.weights.values() if isinstance(w, __import__('numpy').ndarray)
    )

    return MemoryPlan(
        pool_bytes=pool_high_water,
        slots=slots,
        activation_bytes=activation_bytes,
        state_bytes=state_bytes,
        weight_bytes=weight_bytes,
    )


def _estimate_tensor_bytes(ir: EngineIR, tensor_name: str, producer_id: int,
                           dtype_bytes: int) -> int:
    """Estimate tensor size in bytes from the IR."""
    node = ir.nodes[producer_id]
    if node.output_shapes:
        shape = node.output_shapes[0]
        if shape:
            nbytes = dtype_bytes
            for d in shape:
                if d > 0:
                    nbytes *= d
            return nbytes

    # Fallback: look at edges
    for edge in ir.edges:
        if edge.tensor_name == tensor_name:
            return edge.tensor_bytes

    # Fallback: look at Conv output shape computation
    if node.op_type == OpType.Conv2d and node.conv_params:
        cp = node.conv_params
        # Need input spatial dims — approximate from predecessors
        return 0  # unknown

    return 0
