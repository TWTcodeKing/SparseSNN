"""Memory planning for sengine_cpu.

Tensor lifetime analysis + greedy first-fit pool allocation.
All FP32, 64-byte alignment (cache line).

Self-contained — no imports from sengine/.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from sengine_cpu.ir import EngineIR
from sengine_cpu.logger import log


@dataclass
class MemSlot:
    slot_id: int
    offset: int         # byte offset in pool
    size: int           # allocated bytes (aligned)
    tensor_name: str
    birth: int          # topo step produced
    death: int          # topo step last consumed
    is_state: bool = False


@dataclass
class MemoryPlan:
    pool_bytes: int
    slots: list[MemSlot]
    activation_bytes: int
    state_bytes: int
    weight_bytes: int


def _align(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def plan_memory(ir: EngineIR,
                dtype_bytes: int = 4,
                alignment: int = 64,
                execution_order: list[int] | None = None) -> MemoryPlan:
    """Plan memory allocation for all activation tensors.

    Args:
        ir: Optimized EngineIR with topo_order
        dtype_bytes: bytes per element (4 for FP32 on CPU)
        alignment: byte alignment for slots (64 = cache line)
        execution_order: optional custom order (from BA-MTTS scheduler)
    """
    topo = execution_order or ir.topo_order
    node_step = {nid: step for step, nid in enumerate(topo)}

    # 1. Compute tensor lifetimes
    tensor_life: dict[str, tuple[int, int, int]] = {}  # name → (birth, death, bytes)

    for step, nid in enumerate(topo):
        node = ir.nodes.get(nid)
        if node is None:
            continue
        for out_name in node.output_names:
            nbytes = dtype_bytes
            if node.output_shapes:
                for d in node.output_shapes[0]:
                    if d > 0: nbytes *= d
            tensor_life[out_name] = (step, step, nbytes)

    for step, nid in enumerate(topo):
        node = ir.nodes.get(nid)
        if node is None:
            continue
        for in_name in node.input_names:
            if in_name in tensor_life:
                birth, death, size = tensor_life[in_name]
                tensor_life[in_name] = (birth, max(death, step), size)

    # 2. State tensors (membrane): lifetime = entire execution, FP32
    state_tensors = set()
    for node in ir.nodes.values():
        if node.is_stateful and node.neuron_params:
            for out_name in node.output_names:
                state_name = f"__state_{out_name}"
                # Membrane: same spatial dims as output, always FP32
                act_bytes = tensor_life.get(out_name, (0, 0, 0))[2]
                # State is per-timestep spatial (divide by T)
                T = node.neuron_params.T or ir.T or 4
                state_bytes = max(act_bytes // T, dtype_bytes)
                tensor_life[state_name] = (0, len(topo) - 1, state_bytes)
                state_tensors.add(state_name)

    # 3. Filter: only activations (not weights)
    act_tensors = {name: (b, d, s) for name, (b, d, s) in tensor_life.items()
                   if s > 0 and name not in ir.weights}

    # 4. Greedy first-fit allocation
    sorted_tensors = sorted(act_tensors.items(), key=lambda x: (x[1][0], -x[1][2]))

    slots = []
    free_regions: list[tuple[int, int, int]] = []  # (offset, size, free_after_step)
    pool_high = 0
    slot_id = 0

    for name, (birth, death, size) in sorted_tensors:
        aligned_size = _align(size, alignment)

        # Reclaim freed regions
        available = [(off, sz) for off, sz, free_after in free_regions
                     if free_after <= birth]
        free_regions = [(off, sz, free_after) for off, sz, free_after in free_regions
                        if free_after > birth]

        # First-fit placement
        placed = False
        for off, sz in sorted(available, key=lambda x: x[0]):
            if sz >= aligned_size:
                slots.append(MemSlot(
                    slot_id=slot_id, offset=off, size=aligned_size,
                    tensor_name=name, birth=birth, death=death,
                    is_state=(name in state_tensors),
                ))
                leftover = sz - aligned_size
                if leftover > 0:
                    free_regions.append((off + aligned_size, leftover, death + 1))
                placed = True; slot_id += 1; break
            else:
                free_regions.append((off, sz, 0))

        if not placed:
            offset = _align(pool_high, alignment)
            slots.append(MemSlot(
                slot_id=slot_id, offset=offset, size=aligned_size,
                tensor_name=name, birth=birth, death=death,
                is_state=(name in state_tensors),
            ))
            pool_high = offset + aligned_size
            slot_id += 1

    # 5. Compute totals
    act_bytes = sum(s.size for s in slots if not s.is_state)
    state_bytes = sum(s.size for s in slots if s.is_state)
    weight_bytes = sum(w.nbytes for w in ir.weights.values() if hasattr(w, 'nbytes'))

    plan = MemoryPlan(
        pool_bytes=pool_high,
        slots=slots,
        activation_bytes=act_bytes,
        state_bytes=state_bytes,
        weight_bytes=weight_bytes,
    )

    log.info("Memory plan: pool=%.1f KB, act=%.1f KB, state=%.1f KB, weights=%.1f MB",
             pool_high/1024, act_bytes/1024, state_bytes/1024, weight_bytes/1e6)
    return plan
