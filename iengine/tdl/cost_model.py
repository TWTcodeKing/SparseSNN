"""Analytical roofline cost model for SNN operator latency estimation.

Estimates per-operator latency using max(compute_time, memory_time) roofline
model. Calibrated against nsys profiling data from RTX 4090.

Used by the SliceGraph DP partitioner (Phase 2) for balanced stage assignment
and by the PTB scheduler (Phase 3) for SM allocation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from iengine.tdl.graph_ir import OpNode, OperatorDAG


# ===================================================================
# Hardware specifications
# ===================================================================

@dataclass
class HardwareSpec:
    """GPU hardware parameters for roofline estimation."""
    name: str
    sm_count: int
    fp16_tflops: float     # peak FP16 tensor core TFLOPS
    mem_bw_gb_s: float     # peak DRAM bandwidth GB/s
    l2_cache_mb: float     # L2 cache size MB

    @classmethod
    def rtx_4090(cls) -> HardwareSpec:
        return cls('RTX 4090', sm_count=128, fp16_tflops=165.2,
                   mem_bw_gb_s=1008.0, l2_cache_mb=72.0)

    @classmethod
    def rtx_3090(cls) -> HardwareSpec:
        return cls('RTX 3090', sm_count=82, fp16_tflops=71.0,
                   mem_bw_gb_s=936.0, l2_cache_mb=6.0)

    @classmethod
    def orin_agx(cls) -> HardwareSpec:
        """Jetson Orin AGX 64GB — Ampere GA10B, 2048 CUDA + 64 Tensor Cores."""
        return cls('Jetson Orin AGX', sm_count=16, fp16_tflops=5.3,
                   mem_bw_gb_s=204.8, l2_cache_mb=4.0)

    @classmethod
    def orin_nx(cls) -> HardwareSpec:
        """Jetson Orin NX 16GB — 1024 CUDA + 32 Tensor Cores."""
        return cls('Jetson Orin NX', sm_count=8, fp16_tflops=2.65,
                   mem_bw_gb_s=102.4, l2_cache_mb=2.0)

    @classmethod
    def orin_nano(cls) -> HardwareSpec:
        """Jetson Orin Nano 8GB — 512 CUDA + 16 Tensor Cores."""
        return cls('Jetson Orin Nano', sm_count=4, fp16_tflops=1.3,
                   mem_bw_gb_s=68.0, l2_cache_mb=1.0)

    # Alias for backward compatibility
    orin = orin_agx

    @classmethod
    def a100(cls) -> HardwareSpec:
        return cls('A100', sm_count=108, fp16_tflops=312.0,
                   mem_bw_gb_s=2039.0, l2_cache_mb=40.0)


# ===================================================================
# Calibration factors (empirical corrections from nsys profiling)
# ===================================================================

@dataclass
class CalibrationFactors:
    """Empirical correction factors from profiling.

    Roofline estimates are theoretical — these factors account for
    overhead (kernel launch, cache effects, memory controller, etc.).
    """
    conv_factor: float = 1.3      # convs slightly above roofline due to im2col
    neuron_factor: float = 1.0    # neurons are exactly at roofline ceiling
    bn_factor: float = 1.0        # BN is memory-bound, matches roofline
    elementwise_factor: float = 1.0  # add/mul are purely memory-bound
    pool_factor: float = 1.0
    linear_factor: float = 1.2
    matmul_factor: float = 1.2    # attention matmuls
    l2_hit_rate_conv: float = 0.98   # conv working set often fits L2
    l2_hit_rate_neuron: float = 0.50  # neuron streaming, partial L2 benefit


# ===================================================================
# Cost model
# ===================================================================

class CostModel:
    """Analytical roofline latency estimator for SNN operators.

    For each operator: latency_us = max(compute_us, memory_us).
    - compute_us = FLOPs / (peak_flops_per_us)
    - memory_us  = bytes  / (bandwidth_per_us)

    Accounts for L2 cache hit rates and calibration corrections.

    Measured overrides
    ------------------
    After TileLang autotuning, call ``update_from_tilelang()`` to replace
    analytical estimates with measured profiling latencies for fused
    Conv+BN+Neuron groups.  The fused cost replaces the sum of the
    individual Conv, BN, and Neuron costs.
    """

    def __init__(self, hw: HardwareSpec, dtype_bytes: int = 2,
                 calibration: Optional[CalibrationFactors] = None):
        self.hw = hw
        self.dtype_bytes = dtype_bytes
        self.cal = calibration or CalibrationFactors()

        # Precompute throughput in convenient units
        self._peak_flops_per_us = hw.fp16_tflops * 1e6   # FLOP/us
        self._mem_bw_per_us = hw.mem_bw_gb_s * 1e3        # bytes/us
        self._l2_bytes = hw.l2_cache_mb * 1024 * 1024

        # Measured overrides: node_id -> latency_us
        # When a node_id is in this dict, estimate_us returns the measured
        # value instead of the analytical estimate.
        self._measured: dict[int, float] = {}
        # Nodes whose cost is absorbed into a fused group (BN, Neuron
        # that are fused with Conv) — return 0.0 for these.
        self._absorbed: set[int] = set()

    def update_from_tilelang(
        self,
        fusion_groups: list[dict],
        tuner_results: dict[str, "TuneResult"],
    ) -> None:
        """Replace analytical costs with measured TileLang profiling data.

        For each ``conv_bn_neuron`` or ``linear_bn_neuron`` group that has
        a tuning result, the conv/linear node gets the measured latency and
        the BN + neuron nodes are marked as absorbed (cost = 0).

        Parameters
        ----------
        fusion_groups : list[dict]
            Fusion group descriptors from ``_build_fusion_groups()``.
        tuner_results : dict[str, TuneResult]
            Keyed by ``"fused_block_N"`` — the results from
            ``TileLangKernelTuner``.
        """
        for gi, group in enumerate(fusion_groups):
            key = f"fused_block_{gi}"
            if key not in tuner_results:
                continue

            latency_ms = tuner_results[key].latency_ms
            latency_us = latency_ms * 1000.0  # ms → us

            gtype = group['type']

            if gtype == 'conv_bn_neuron':
                self._measured[group['conv']] = latency_us
                self._absorbed.add(group['bn'])
                self._absorbed.add(group['neuron'])

            elif gtype == 'linear_bn_neuron':
                self._measured[group['linear']] = latency_us
                self._absorbed.add(group['bn'])
                self._absorbed.add(group['neuron'])

            elif gtype in ('conv_bn', 'linear_bn'):
                node_key = 'conv' if 'conv' in group else 'linear'
                self._measured[group[node_key]] = latency_us
                self._absorbed.add(group['bn'])

    def estimate_us(self, node: OpNode) -> float:
        """Estimate operator latency in microseconds.

        Returns measured latency if available (from ``update_from_tilelang``),
        0.0 if absorbed into a fused group, or the analytical estimate.
        """
        nid = node.id if hasattr(node, 'id') else id(node)

        # Check for measured override
        if nid in self._measured:
            return self._measured[nid]

        # Check if absorbed into a fused group
        if nid in self._absorbed:
            return 0.0

        op = node.op_type
        if op == 'conv2d':
            return self._conv2d_us(node)
        elif op == 'bn2d':
            return self._bn_us(node)
        elif op in ('if_neuron', 'lif_neuron', 'ms_neuron'):
            return self._neuron_us(node)
        elif op in ('add', 'mul', 'iand'):
            return self._elementwise_us(node)
        elif op in ('maxpool2d', 'avgpool'):
            return self._pool_us(node)
        elif op == 'linear':
            return self._linear_us(node)
        elif op == 'matmul':
            return self._matmul_us(node)
        else:
            return 0.0

    def estimate_grid_blocks(self, node: OpNode) -> int:
        """Estimate CUDA grid size (number of thread blocks) for this op.

        Used by PTB scheduler for SM allocation.
        """
        op = node.op_type
        threads_per_block = 256

        def _prod(shape):
            r = 1
            for d in shape:
                r *= d
            return r

        if op == 'conv2d':
            return max(1, (_prod(node.output_shape) + threads_per_block - 1) // threads_per_block)
        elif op in ('if_neuron', 'lif_neuron', 'ms_neuron'):
            return max(1, (_prod(node.input_shape) + threads_per_block - 1) // threads_per_block)
        elif op in ('add', 'mul', 'iand', 'bn2d'):
            return max(1, (_prod(node.output_shape) + threads_per_block - 1) // threads_per_block)
        elif op == 'linear':
            return max(1, (_prod(node.output_shape) + threads_per_block - 1) // threads_per_block)
        elif op == 'matmul':
            total = 1
            for d in node.output_shape:
                total *= d
            return max(1, (total + threads_per_block - 1) // threads_per_block)
        elif op in ('maxpool2d', 'avgpool'):
            total = 1
            for d in node.output_shape:
                total *= d
            return max(1, (total + threads_per_block - 1) // threads_per_block)
        return 1

    def estimate_all(self, dag: OperatorDAG) -> dict[int, float]:
        """Estimate latency for all nodes in DAG. Returns {node_id: us}."""
        return {nid: self.estimate_us(n) for nid, n in dag.nodes.items()}

    def total_sequential_us(self, dag: OperatorDAG) -> float:
        """Total latency if all ops run sequentially (no pipeline)."""
        return sum(self.estimate_us(n) for n in dag.nodes.values())

    # -------------------------------------------------------------------
    # Per-operator estimators
    # -------------------------------------------------------------------

    def _conv2d_us(self, node: OpNode) -> float:
        p = node.params
        B, C_in, H_in, W_in = node.input_shape
        B2, C_out, H_out, W_out = node.output_shape
        K = p.get('kernel_size', 3)
        groups = p.get('groups', 1)

        # FLOPs: 2 * K * K * C_in/groups * C_out * H_out * W_out * B
        flops = 2 * K * K * (C_in // groups) * C_out * H_out * W_out * B
        compute_us = flops / self._peak_flops_per_us

        # Memory: input + weight + output
        input_bytes = B * C_in * H_in * W_in * self.dtype_bytes
        weight_bytes = C_out * (C_in // groups) * K * K * self.dtype_bytes
        output_bytes = B * C_out * H_out * W_out * self.dtype_bytes
        total_bytes = input_bytes + weight_bytes + output_bytes

        # L2 cache effect: if working set fits, effective bandwidth is higher
        effective_bw = self._mem_bw_per_us
        if total_bytes < self._l2_bytes:
            # Approximate: L2 hit reduces effective DRAM traffic
            effective_bw = self._mem_bw_per_us / (1 - self.cal.l2_hit_rate_conv + 0.01)
        memory_us = total_bytes / effective_bw

        return max(compute_us, memory_us) * self.cal.conv_factor

    def _bn_us(self, node: OpNode) -> float:
        """BN at eval time: y = x * scale + offset (pointwise)."""
        shape = node.input_shape
        n_elements = 1
        for d in shape:
            n_elements *= d
        # Read input + write output + read scale/bias (negligible)
        total_bytes = 2 * n_elements * self.dtype_bytes
        memory_us = total_bytes / self._mem_bw_per_us
        return memory_us * self.cal.bn_factor

    def _neuron_us(self, node: OpNode) -> float:
        """Neuron: memory-bound at 0.125 FLOP/byte (from ncu profiling)."""
        shape = node.input_shape
        n_elements = 1
        for d in shape:
            n_elements *= d
        # Read input + write output + read/write membrane state
        total_bytes = (2 * n_elements + 2 * n_elements) * self.dtype_bytes
        # Partial L2 benefit
        effective_bw = self._mem_bw_per_us / (1 - self.cal.l2_hit_rate_neuron + 0.01)
        memory_us = total_bytes / effective_bw
        return memory_us * self.cal.neuron_factor

    def _elementwise_us(self, node: OpNode) -> float:
        """Add/Mul/IAND: read 2 inputs + write 1 output."""
        shape = node.input_shape
        n_elements = 1
        for d in shape:
            n_elements *= d
        total_bytes = 3 * n_elements * self.dtype_bytes
        memory_us = total_bytes / self._mem_bw_per_us
        return memory_us * self.cal.elementwise_factor

    def _pool_us(self, node: OpNode) -> float:
        """Pooling: memory-bound, read input + write output."""
        in_elements = 1
        for d in node.input_shape:
            in_elements *= d
        out_elements = 1
        for d in node.output_shape:
            out_elements *= d
        total_bytes = (in_elements + out_elements) * self.dtype_bytes
        memory_us = total_bytes / self._mem_bw_per_us
        return memory_us * self.cal.pool_factor

    def _linear_us(self, node: OpNode) -> float:
        p = node.params
        M = node.input_shape[0]   # batch
        K = p['in_features']
        N = p['out_features']

        flops = 2 * M * K * N
        compute_us = flops / self._peak_flops_per_us

        input_bytes = M * K * self.dtype_bytes
        weight_bytes = K * N * self.dtype_bytes
        output_bytes = M * N * self.dtype_bytes
        total_bytes = input_bytes + weight_bytes + output_bytes
        memory_us = total_bytes / self._mem_bw_per_us

        return max(compute_us, memory_us) * self.cal.linear_factor

    def _matmul_us(self, node: OpNode) -> float:
        """MatMul (attention): estimate from input/output shapes."""
        in_shape = node.input_shape
        out_shape = node.output_shape
        # Approximate FLOPs from output volume × reduction dim
        out_elements = 1
        for d in out_shape:
            out_elements *= d
        in_elements = 1
        for d in in_shape:
            in_elements *= d
        # Heuristic: FLOPs ~ 2 * output_elements * inner_dim
        # inner_dim approximated as in_elements / out_elements * last_dim
        flops = 2 * max(in_elements, out_elements)
        compute_us = flops / self._peak_flops_per_us

        total_bytes = (in_elements + out_elements) * self.dtype_bytes * 2
        memory_us = total_bytes / self._mem_bw_per_us

        return max(compute_us, memory_us) * self.cal.matmul_factor
