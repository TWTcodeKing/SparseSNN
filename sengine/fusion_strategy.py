"""Pluggable fusion strategies for the sengine build pipeline.

Decouples fusion decisions from IR transformation. The optimizer handles
structural transforms (BN fold, DCE, shape propagation, bound classification).
The fusion strategy then decides which ops to fuse into combined kernels.

Usage:
    from sengine.fusion_strategy import FusionStrategy, apply_fusion

    # In the build pipeline:
    optimize_ir(ir, tilelang=True)           # structural transforms only
    apply_fusion(ir, strategy='slicer')       # fusion decisions
    kernels = compiler.compile_all()          # compile based on assignments
"""

from __future__ import annotations
from enum import Enum

from sengine.ir import EngineIR, OpType, KernelVariant, BoundType
from sengine.logger import logger


class FusionStrategy(Enum):
    NONE = "none"           # All decomposed — baseline for ablation
    SLICER = "slicer"       # Graph slicer: greedy compute-anchored fusion


def apply_fusion(ir: EngineIR, strategy: str = "slicer",
                 batch_size: int = 1,
                 fusion_rec: str = None) -> dict:
    """Apply a fusion strategy to the IR.

    Args:
        ir: Optimized EngineIR.
        strategy: 'none' or 'slicer'.
        batch_size: Batch size.
        fusion_rec: Path to fusion recommendation JSON from validator pre-pass.
                    If provided, shapes marked 'revert' are kept decomposed.
    """
    if strategy == FusionStrategy.NONE.value or strategy == "none":
        return _apply_none(ir)
    elif strategy == FusionStrategy.SLICER.value or strategy == "slicer":
        return _apply_slicer(ir, batch_size, fusion_rec=fusion_rec)
    else:
        logger.warning("Unknown fusion strategy '%s', using 'none'", strategy)
        return _apply_none(ir)


def _apply_none(ir: EngineIR) -> dict:
    """No fusion — all ops stay decomposed. Baseline for ablation."""
    logger.phase("FUSION_STRATEGY", "none (all decomposed)")
    return {"strategy": "none", "n_fused": 0, "n_slices": 0}


def _apply_slicer(ir: EngineIR, batch_size: int,
                   fusion_rec: str = None) -> dict:
    """Graph slicer fusion: greedy compute-anchored absorption.

    If fusion_rec is provided, reads the validator's per-shape recommendations
    and only fuses shapes marked 'keep'. Shapes marked 'revert' stay decomposed.
    Without fusion_rec, fuses all eligible patterns unconditionally.
    """
    from sengine.graph_slicer import slice_graph
    from sengine.kernels.interleaved_templates import KERNEL_TEMPLATES

    # Load validator recommendations (if available)
    rec = {}
    if fusion_rec:
        from sengine.build.fusion_validator import load_recommendations
        rec = load_recommendations(fusion_rec)
        if rec:
            n_keep = sum(1 for r in rec.values() if r.get('decision') == 'keep')
            n_rev = sum(1 for r in rec.values() if r.get('decision') == 'revert')
            logger.phase("FUSION_STRATEGY", "Loaded recommendations: %d keep, %d revert",
                         n_keep, n_rev)

    slices = slice_graph(ir)
    T = ir.T if ir.T > 0 else 4

    n_fused = 0
    n_template_hit = 0
    n_decomposed = 0
    n_rec_reverted = 0

    for sl in slices:
        if not sl.absorbed_nids:
            n_decomposed += 1
            continue

        if sl.pattern not in KERNEL_TEMPLATES:
            n_decomposed += 1
            continue

        anchor = ir.nodes.get(sl.anchor_nid)

        cp = anchor.conv_params if anchor else None
        if anchor.op_type == OpType.Conv2d and cp:
            K_red = cp.kernel_h * cp.kernel_w * cp.in_channels

            # Conv3x3: K_red must be aligned for tensor cores
            if cp.kernel_h != 1 and K_red % 8 != 0:
                n_decomposed += 1
                continue

            # Stem conv (C_in < 4): use cuDNN, not fusible
            if cp.in_channels < 4:
                n_decomposed += 1
                continue

            # Check validator recommendation for this shape
            if not anchor.output_shapes:
                n_decomposed += 1
                continue
            _, _, OH, OW = anchor.output_shapes[0]
            rec_key = f"{cp.in_channels}_{cp.out_channels}_K{cp.kernel_h}_S{cp.stride_h}_{OH}x{OW}"
            if rec_key in rec and rec[rec_key].get('decision') == 'revert':
                n_decomposed += 1
                n_rec_reverted += 1
                continue

            # Fuse this node
            if cp.kernel_h == 1 and cp.kernel_w == 1:
                anchor.assigned_kernel = KernelVariant.TileLangFusedConv1x1BNIF
            else:
                anchor.assigned_kernel = KernelVariant.TileLangFusedConvBNIF
            anchor.bound_type = BoundType.COMPUTE

            for ab_nid in sl.absorbed_nids:
                ab = ir.nodes.get(ab_nid)
                if ab:
                    ab.assigned_kernel = KernelVariant.ZeroCost
                    ab.bound_type = BoundType.ZERO

            # Store fusion metadata on the anchor
            anchor.extra_attrs["fusion_slice"] = sl.pattern
            anchor.extra_attrs["absorbed_nids"] = sl.absorbed_nids
            if sl.residual_nid is not None:
                anchor.extra_attrs["residual_nid"] = sl.residual_nid

            n_fused += 1
            n_template_hit += 1

        elif anchor.op_type in (OpType.MatMul, OpType.Linear):
            # MatMul/Linear fusion: same interleaved approach as Conv1x1
            # MatMul is (TB, M, K) @ (K, N) — same GEMM structure
            if not anchor.output_shapes:
                n_decomposed += 1
                continue

            # Check validator recommendation
            out_shape = anchor.output_shapes[0]
            K_in = anchor.input_shapes[0][-1] if anchor.input_shapes else 0
            N_out = out_shape[-1] if out_shape else 0
            rec_key = f"{K_in}_{N_out}_MatMul"
            if rec_key in rec and rec[rec_key].get('decision') == 'revert':
                n_decomposed += 1
                n_rec_reverted += 1
                continue

            anchor.assigned_kernel = KernelVariant.TileLangFusedMatMulLIF
            anchor.bound_type = BoundType.COMPUTE

            for ab_nid in sl.absorbed_nids:
                ab = ir.nodes.get(ab_nid)
                if ab:
                    ab.assigned_kernel = KernelVariant.ZeroCost
                    ab.bound_type = BoundType.ZERO

            anchor.extra_attrs["fusion_slice"] = sl.pattern
            anchor.extra_attrs["absorbed_nids"] = sl.absorbed_nids
            if sl.residual_nid is not None:
                anchor.extra_attrs["residual_nid"] = sl.residual_nid

            n_fused += 1
            n_template_hit += 1

        else:
            n_decomposed += 1

    from collections import Counter
    patterns = Counter(s.pattern for s in slices)

    logger.phase("FUSION_STRATEGY", "slicer: %d fused, %d decomposed, %d slices "
                 "(%d template hits)", n_fused, n_decomposed, len(slices), n_template_hit)

    return {
        "strategy": "slicer",
        "n_fused": n_fused,
        "n_decomposed": n_decomposed,
        "n_slices": len(slices),
        "n_template_hit": n_template_hit,
        "patterns": dict(patterns),
    }
