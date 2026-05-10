"""Kernel Code Generator for FusionSlice patterns.

Looks up the epilogue pattern in the template library
(sengine/kernels/interleaved_templates.py) and compiles
the matching kernel with concrete shape parameters.

Usage:
    from sengine.kernel_codegen import compile_fused_kernel
    kern, n_args, names = compile_fused_kernel(ir, slice, T=4, B=4)
"""

from __future__ import annotations
from sengine.ir import EngineIR, OpType
from sengine.kernels.interleaved_templates import KERNEL_TEMPLATES, maxpool_lif


def compile_fused_kernel(ir, anchor_nid, absorbed_nids, T_steps, B,
                         block_M=64, block_N=64, block_K=32,
                         num_stages=2, threads=128,
                         v_threshold=1.0, v_reset=0.0):
    """Compile a fused kernel by looking up the pattern in KERNEL_TEMPLATES.

    Returns (kernel_callable, n_caller_args, arg_names) or (None, 0, []).
    """
    anchor = ir.nodes[anchor_nid]

    # Determine anchor shape: Conv2d, MatMul, or Linear
    if anchor.op_type == OpType.Conv2d:
        cp = anchor.conv_params
        if not cp or cp.kernel_h != 1 or cp.kernel_w != 1:
            return None, 0, []
        if not anchor.output_shapes or len(anchor.output_shapes[0]) != 4:
            return None, 0, []
        _, C_out, OH, OW = anchor.output_shapes[0]
        C_in, F = cp.in_channels, C_out
        H, W = OH * cp.stride_h, OW * cp.stride_w

    elif anchor.op_type in (OpType.MatMul, OpType.Linear):
        if not anchor.output_shapes:
            return None, 0, []
        out_shape = anchor.output_shapes[0]
        if len(out_shape) == 2:
            # (TB*spatial, F) — treat as (TB, 1, 1, F) for Conv1x1 template
            M_total, F = out_shape
            C_in = anchor.input_shapes[0][-1] if anchor.input_shapes else F
            H, W = 1, M_total // (T_steps * B) if (T_steps * B) > 0 else 1
        elif len(out_shape) == 4:
            _, C_out, OH, OW = out_shape
            C_in = anchor.input_shapes[0][-1] if anchor.input_shapes else C_out
            F = C_out; H, W = OH, OW
        else:
            return None, 0, []
    else:
        return None, 0, []

    # Build pattern string from absorbed ops
    ops = []
    for nid in absorbed_nids:
        n = ir.nodes.get(nid)
        if n and n.op_type not in (OpType.Reshape, OpType.Transpose,
                                    OpType.Identity, OpType.Flatten):
            ops.append(n.op_type.name)
    pattern = 'Conv2d+' + '+'.join(ops) if ops else 'Conv2d'

    # Look up template
    template_fn = KERNEL_TEMPLATES.get(pattern)
    if template_fn is None:
        return None, 0, []

    # Count args based on template
    n_neurons = sum(1 for o in ops if o in ('IF', 'LIF', 'MS'))
    has_add = 'Add' in ops

    tile = dict(block_M=block_M, block_N=block_N, block_K=block_K,
                num_stages=num_stages, threads=threads)

    if n_neurons <= 1 and not has_add:
        kern = template_fn(B=B, C_in=C_in, H=H, W=W, F=F, T_steps=T_steps,
                           **tile, v_threshold=v_threshold, v_reset=v_reset)
        names = ['data', 'weight', 'membrane', 'bn_scale', 'bn_bias', 'output']
    elif n_neurons <= 1 and has_add:
        kern = template_fn(B=B, C_in=C_in, H=H, W=W, F=F, T_steps=T_steps,
                           **tile, v_threshold=v_threshold, v_reset=v_reset)
        names = ['data', 'weight', 'membrane', 'bn_scale', 'bn_bias', 'residual', 'output']
    elif n_neurons == 2:
        kern = template_fn(B=B, C_in=C_in, H=H, W=W, F=F, T_steps=T_steps,
                           **tile, v_threshold=v_threshold, v_reset=v_reset)
        names = ['data', 'weight', 'membrane1', 'membrane2', 'bn_scale', 'bn_bias', 'residual', 'output']
    else:
        return None, 0, []

    return kern, len(names) - 1, names  # -1: output handled by out_idx
