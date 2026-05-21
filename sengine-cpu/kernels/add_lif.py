"""Fused Add+LIF kernel for CPU (TVM TE).

Pattern: spikes = LIF(membrane + input_a + input_b)
Single-timestep; C runtime loops over T.

Args signature (5 buffers):
  input_a(M, F), input_b(M, F), mem(M, F)
  -> spike(M, F), new_mem(M, F)
"""

import os, sys

_TVM_VENV = "/home/twt/tvm_build/tvm_venv"
if os.path.isdir(_TVM_VENV):
    _TVM_SITE = os.path.join(_TVM_VENV, "lib/python3.12/site-packages")
    if _TVM_SITE not in sys.path:
        sys.path.insert(0, _TVM_SITE)


def make_add_lif(M: int, F: int,
                  v_threshold: float = 1.0,
                  v_reset: float = 0.0,
                  tau: float = 2.0,
                  target_str: dict | str | None = None):
    """Build fused Add+LIF kernel."""
    import tvm
    from tvm import te

    target = tvm.target.Target(target_str or {"kind": "llvm"})
    decay = 1.0 - 1.0 / tau
    recip_tau = 1.0 / tau

    input_a = te.placeholder((M, F), name="input_a", dtype="float32")
    input_b = te.placeholder((M, F), name="input_b", dtype="float32")
    mem = te.placeholder((M, F), name="mem", dtype="float32")

    # LIF: h = decay * mem + recip_tau * (a + b)
    h = te.compute(
        (M, F),
        lambda i, j: float(decay) * mem[i, j] + float(recip_tau) * (
            input_a[i, j] + input_b[i, j]),
        name="h"
    )

    spike = te.compute(
        (M, F),
        lambda i, j: te.if_then_else(
            h[i, j] >= float(v_threshold), 1.0, 0.0),
        name="spike"
    )

    new_mem = te.compute(
        (M, F),
        lambda i, j: (1.0 - spike[i, j]) * h[i, j] + spike[i, j] * float(v_reset),
        name="new_mem"
    )

    func = te.create_prim_func([input_a, input_b, mem, spike, new_mem])
    mod = tvm.IRModule({"main": func})
    return tvm.build(mod, target=target)


def make_add_if(M: int, F: int,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 target_str: dict | str | None = None):
    """Build fused Add+IF kernel (no decay)."""
    import tvm
    from tvm import te

    target = tvm.target.Target(target_str or {"kind": "llvm"})

    input_a = te.placeholder((M, F), name="input_a", dtype="float32")
    input_b = te.placeholder((M, F), name="input_b", dtype="float32")
    mem = te.placeholder((M, F), name="mem", dtype="float32")

    h = te.compute(
        (M, F),
        lambda i, j: mem[i, j] + input_a[i, j] + input_b[i, j],
        name="h"
    )

    spike = te.compute(
        (M, F),
        lambda i, j: te.if_then_else(
            h[i, j] >= float(v_threshold), 1.0, 0.0),
        name="spike"
    )

    new_mem = te.compute(
        (M, F),
        lambda i, j: (1.0 - spike[i, j]) * h[i, j] + spike[i, j] * float(v_reset),
        name="new_mem"
    )

    func = te.create_prim_func([input_a, input_b, mem, spike, new_mem])
    mod = tvm.IRModule({"main": func})
    return tvm.build(mod, target=target)


def export_kernel(lib, output_path: str):
    """Export compiled kernel as standalone .so."""
    lib.export_library(output_path)
    return output_path
