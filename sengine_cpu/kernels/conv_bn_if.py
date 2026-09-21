"""Fused Conv1x1+BN+IF/LIF kernel for CPU (TVM TE).

Single-timestep kernel with configurable scheduling:
- GEMM: tiled i (parallel) + vectorized j (AVX2/512) + unrolled k
- Epilogue (BN+IF): separate loops (TVM TE limitation — can't inline output buffers)
- The C runtime T-loop keeps membrane in L1 across timestep calls.

Tile config interface for custom autotuning:
  tile_config = {
      "tile_M": 64,         # spatial tile for parallel chunking
      "tile_N": 8,          # channel tile = SIMD width (8=AVX2, 16=AVX-512)
      "tile_K": 4,          # reduction unroll factor
      "parallel_M": True,   # OpenMP on outer M tiles
      "vectorize_N": True,  # SIMD vectorize inner N
  }
"""

from sengine_cpu.tvm_env import inject_tvm_site_packages

inject_tvm_site_packages()


def default_tile_config(M, C_in, F):
    """Heuristic tile config based on problem dimensions."""
    simd_w = 8  # AVX2
    tile_N = min(simd_w, F) if F >= simd_w else F

    if M <= 64:
        # Small spatial: no parallel, just vectorize
        return {"tile_M": 0, "tile_N": tile_N, "tile_K": min(4, C_in),
                "parallel_M": False, "vectorize_N": F >= simd_w}
    else:
        # Large: parallel tiles + vectorize + unroll
        tile_M = 64 if M >= 256 else min(32, M)
        return {"tile_M": tile_M, "tile_N": tile_N, "tile_K": min(4, C_in),
                "parallel_M": True, "vectorize_N": F >= simd_w}


def _apply_schedule(mod, tile_config):
    """Apply schedule transforms to the GEMM block."""
    import tvm
    import tvm.s_tir as s_tir

    cfg = tile_config
    sch = s_tir.Schedule(mod)

    try:
        gemm = sch.get_sblock("gemm")
        loops = sch.get_loops(gemm)
        if len(loops) < 3:
            return mod

        i_loop, j_loop, k_loop = loops[0], loops[1], loops[2]

        tile_M = cfg.get("tile_M", 0)
        tile_N = cfg.get("tile_N", 8)
        tile_K = cfg.get("tile_K", 4)

        # Split j for vectorization (innermost)
        if cfg.get("vectorize_N") and tile_N > 1:
            j_o, j_i = sch.split(j_loop, factors=[None, tile_N])
        else:
            j_o, j_i = j_loop, None

        # Split k for unroll
        if tile_K > 1 and tile_K < loops[2].extent.value if hasattr(loops[2], 'extent') else True:
            k_o, k_i = sch.split(k_loop, factors=[None, tile_K])
        else:
            k_o, k_i = k_loop, None

        # Split i for parallel tiling
        if tile_M > 1 and cfg.get("parallel_M"):
            i_o, i_i = sch.split(i_loop, factors=[None, tile_M])
        else:
            i_o, i_i = i_loop, None

        # Reorder: i_o, j_o, k_o, i_i, k_i, j_i
        order = []
        if i_i is not None: order.append(i_o)
        if j_i is not None: order.append(j_o)
        if k_i is not None: order.append(k_o)
        if i_i is not None: order.append(i_i)
        if k_i is not None: order.append(k_i)
        if j_i is not None: order.append(j_i)

        if len(order) >= 2:
            sch.reorder(*order)

        # Apply transforms
        if i_i is not None and cfg.get("parallel_M"):
            sch.parallel(i_o)
        if k_i is not None:
            sch.unroll(k_i)
        if j_i is not None and cfg.get("vectorize_N"):
            sch.vectorize(j_i)

        return sch.mod
    except Exception:
        return mod


def make_conv1x1_bn_if(M: int, C_in: int, F: int,
                        v_threshold: float = 1.0,
                        v_reset: float = 0.0,
                        tile_config: dict | None = None,
                        target_str: dict | str | None = None):
    """Build fused Conv1x1+BN+IF kernel."""
    import tvm
    from tvm import te

    target = tvm.target.Target(
        target_str or {"kind": "llvm", "mattr": ["+avx2", "+fma"]})
    cfg = tile_config or default_tile_config(M, C_in, F)

    data = te.placeholder((M, C_in), name="data", dtype="float32")
    weight = te.placeholder((C_in, F), name="weight", dtype="float32")
    scale = te.placeholder((F,), name="scale", dtype="float32")
    bias = te.placeholder((F,), name="bias", dtype="float32")
    mem = te.placeholder((M, F), name="mem", dtype="float32")

    k = te.reduce_axis((0, C_in), name="k")
    gemm = te.compute(
        (M, F), lambda i, j: te.sum(data[i, k] * weight[k, j], axis=k),
        name="gemm")
    h = te.compute(
        (M, F), lambda i, j: mem[i, j] + gemm[i, j] * scale[j] + bias[j],
        name="h")
    spike = te.compute(
        (M, F), lambda i, j: te.if_then_else(
            h[i, j] >= float(v_threshold), 1.0, 0.0),
        name="spike")
    new_mem = te.compute(
        (M, F), lambda i, j: (1.0 - spike[i, j]) * h[i, j] + spike[i, j] * float(v_reset),
        name="new_mem")

    func = te.create_prim_func([data, weight, scale, bias, mem, spike, new_mem])
    mod = tvm.IRModule({"main": func})
    mod = _apply_schedule(mod, cfg)
    return tvm.build(mod, target=target)


def make_conv1x1_bn_lif(M: int, C_in: int, F: int,
                         v_threshold: float = 1.0,
                         v_reset: float = 0.0,
                         tau: float = 2.0,
                         tile_config: dict | None = None,
                         target_str: dict | str | None = None):
    """Build fused Conv1x1+BN+LIF kernel."""
    import tvm
    from tvm import te

    target = tvm.target.Target(
        target_str or {"kind": "llvm", "mattr": ["+avx2", "+fma"]})
    cfg = tile_config or default_tile_config(M, C_in, F)
    decay = 1.0 - 1.0 / tau
    recip_tau = 1.0 / tau

    data = te.placeholder((M, C_in), name="data", dtype="float32")
    weight = te.placeholder((C_in, F), name="weight", dtype="float32")
    scale = te.placeholder((F,), name="scale", dtype="float32")
    bias = te.placeholder((F,), name="bias", dtype="float32")
    mem = te.placeholder((M, F), name="mem", dtype="float32")

    k = te.reduce_axis((0, C_in), name="k")
    gemm = te.compute(
        (M, F), lambda i, j: te.sum(data[i, k] * weight[k, j], axis=k),
        name="gemm")
    h = te.compute(
        (M, F), lambda i, j: float(decay) * mem[i, j] + float(recip_tau) * (
            gemm[i, j] * scale[j] + bias[j]),
        name="h")
    spike = te.compute(
        (M, F), lambda i, j: te.if_then_else(
            h[i, j] >= float(v_threshold), 1.0, 0.0),
        name="spike")
    new_mem = te.compute(
        (M, F), lambda i, j: (1.0 - spike[i, j]) * h[i, j] + spike[i, j] * float(v_reset),
        name="new_mem")

    func = te.create_prim_func([data, weight, scale, bias, mem, spike, new_mem])
    mod = tvm.IRModule({"main": func})
    mod = _apply_schedule(mod, cfg)
    return tvm.build(mod, target=target)


def export_kernel(lib, output_path: str):
    """Export compiled kernel as standalone .so."""
    lib.export_library(output_path)
    return output_path
