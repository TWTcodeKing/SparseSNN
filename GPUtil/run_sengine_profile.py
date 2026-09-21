#!/usr/bin/env python3
"""Minimal sengine runner for ncu profiling.

Loads a pre-built .sengine file, sets up C++ executor via plan-driven path
(same as SEngine.build()), but SKIPS capture_graph().

Uses cudaProfilerStart/Stop so ncu --profile-from-start off only captures
inference kernels.

Usage (standalone test):
    python GPUtil/run_sengine_profile.py --model maxformer_10_512 --batch 16 --warmup 10 --iters 1

Under ncu:
    sudo CUDA_VISIBLE_DEVICES=3 /opt/.../ncu --profile-from-start off \
        --target-processes all --metrics ... \
        .venv/bin/python GPUtil/run_sengine_profile.py --model maxformer_10_512 --batch 16
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

for cuda_path in ['/usr/local/cuda-12.8', '/usr/local/cuda', '/usr/local/cuda-12.6']:
    if os.path.isdir(cuda_path):
        os.environ.setdefault('CUDA_HOME', cuda_path)
        os.environ['PATH'] = os.path.join(cuda_path, 'bin') + ':' + os.environ.get('PATH', '')
        break

import torch

from GPUtil.config import MODELS, T, PRECISION, sengine_path


def load_sengine_no_graph(sengine_file):
    """Load .sengine and set up C++ executor via plan-driven path, NO capture_graph().

    Replicates SEngine.build() steps 1-4 (engine.py:111-184) but:
    - Loads IR/schedule/weights from .sengine instead of parsing ONNX
    - Uses the saved exec_plan (ir._exec_plan) instead of recomputing
    - Skips capture_graph() so sengine_replay() → sengine_execute()
    """
    from sengine.engine import SEngine, _detect_arch, _detect_nvcc
    from sengine.ir import OpType
    from sengine.build.sengine_io import load_sengine
    from sengine.build.tilelang_compiler import TileLangCompiler
    from sengine.build.export_standalone import export_all_kernels
    from sengine.cuda_graph_runtime import CUDAGraphEngine
    from sengine.logger import logger

    eng = SEngine()
    t0 = time.time()
    arch = _detect_arch()
    nvcc = _detect_nvcc()

    # 1. Load IR + schedule from .sengine
    ir, schedule, T_loaded, batch_size = load_sengine(sengine_file)
    eng._ir = ir
    eng._schedule = schedule
    eng.T = T_loaded
    eng.batch_size = batch_size

    # 2. Recompile TileLang kernels (cached tile configs, no autotuning)
    compiler = TileLangCompiler(ir, T=T_loaded, batch_size=batch_size, autotune=False)
    eng._kernels = compiler.compile_all()

    # 3. Build Python engine (buffer allocation)
    eng._py_engine = CUDAGraphEngine()
    eng._py_engine.build(ir, eng._kernels, schedule, T=T_loaded, batch_size=batch_size)

    # 4. Export standalone .so
    build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             '.cache', f'sengine_B{batch_size}')
    eng._kernel_so_map = export_all_kernels(
        eng._kernels, ir, build_dir, nvcc=nvcc, arch=arch)

    # 5. Use exec_plan from .sengine if available, otherwise recompute
    if hasattr(ir, '_exec_plan') and ir._exec_plan is not None:
        eng._exec_plan = ir._exec_plan
    else:
        from sengine.build.buffer_planner import plan_buffers
        eng._exec_plan = plan_buffers(ir, schedule, T_loaded, batch_size,
                                       kernel_so_map=eng._kernel_so_map,
                                       precision=PRECISION)

    # 5b. Inject buffer aliases for absorbed (ZeroCost) nodes
    for nid, node in ir.nodes.items():
        absorbed = node.extra_attrs.get("absorbed_nids", [])
        if not absorbed:
            continue
        anchor_buf = eng._py_engine.activations.get(nid)
        if anchor_buf is None:
            continue
        for ab_nid in absorbed:
            if ab_nid not in eng._py_engine.activations:
                eng._py_engine.activations[ab_nid] = anchor_buf

    # 6. Wire C++ executor via plan-driven path (same as SEngine.build)
    # NOTE: we do NOT fall back to Python runtime for uncompiled attention.
    # Uncompiled attention nodes are auto-skipped by _setup_attn_node().
    # This ensures TileLang kernels are dispatched via C++ executor for profiling.
    if eng._py_engine._lazy_mode:
        logger.phase("LOAD", "Python runtime: lazy mode (OOM)")
        eng._use_python_runtime = True
        eng._graph_captured = True
        return eng

    from sengine.runtime.plan_executor import setup_executor_from_plan

    def _attn_handler(exe, nid, nid_tl_idx, buf_ptrs, plan):
        eng._setup_attn_node(exe, nid, nid_tl_idx, buf_ptrs, plan)

    eng._cpp_exec = setup_executor_from_plan(
        eng._exec_plan, eng._py_engine, eng._kernel_so_map, eng._ir,
        attn_setup_fn=_attn_handler)

    # *** KEY: DO NOT call eng._cpp_exec.capture_graph() ***
    eng._cpp_exec.sync()
    torch.cuda.synchronize()
    eng._use_python_runtime = False
    eng._graph_captured = True

    elapsed = time.time() - t0
    logger.phase("LOAD", "Loaded for ncu profiling in %.1fs (%d ops, plan-driven, no graph)", elapsed, len(schedule))
    return eng


def main():
    parser = argparse.ArgumentParser(description="sengine runner for ncu profiling")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=1)
    args = parser.parse_args()

    se_path = sengine_path(args.model, args.batch)
    if not os.path.exists(se_path):
        print(f"ERROR: .sengine file not found: {se_path}")
        print(f"Build first: python GPUtil/build_one.py --model {args.model} "
              f"--batch {args.batch} --backend sengine --gpu-id <N>")
        sys.exit(1)

    print(f"=== sengine ncu profile runner ===")
    print(f"Model: {args.model}, B={args.batch}")
    print(f"Loading: {se_path}")

    eng = load_sengine_no_graph(se_path)

    if eng._use_python_runtime:
        replay = eng._py_engine._execute_schedule
        n_ops = len(eng._schedule)
        print(f"Warmup: {args.warmup} iters ({n_ops} ops, Python runtime)...")
        for _ in range(args.warmup):
            for mem in eng._py_engine.membranes.values():
                mem.zero_()
            replay()
        torch.cuda.synchronize()

        print(f"Profiling: {args.iters} iters...")
        t0 = time.time()
        for _ in range(args.iters):
            for mem in eng._py_engine.membranes.values():
                mem.zero_()
            torch.cuda.cudart().cudaProfilerStart()
            replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

    else:
        cpp = eng._cpp_exec
        n_ops = len(eng._schedule)
        print(f"Warmup: {args.warmup} iters ({n_ops} ops, C++ executor)...")
        for _ in range(args.warmup):
            cpp.reset_membranes()
            cpp.replay()
        cpp.sync()

        print(f"Profiling: {args.iters} iters...")
        t0 = time.time()
        for _ in range(args.iters):
            cpp.reset_membranes()
            torch.cuda.cudart().cudaProfilerStart()
            cpp.replay()
            cpp.sync()
            torch.cuda.cudart().cudaProfilerStop()


    ms = (time.time() - t0) / args.iters * 1000
    print(f"Done: {ms:.3f} ms/iter, {n_ops} ops profiled")


if __name__ == '__main__':
    main()
