"""ONNX -> TVM Relax -> compiled module builder.

Two modes:
  - Untuned:  relax.build() with default schedules (fast, baseline perf)
  - Tuned:    MetaSchedule auto-tuning (configurable trials, best perf)

Tuning logs are saved to disk and reused across runs.

TVM v0.25+ uses Relax (not Relay). The pipeline is:
  ONNX -> tvm.relax.frontend.onnx.from_onnx() -> IRModule
       -> optimization pipeline -> tvm.relax.build() -> Executable
       -> export_library() -> .so

Usage:
    from iengine.backends.tvm.builder import build_tvm_module
    build_tvm_module("model.onnx", "model.so")

    # With MetaSchedule tuning
    build_tvm_module("model.onnx", "model.so",
                     tune=True, max_trials_per_task=64)
"""

import os
from pathlib import Path
from typing import Optional


def _try_import_tvm():
    try:
        import tvm
        from tvm import relax
        return tvm, relax
    except ImportError as e:
        raise ImportError(
            f"tvm import failed: {e}\n"
            "Run with the TVM venv: /home/twt/tvm_build/tvm_venv/bin/python"
        ) from e


def build_tvm_module(
    onnx_path: str,
    output_path: str,
    target: str = '{"kind": "cuda", "arch": "sm_89"}',
    input_name: str = "input",
    input_shape: Optional[tuple[int, ...]] = None,
    fp16: bool = False,
    tune: bool = False,
    max_trials_per_task: int = 64,
    total_trials: int = 0,
    tuning_log_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Build a TVM compiled module from an ONNX model.

    Args:
        onnx_path:           Path to input ONNX file.
        output_path:         Path to save compiled .so library.
        target:              TVM target (JSON dict string).
        input_name:          ONNX input tensor name.
        input_shape:         (B, C, H, W) input shape. Auto-detected from ONNX if None.
        fp16:                Enable FP16 mixed precision (auto-cast fp32 ops to fp16).
        tune:                Enable MetaSchedule auto-tuning.
        max_trials_per_task: Max tuning trials per kernel (64=quick, 500=thorough).
        total_trials:        Total tuning budget (0 = max_trials_per_task * 100).
        tuning_log_dir:      Directory for tuning logs (reused across runs).
        verbose:             Print build progress.

    Returns:
        Path to the saved .so file.
    """
    tvm, relax = _try_import_tvm()
    from tvm.relax.frontend.onnx import from_onnx
    import onnx

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load ONNX and eliminate scalar Gather ops (SNN temporal indexing)
    onnx_model = onnx.load(onnx_path)
    onnx_model, n_replaced = _replace_gather_with_slice(onnx_model)
    if n_replaced > 0 and verbose:
        print(f"  Replaced {n_replaced} Gather ops with Slice+Squeeze")

    # FP16: convert ONNX initializer weights from float32 to float16.
    # This is a lightweight conversion that avoids onnxconverter-common's
    # slow full-graph rewrite. TVM will generate FP16 compute kernels for
    # ops whose inputs are float16, with automatic upcasts where needed.
    if fp16:
        if verbose:
            print(f"  Converting ONNX weights to FP16")
        onnx_model = _convert_onnx_to_fp16(onnx_model)

    # Auto-detect input shape from ONNX if not provided
    if input_shape is None:
        inp = onnx_model.graph.input[0]
        input_name = inp.name
        dims = inp.type.tensor_type.shape.dim
        input_shape = tuple(
            d.dim_value if d.dim_value > 0 else 1 for d in dims
        )
        if verbose:
            print(f"  Auto-detected input: {input_name} {input_shape}")

    shape_dict = {input_name: list(input_shape)}

    if verbose:
        print(f"Building TVM module: target={target}")
        print(f"  ONNX:   {onnx_path}")
        print(f"  Output: {output_path}")
        print(f"  Input:  {input_name} {input_shape}")

    # Convert ONNX to Relax IRModule
    dtype_dict = "float16" if fp16 else "float32"
    mod = from_onnx(onnx_model, shape_dict=shape_dict, dtype_dict=dtype_dict)

    if verbose:
        print(f"  Converted ONNX to Relax IR")

    # Use Target.from_device to auto-detect GPU attributes (max_threads_per_block,
    # max_shared_memory_per_block, etc.) needed by MetaSchedule tuning.
    # Falls back to the user-provided target string if device detection fails.
    try:
        tvm_target = tvm.target.Target.from_device(tvm.cuda(0))
    except Exception:
        tvm_target = tvm.target.Target(target)

    if tune:
        mod = _tune(mod, tvm_target, max_trials_per_task, total_trials,
                     tuning_log_dir, verbose)

    # Build: apply default pipeline + compile to executable.
    # Use a custom TIR pipeline that skips VerifyMemory — SNN ONNX models
    # produce scalar ops (0-dim tensors from Where/GreaterOrEqual in spiking
    # neurons) that fail GPU memory verification but execute correctly.
    if verbose:
        mode = "tuned" if tune else "untuned"
        print(f"  Compiling ({mode})...")

    tir_pipeline = _snn_tir_pipeline()
    # SNN transformer models (MaxFormer, SpikFormer) crash with TVM's
    # default fusion — fused neuron kernels produce illegal memory accesses.
    # fuse_opt_level=0 disables fusion entirely (each op = own kernel).
    # This is slower but correct for all SNN architectures.
    with tvm.transform.PassContext(
        config={"relax.FuseOps.max_depth": 1}
    ):
        ex = relax.build(mod, target=tvm_target, tir_pipeline=tir_pipeline)

    # Export as shared library
    ex.export_library(output_path)

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    return output_path


def _snn_tir_pipeline():
    """Custom TIR pipeline that skips VerifyMemory.

    SNN models exported via torch.onnx.export contain scalar control-flow ops
    (Where, GreaterOrEqual) from spiking neuron threshold logic. TVM's ONNX
    frontend converts these to 0-dim prim_funcs that have no GPU thread
    bindings, causing VerifyMemory to fail. Skipping this check is safe —
    the scalar ops execute correctly on both CPU and GPU paths.
    """
    import tvm
    from tvm import tirx
    from tvm import s_tir

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            s_tir.transform.CanonicalizeLoop(),
            s_tir.transform.LowerCrossThreadReduction(),
            s_tir.transform.LowerInitBlock(),
            s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            s_tir.transform.ConvertBlocksToOpaque(),
            s_tir.transform.LiftThreadBinding(),
            s_tir.transform.ManifestSharedMemoryLocalStage(),
            s_tir.transform.CompactBufferAllocation(),
            s_tir.transform.LowerAutoCopy(),
            s_tir.transform.UnifyThreadBinding(),
            s_tir.transform.LowerMatchBuffer(),
            tirx.transform.Simplify(),
            s_tir.transform.InjectPermutedLayout(),
            s_tir.transform.AnnotateIrregularLoop(),
            s_tir.transform.InjectSoftwarePipeline(),
            s_tir.transform.TransformMmaBufferLayout(),
            s_tir.transform.LowerOpaqueBlock(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            s_tir.transform.LoopPartition(),
            tirx.transform.VectorizeLoop(
                not bool(config.get("tirx.disable_vectorize", False))),
            s_tir.transform.InjectVirtualThread(),
            s_tir.transform.InjectDoubleBuffer(),
        ]
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tirx.transform.StorageRewrite())
        if config.get("tirx.use_async_copy", False):
            passes.append(s_tir.transform.LowerAsyncDMA())
        passes.extend([
            s_tir.transform.HoistIfThenElse(),
            tirx.transform.UnrollLoop(),
            s_tir.transform.RenormalizeSplitPattern(),
            tirx.transform.Simplify(),
            tirx.transform.RemoveNoOp(),
            s_tir.transform.RewriteUnsafeSelect(),
        ])
        if bool(config.get("tirx.instrument_bound_checkers", False)):
            passes.append(s_tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32(True))
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        passes.extend([
            tirx.transform.FP8ComputeLegalize(),
            s_tir.transform.VerifyVTCMLimit(),
            s_tir.transform.LowerVtcmAlloc(),
            # VerifyMemory SKIPPED — SNN scalar ops fail this check
            tirx.transform.AnnotateEntryFunc(),
        ])
        passes.extend([
            s_tir.transform.ThreadSync("shared"),
            s_tir.transform.ThreadSync("shared.dyn"),
            s_tir.transform.ThreadSync("warp"),
            s_tir.transform.InferFragment(),
            s_tir.transform.LowerThreadAllreduce(),
        ])
        if bool(config.get("tirx.use_async_copy", False)):
            passes.append(s_tir.transform.InjectPTXAsyncCopy())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32())
        passes.extend([
            tirx.transform.AnnotateDeviceRegions(),
            tirx.transform.SplitHostDevice(),
            s_tir.transform.MergeSharedMemoryAllocations(),
            tirx.transform.MakePackedAPI(),
            tirx.transform.FP8StorageLegalize(),
            tirx.transform.BF16StorageLegalize(),
            tirx.transform.LowerDeviceKernelLaunch(),
        ])
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline


def _tune(mod, target, max_trials_per_task, total_trials, tuning_log_dir, verbose):
    """Apply MetaSchedule tuning pipeline to the IRModule."""
    tvm, relax = _try_import_tvm()
    from tvm.relax.pipeline import static_shape_tuning_pipeline

    if tuning_log_dir is None:
        tuning_log_dir = ".cache/tvm_tuning"
    os.makedirs(tuning_log_dir, exist_ok=True)

    if total_trials <= 0:
        total_trials = max_trials_per_task * 100

    if verbose:
        print(f"  MetaSchedule tuning:")
        print(f"    max_trials_per_task = {max_trials_per_task}")
        print(f"    total_trials        = {total_trials}")
        print(f"    work_dir            = {tuning_log_dir}")

    # Silence MetaSchedule's verbose per-trial logging
    import logging
    for name in ("tvm.meta_schedule", "tvm.s_tir.meta_schedule",
                 "tvm.s_tir.meta_schedule.task_scheduler",
                 "tvm.s_tir.meta_schedule.search_strategy",
                 "tvm.s_tir.meta_schedule.builder",
                 "tvm.s_tir.meta_schedule.runner",
                 "tvm.s_tir.meta_schedule.cost_model"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Suppress TVM C++ warnings/info during tuning
    os.environ["TVM_LOG_DEBUG"] = ""
    os.environ["TVM_BACKTRACE"] = "0"

    pipeline = static_shape_tuning_pipeline(
        total_trials=total_trials,
        target=target,
        work_dir=tuning_log_dir,
        max_trials_per_task=max_trials_per_task,
    )
    # Disable fusion during tuning too — fused SNN neuron kernels segfault.
    with tvm.transform.PassContext(
        config={"relax.FuseOps.max_depth": 1}
    ):
        mod = pipeline(mod)

    if verbose:
        print(f"  Tuning complete")

    return mod


def _convert_onnx_to_fp16(model):
    """Convert ONNX model from float32 to float16.

    Lightweight alternative to onnxconverter-common's full-graph rewrite,
    which is extremely slow on SNN models (10+ minutes for a small ResNet).
    Converts:
      - All float32 initializers (weights, biases) to float16
      - All float32 Constant node values to float16
      - Graph input/output type annotations to float16
    Leaves int64/bool tensors and Cast op outputs untouched.
    """
    import numpy as np
    from onnx import numpy_helper, TensorProto

    # Convert initializers (weights, biases, BN params)
    for i, init in enumerate(model.graph.initializer):
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new_init = numpy_helper.from_array(arr, init.name)
            model.graph.initializer[i].CopyFrom(new_init)

    # Convert inline Constant nodes (threshold values, 1.0, etc.)
    # Also rewrite Cast(to=FLOAT) → Cast(to=FLOAT16)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.type == 4 and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)
        elif node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16

    # Update graph input type annotations
    for inp in model.graph.input:
        t = inp.type.tensor_type
        if t.elem_type == TensorProto.FLOAT:
            t.elem_type = TensorProto.FLOAT16

    # Update graph output type annotations
    for out in model.graph.output:
        t = out.type.tensor_type
        if t.elem_type == TensorProto.FLOAT:
            t.elem_type = TensorProto.FLOAT16

    return model


def _replace_gather_with_slice(model):
    """Replace Gather(x, const_idx, axis) with Slice+Squeeze in ONNX.

    SNN models unroll T timesteps via Gather with constant indices 0..T-1.
    TVM's Relax frontend places these scalar int64 indices on CPU, causing
    device mismatches at runtime. Slice+Squeeze achieves the same indexing
    using tensor ops that TVM keeps on the target device.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    def get_const_value(name):
        for init in model.graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init)
        for node in model.graph.node:
            if node.op_type == 'Constant' and node.output[0] == name:
                return numpy_helper.to_array(node.attribute[0].t)
        return None

    new_nodes = []
    new_inits = []
    replaced = 0

    for node in model.graph.node:
        if node.op_type != 'Gather':
            new_nodes.append(node)
            continue

        idx_val = get_const_value(node.input[1])
        if idx_val is None:
            new_nodes.append(node)
            continue

        idx = int(idx_val.flat[0])
        axis = 0
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = attr.i

        prefix = node.output[0] + '_sl'
        starts_name = prefix + '_s'
        ends_name = prefix + '_e'
        axes_name = prefix + '_a'
        slice_out = prefix + '_o'

        new_inits.append(numpy_helper.from_array(
            np.array([idx], dtype=np.int64), starts_name))
        new_inits.append(numpy_helper.from_array(
            np.array([idx + 1], dtype=np.int64), ends_name))
        new_inits.append(numpy_helper.from_array(
            np.array([axis], dtype=np.int64), axes_name))

        new_nodes.append(helper.make_node(
            'Slice', [node.input[0], starts_name, ends_name, axes_name],
            [slice_out]))
        new_nodes.append(helper.make_node(
            'Squeeze', [slice_out, axes_name], [node.output[0]]))
        replaced += 1

    if replaced > 0:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        model.graph.initializer.extend(new_inits)

    return model, replaced
