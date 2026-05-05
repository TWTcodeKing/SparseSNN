"""ONNX → TensorRT engine builder.

Two pipelines:
  - Dense FP16:  standard TensorRT optimization
  - Sparse FP16: enables kSPARSE_WEIGHTS to exploit 2:4 Sparse Tensor Cores
                  (weights must already be pruned to 2:4 pattern by SBC)

Optional INT8 quantization with calibration data.

Usage:
    from iengine.backends.tensorrt.builder import build_engine
    build_engine("model.onnx", "model.engine", sparse=True)
"""

import os
from pathlib import Path
from typing import Optional


def _try_import_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        raise ImportError(
            "tensorrt not found. Install with:\n"
            "  pip install tensorrt-cu12\n"
            "On Jetson, TensorRT is system-installed — if using conda/venv, "
            "link it:\n"
            "  ln -s /usr/lib/python3.*/dist-packages/tensorrt* "
            "$(python -c 'import site; print(site.getsitepackages()[0])')/"
        )


class _TRTLogger:
    """Thin wrapper to create a TRT logger at call time."""

    @staticmethod
    def get(verbose: bool = False):
        trt = _try_import_trt()
        severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        return trt.Logger(severity)


def build_engine(
    onnx_path: str,
    engine_path: str,
    sparse: bool = False,
    fp16: bool = True,
    int8: bool = False,
    calibrator=None,
    max_batch_size: int = 64,
    workspace_gb: float = 4.0,
    min_batch: int = 1,
    opt_batch: int = 32,
    max_batch: int = 64,
    verbose: bool = True,
    timing_cache_path: Optional[str] = None,
) -> str:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path:     Path to input ONNX file.
        engine_path:   Path to save the built engine.
        sparse:        Enable kSPARSE_WEIGHTS for 2:4 Sparse Tensor Cores.
                       Weights must already have valid 2:4 sparsity pattern.
        fp16:          Enable FP16 precision.
        int8:          Enable INT8 precision (requires calibrator).
        calibrator:    INT8 calibrator instance (from calibrator.py).
        max_batch_size: Maximum batch size for implicit batch mode.
        workspace_gb:  Maximum workspace memory in GB.
        min_batch:     Minimum batch for optimization profile.
        opt_batch:     Optimal batch for optimization profile.
        max_batch:     Maximum batch for optimization profile.
        verbose:       Print build progress.
        timing_cache_path: Path to timing cache file for faster rebuilds.

    Returns:
        Path to the saved engine file.
    """
    trt = _try_import_trt()
    logger = _TRTLogger.get(verbose=False)

    engine_path = str(Path(engine_path).resolve())
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)

    mode_tag = "sparse 2:4" if sparse else "dense"
    prec_parts = []
    if fp16:
        prec_parts.append("FP16")
    if int8:
        prec_parts.append("INT8")
    prec_tag = "+".join(prec_parts) if prec_parts else "FP32"

    if verbose:
        print(f"Building TensorRT engine: {mode_tag} {prec_tag}")
        print(f"  ONNX:   {onnx_path}")
        print(f"  Engine: {engine_path}")
        print(f"  Batch:  min={min_batch}, opt={opt_batch}, max={max_batch}")

    # Create builder and network
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX (use absolute path so TRT finds external data files)
    abs_onnx = os.path.abspath(onnx_path)
    if not parser.parse_from_file(abs_onnx):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    if verbose:
        print(f"  Parsed ONNX: {network.num_inputs} inputs, "
              f"{network.num_outputs} outputs, {network.num_layers} layers")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    # Precision flags
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("  WARNING: Platform does not have fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        if not builder.platform_has_fast_int8:
            print("  WARNING: Platform does not have fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            config.int8_calibrator = calibrator
        else:
            raise ValueError("INT8 requires a calibrator")

    # Sparse weights — TensorRT detects 2:4 pattern and uses Sparse TC
    if sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        if verbose:
            print(f"  Enabled SPARSE_WEIGHTS (2:4 Sparse Tensor Cores)")

    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape  # (batch, C, H, W) with -1 for dynamic

    # Replace dynamic dimension with profile values
    shape_min = list(input_shape)
    shape_opt = list(input_shape)
    shape_max = list(input_shape)

    # Find dynamic dimensions (value -1) and set profiles
    for i in range(len(shape_min)):
        if shape_min[i] == -1:
            shape_min[i] = min_batch
            shape_opt[i] = opt_batch
            shape_max[i] = max_batch

    profile.set_shape(
        input_tensor.name,
        tuple(shape_min), tuple(shape_opt), tuple(shape_max),
    )
    config.add_optimization_profile(profile)

    # Timing cache for faster rebuilds
    timing_cache = None
    if timing_cache_path and os.path.exists(timing_cache_path):
        with open(timing_cache_path, 'rb') as f:
            timing_cache = config.create_timing_cache(f.read())
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        if verbose:
            print(f"  Loaded timing cache: {timing_cache_path}")
    else:
        timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

    # Build engine
    if verbose:
        print(f"  Building engine (this may take a few minutes)...")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized)

    # Save timing cache
    if timing_cache_path:
        cache_data = config.get_timing_cache()
        with open(timing_cache_path, 'wb') as f:
            f.write(cache_data.serialize())
        if verbose:
            print(f"  Saved timing cache: {timing_cache_path}")

    if verbose:
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  Engine saved: {size_mb:.1f} MB")

    return engine_path


def build_dense_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    **kwargs,
) -> str:
    """Build a dense (no sparsity) TensorRT engine.

    Convenience wrapper around build_engine with sparse=False.
    """
    return build_engine(onnx_path, engine_path, sparse=False, fp16=fp16, **kwargs)


def build_sparse_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    **kwargs,
) -> str:
    """Build a 2:4 sparse TensorRT engine.

    Convenience wrapper around build_engine with sparse=True.
    The ONNX model must contain weights with valid 2:4 sparsity pattern
    (produced by the SBC pruning pipeline in sparse/snn_sbc.py).
    """
    return build_engine(onnx_path, engine_path, sparse=True, fp16=fp16, **kwargs)
