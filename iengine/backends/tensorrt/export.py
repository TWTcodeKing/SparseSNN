"""PyTorch SNN model → ONNX export.

Handles SNN-specific concerns:
  - Neuron membrane potential reset before tracing
  - Temporal dimension baked into the model (T is a construction param)
  - Input shape: (B, C, H, W) for static images — model internally repeats T times
  - Dynamic batch size support via dynamic_axes

Usage:
    from iengine.backends.tensorrt.export import export_onnx
    export_onnx(model, onnx_path, input_shape=(1, 3, 32, 32), opset=17)
"""

import os

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from models.neurons import reset_net


def export_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    opset: int = 17,
    dynamic_batch: bool = True,
    fp16_inputs: bool = False,
    simplify: bool = True,
    verbose: bool = True,
) -> str:
    """Export a PyTorch SNN model to ONNX.

    The model should accept (B, C, H, W) input and handle temporal
    unrolling internally. Neuron states are reset before tracing to
    ensure a clean forward pass.

    Args:
        model:         PyTorch SNN model (eval mode).
        onnx_path:     Output .onnx file path.
        input_shape:   (B, C, H, W) dummy input shape for tracing.
        opset:         ONNX opset version (17+ recommended for TRT 10).
        dynamic_batch: Enable dynamic batch dimension.
        fp16_inputs:   If True, trace with fp16 dummy input.
        simplify:      Run onnxsim to simplify the graph.
        verbose:       Print export details.

    Returns:
        Path to the saved ONNX file.
    """
    onnx_path = str(Path(onnx_path).resolve())
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    dtype = torch.float16 if fp16_inputs else torch.float32

    # Reset neuron states for a clean trace
    model.eval()
    reset_net(model)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    if verbose:
        print(f"Exporting ONNX: input={list(input_shape)}, opset={opset}, "
              f"dynamic_batch={dynamic_batch}")

    export_kwargs = dict(
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    def _do_export(model, device):
        dummy = torch.randn(*input_shape, dtype=dtype, device=device)
        with torch.no_grad():
            try:
                torch.onnx.export(model, dummy, onnx_path, dynamo=False, **export_kwargs)
            except TypeError:
                torch.onnx.export(model, dummy, onnx_path, **export_kwargs)
        reset_net(model)

    # Try GPU first (produces correct external data); fall back to CPU on OOM
    try:
        _do_export(model, device)
    except torch.OutOfMemoryError:
        if verbose:
            print(f"  GPU OOM, retrying on CPU...")
        torch.cuda.empty_cache()
        model_cpu = model.cpu()
        _do_export(model_cpu, 'cpu')
        model.cuda()

    # PyTorch 2.9+ externalizes large Constant tensors (e.g. neuron membrane
    # init zeros) but may write 0-byte files. Fix: load all external data,
    # regenerate broken ones, then save everything to a single .data file.
    import onnx
    import numpy as np
    onnx_dir = os.path.dirname(onnx_path) or '.'

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    has_external = any(t.data_location == 1 for t in onnx_model.graph.initializer)
    has_broken_const = False
    for node in onnx_model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.type == 4 and attr.t.data_location == 1:
                    has_broken_const = True
                    break
    del onnx_model

    if has_external or has_broken_const:
        # Reload with all valid external data
        onnx_model = onnx.load(onnx_path, load_external_data=True)

        # Fix broken Constant nodes: replace 0-byte external with inline zeros
        for node in onnx_model.graph.node:
            if node.op_type != 'Constant':
                continue
            for attr in node.attribute:
                if attr.type != 4 or attr.t.data_location != 1:
                    continue
                shape = list(attr.t.dims)
                # External data was broken (0 bytes) — fill with zeros
                if len(attr.t.raw_data) == 0:
                    dtype_map = {1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32}
                    dt = dtype_map.get(attr.t.data_type, np.float32)
                    attr.t.raw_data = np.zeros(shape, dtype=dt).tobytes()
                attr.t.data_location = 0
                del attr.t.external_data[:]

        # Save with single external data file for all large tensors
        data_file = os.path.basename(onnx_path) + '.data'
        onnx.save_model(onnx_model, onnx_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=data_file,
                        size_threshold=1024)

        # Clean up stale per-tensor external files
        data_path = os.path.join(onnx_dir, data_file)
        keep = {os.path.abspath(data_path), os.path.abspath(onnx_path)}
        for f in os.listdir(onnx_dir):
            fp = os.path.join(onnx_dir, f)
            if os.path.abspath(fp) in keep:
                continue
            if f.startswith(('_', 'onnx__')) or f in ('fc.weight', 'fc.bias'):
                os.remove(fp)

        del onnx_model
        if verbose:
            print(f"  Weights: {data_file} ({os.path.getsize(data_path)/1e6:.1f} MB)")

    if verbose:
        print(f"  Saved raw ONNX to {onnx_path}")

    # Simplify (skip for models with external data — onnxsim can't serialize them)
    if simplify and not has_external:
        try:
            from onnxsim import simplify as onnxsim_simplify

            onnx_model = onnx.load(onnx_path)
            simplified, ok = onnxsim_simplify(
                onnx_model,
                skipped_optimizers=['fuse_bn_into_conv'],
            )
            if ok:
                onnx.save(simplified, onnx_path)
                if verbose:
                    print(f"  Simplified ONNX graph")
            else:
                if verbose:
                    print(f"  onnxsim: simplification returned False, keeping original")
        except ImportError:
            if verbose:
                print(f"  onnxsim not installed, skipping simplification")
        except Exception as e:
            if verbose:
                print(f"  onnxsim failed ({e}), keeping original (non-fatal)")

    if verbose:
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Final ONNX: {size_mb:.1f} MB")

    return onnx_path


def validate_onnx(
    model: nn.Module,
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Validate ONNX model output matches PyTorch output.

    Args:
        model:       Original PyTorch model.
        onnx_path:   Path to exported ONNX.
        input_shape: Input shape for validation.
        atol/rtol:   Tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import onnxruntime as ort
    import numpy as np

    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)

    # PyTorch reference
    model.eval()
    reset_net(model)
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    reset_net(model)

    # ONNX runtime
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'input': dummy.cpu().numpy()})[0]

    match = np.allclose(pt_out, ort_out, atol=atol, rtol=rtol)
    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"  ONNX validation: {'PASS' if match else 'FAIL'} "
          f"(max_diff={max_diff:.6f}, atol={atol})")
    return match
