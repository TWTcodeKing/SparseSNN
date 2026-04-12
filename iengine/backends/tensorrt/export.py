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

    dummy = torch.randn(*input_shape, device=device, dtype=dtype)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    if verbose:
        print(f"Exporting ONNX: input={list(input_shape)}, opset={opset}, "
              f"dynamic_batch={dynamic_batch}")

    with torch.no_grad():
        torch.onnx.export(
            model, dummy,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    reset_net(model)

    if verbose:
        print(f"  Saved raw ONNX to {onnx_path}")

    # Simplify
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnxsim_simplify

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            simplified, ok = onnxsim_simplify(onnx_model)
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
                print(f"  onnxsim failed ({e}), keeping original")

    if verbose:
        import os
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
