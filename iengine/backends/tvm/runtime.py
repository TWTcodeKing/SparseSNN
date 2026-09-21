"""TVM compiled module runtime: load and run inference.

Provides TVMRunner — mirrors the TRTRunner interface from
iengine.backends.tensorrt.runtime.

TVM v0.25+ uses Relax VirtualMachine for execution:
  load .so -> VirtualMachine -> vm["main"](input) -> output

Usage:
    runner = TVMRunner("model.so", input_shape=(1, 3, 32, 32))
    output = runner.infer(input_tensor)
    runner.release()

    # Or as context manager:
    with TVMRunner("model.so", input_shape=(1, 3, 32, 32)) as runner:
        result = runner.benchmark_latency(input_shape=(1, 3, 32, 32))
"""

import numpy as np
import torch
from typing import Optional


def _try_import_tvm():
    try:
        import tvm
        import tvm_ffi
        from tvm.runtime.vm import VirtualMachine
        return tvm, tvm_ffi, VirtualMachine
    except ImportError as e:
        raise ImportError(
            f"tvm import failed: {e}\n"
            "Run with an interpreter that has TVM (Relax) installed; see README, section TVM baseline"
        ) from e


class TVMRunner:
    """TVM compiled module inference wrapper.

    Loads a compiled .so library, creates a Relax VirtualMachine on the
    specified CUDA device, and runs synchronous inference.
    """

    def __init__(self, lib_path: str, input_shape: tuple[int, ...],
                 device: int = 0):
        """Load a compiled TVM module.

        Args:
            lib_path:    Path to the compiled .so file.
            input_shape: (B, C, H, W) input shape (must match build shape).
            device:      CUDA device index.
        """
        tvm, tvm_ffi, VirtualMachine = _try_import_tvm()

        self.device_id = device
        self.input_shape = input_shape
        self._tvm = tvm
        self._tvm_ffi = tvm_ffi

        self._dev = tvm.cuda(device)
        self._lib = tvm.runtime.load_module(lib_path)
        self._vm = VirtualMachine(self._lib, self._dev)

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_tensor: (B, C, H, W) float32 CUDA tensor.

        Returns:
            Output tensor on the same CUDA device.
        """
        input_tensor = input_tensor.contiguous()
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda(self.device_id)

        # Zero-copy: torch CUDA tensor -> TVM Tensor via DLPack
        tvm_input = self._tvm_ffi.from_dlpack(input_tensor)
        tvm_output = self._vm["main"](tvm_input)

        # Convert back to torch via DLPack
        if isinstance(tvm_output, self._tvm_ffi.Tensor):
            return torch.from_dlpack(tvm_output)
        # Some models return a tuple
        return torch.from_dlpack(tvm_output[0])

    def benchmark_latency(
        self,
        input_shape: tuple[int, ...],
        n_warmup: int = 20,
        n_measure: int = 100,
        input_dtype: torch.dtype = torch.float32,
    ) -> dict:
        """Benchmark inference latency.

        Args:
            input_shape: (B, C, H, W) input shape.
            n_warmup:    Warmup iterations.
            n_measure:   Measurement iterations.
            input_dtype: Input tensor dtype.

        Returns:
            Dict with 'mean_ms', 'median_ms', 'std_ms', 'throughput_img_s'.
        """
        dummy = torch.randn(*input_shape, dtype=input_dtype,
                            device=f'cuda:{self.device_id}')
        stream = torch.cuda.Stream(device=self.device_id)

        # Use raw VM call for benchmarking (avoids DLPack overhead)
        tvm_dummy = self._tvm_ffi.from_dlpack(dummy)
        main_fn = self._vm["main"]

        # Warmup
        for _ in range(n_warmup):
            main_fn(tvm_dummy)
        stream.synchronize()

        # Measure with CUDA events
        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(n_measure)]
        end_events = [torch.cuda.Event(enable_timing=True)
                      for _ in range(n_measure)]

        for i in range(n_measure):
            start_events[i].record(stream)
            main_fn(tvm_dummy)
            end_events[i].record(stream)

        stream.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_arr = torch.tensor(times)

        batch_size = input_shape[0]
        mean_ms = times_arr.mean().item()

        return {
            'mean_ms': mean_ms,
            'median_ms': times_arr.median().item(),
            'std_ms': times_arr.std().item(),
            'min_ms': times_arr.min().item(),
            'max_ms': times_arr.max().item(),
            'throughput_img_s': batch_size * 1000.0 / mean_ms,
        }

    def release(self):
        """Release TVM resources."""
        if hasattr(self, '_vm'):
            del self._vm
        if hasattr(self, '_lib'):
            del self._lib

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
