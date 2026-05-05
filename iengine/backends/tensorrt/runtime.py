"""TensorRT engine runtime: load, allocate, and run inference.

Provides TRTRunner — a thin wrapper around TensorRT execution that
manages device memory, CUDA streams, and batch inference.

Usage:
    runner = TRTRunner("model.engine")
    output = runner.infer(input_tensor)      # single batch
    runner.release()

    # Or as context manager:
    with TRTRunner("model.engine") as runner:
        output = runner.infer(input_tensor)
"""

import numpy as np
import torch
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


class TRTRunner:
    """TensorRT engine inference wrapper.

    Handles engine deserialization, execution context creation,
    device buffer allocation, and synchronous inference.

    Supports dynamic batch sizes within the optimization profile
    range set during engine building.
    """

    def __init__(self, engine_path: str, device: int = 0):
        """Load a serialized TensorRT engine.

        Args:
            engine_path: Path to the .engine file.
            device:      CUDA device index.
        """
        self.trt = _try_import_trt()
        self.device = device
        torch.cuda.set_device(device)

        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        # Inspect I/O tensors
        self._input_names = []
        self._output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        self._output_buffers = {}

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_tensor: (B, C, H, W) float32 or float16 CUDA tensor.

        Returns:
            Output tensor (B, num_classes) on the same device.
        """
        input_tensor = input_tensor.contiguous()
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda(self.device)

        batch_size = input_tensor.shape[0]
        input_name = self._input_names[0]

        # Set input shape (for dynamic batch)
        actual_shape = tuple(input_tensor.shape)
        self.context.set_input_shape(input_name, actual_shape)

        # Set input tensor address
        self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        # Allocate output buffers
        for name in self._output_names:
            out_shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_dtype_to_torch(dtype)

            if (name not in self._output_buffers
                    or self._output_buffers[name].shape != out_shape):
                self._output_buffers[name] = torch.empty(
                    out_shape, dtype=torch_dtype, device=f'cuda:{self.device}'
                )
            self.context.set_tensor_address(
                name, self._output_buffers[name].data_ptr()
            )

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Return first output
        return self._output_buffers[self._output_names[0]]

    def infer_batch(
        self,
        dataloader,
        max_batches: int = 0,
        input_dtype: Optional[torch.dtype] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run inference on a DataLoader, returning (predictions, targets).

        Args:
            dataloader:  PyTorch DataLoader yielding (images, targets).
            max_batches: Max batches to process (0 = all).
            input_dtype: Cast input to this dtype before inference.

        Returns:
            List of (output_tensor, target_tensor) tuples.
        """
        results = []
        for i, (images, targets) in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.cuda(self.device)
            if input_dtype is not None:
                images = images.to(dtype=input_dtype)
            output = self.infer(images)
            results.append((output.clone(), targets.cuda(self.device)))
        return results

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
                            device=f'cuda:{self.device}')

        # Warmup
        for _ in range(n_warmup):
            self.infer(dummy)

        # Measure with CUDA events
        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(n_measure)]
        end_events = [torch.cuda.Event(enable_timing=True)
                      for _ in range(n_measure)]

        for i in range(n_measure):
            start_events[i].record(self.stream)
            self.infer(dummy)
            end_events[i].record(self.stream)

        self.stream.synchronize()

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

    def _trt_dtype_to_torch(self, trt_dtype) -> torch.dtype:
        """Map TensorRT dtype to PyTorch dtype."""
        trt = self.trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)

    def get_engine_info(self) -> dict:
        """Return engine metadata."""
        info = {
            'num_layers': self.engine.num_layers,
            'num_io_tensors': self.engine.num_io_tensors,
            'inputs': {},
            'outputs': {},
        }
        for name in self._input_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            info['inputs'][name] = {'shape': tuple(shape), 'dtype': str(dtype)}
        for name in self._output_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            info['outputs'][name] = {'shape': tuple(shape), 'dtype': str(dtype)}
        return info

    def release(self):
        """Release TensorRT resources."""
        self._output_buffers.clear()
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
