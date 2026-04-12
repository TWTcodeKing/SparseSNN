"""INT8 calibration for TensorRT engine building.

Provides an IInt8EntropyCalibrator2 that feeds SNN calibration data
(from the training set) to TensorRT's INT8 quantization pipeline.

The calibrator caches calibration results to disk so repeated builds
don't re-run calibration.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional

import torch


def _try_import_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        raise ImportError(
            "tensorrt not found. Install with: "
            "uv pip install tensorrt-cu12 --python .venv/bin/python"
        )


class ImageNetCalibrator:
    """INT8 entropy calibrator backed by a PyTorch DataLoader.

    Feeds batches from the dataloader to TensorRT calibration.
    Caches calibration table to disk for reuse.
    """

    def __init__(
        self,
        dataloader,
        cache_file: str = "calibration.cache",
        max_batches: int = 64,
        input_name: str = "input",
    ):
        trt = _try_import_trt()
        self._trt = trt

        self.dataloader = dataloader
        self.cache_file = cache_file
        self.max_batches = max_batches
        self.input_name = input_name

        self._iter = iter(dataloader)
        self._batch_idx = 0

        # Pre-allocate device buffer for one batch
        first_batch = next(iter(dataloader))[0]
        self._batch_shape = first_batch.shape
        self._device_input = torch.empty(
            self._batch_shape, dtype=torch.float32, device='cuda'
        )
        self._d_input = self._device_input.data_ptr()

    def get_batch_size(self):
        return self._batch_shape[0]

    def get_batch(self, names):
        if self._batch_idx >= self.max_batches:
            return None
        try:
            images, _ = next(self._iter)
        except StopIteration:
            return None

        self._device_input.copy_(images.float())
        self._batch_idx += 1
        return [self._d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def make_calibrator(
    dataloader,
    cache_dir: str = "trt_engines",
    cache_name: str = "calibration.cache",
    max_batches: int = 64,
):
    """Create an INT8 calibrator from a DataLoader.

    Returns a calibrator instance compatible with TensorRT builder config.
    The calibrator inherits from trt.IInt8EntropyCalibrator2 at runtime
    to avoid import errors when tensorrt is not installed.

    Args:
        dataloader:  PyTorch DataLoader yielding (images, labels).
        cache_dir:   Directory for calibration cache file.
        cache_name:  Cache file name.
        max_batches: Number of batches to calibrate on.

    Returns:
        Calibrator instance.
    """
    trt = _try_import_trt()

    cache_path = os.path.join(cache_dir, cache_name)
    os.makedirs(cache_dir, exist_ok=True)

    # Dynamically create a class that inherits from trt.IInt8EntropyCalibrator2
    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self._inner = ImageNetCalibrator(
                dataloader, cache_path, max_batches
            )

        def get_batch_size(self):
            return self._inner.get_batch_size()

        def get_batch(self, names):
            return self._inner.get_batch(names)

        def read_calibration_cache(self):
            return self._inner.read_calibration_cache()

        def write_calibration_cache(self, cache):
            self._inner.write_calibration_cache(cache)

    return _Calibrator()
