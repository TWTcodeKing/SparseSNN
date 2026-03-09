"""Benchmarking utilities for sparse vs dense execution comparison."""

import time
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Optional

from .density import measure_density


class SparseBenchmark:
    """Benchmark sparse vs dense execution on a model.

    Usage:
        bench = SparseBenchmark()
        dense_stats = bench.run(model, dataloader, device)
        accel.prepare(model)
        sparse_stats = bench.run(model, dataloader, device)
        bench.compare(dense_stats, sparse_stats)
    """

    @staticmethod
    @torch.no_grad()
    def run(model: nn.Module, dataloader, device: torch.device,
            max_samples: int = 100, reset_fn=None) -> dict:
        """Run model on data and measure latency.

        Args:
            model: The model (dense or sparse-instrumented).
            dataloader: Validation dataloader.
            device: CUDA device.
            max_samples: Max samples to benchmark.
            reset_fn: Function to call after each forward pass (e.g., reset_net).

        Returns:
            dict with 'total_time_ms', 'per_sample_ms', 'samples', 'correct', 'total'
        """
        model.eval()
        total_time = 0.0
        correct = 0
        total = 0
        processed = 0

        # Warmup
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if reset_fn:
                reset_fn(model)
            break

        torch.cuda.synchronize()
        for images, targets in dataloader:
            if processed >= max_samples:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            output = model(images)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            if reset_fn:
                reset_fn(model)

            total_time += (t1 - t0)
            preds = output.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            processed += targets.size(0)

        per_sample = (total_time / max(total, 1)) * 1000  # ms

        return {
            'total_time_ms': total_time * 1000,
            'per_sample_ms': per_sample,
            'samples': total,
            'correct': correct,
            'accuracy': correct / max(total, 1) * 100,
        }

    @staticmethod
    def compare(dense_stats: dict, sparse_stats: dict,
                backend_name: str = 'sparse') -> dict:
        """Compare dense vs sparse benchmark results."""
        speedup = dense_stats['per_sample_ms'] / max(sparse_stats['per_sample_ms'], 1e-6)
        acc_delta = sparse_stats['accuracy'] - dense_stats['accuracy']

        result = {
            'backend': backend_name,
            'dense_ms': dense_stats['per_sample_ms'],
            'sparse_ms': sparse_stats['per_sample_ms'],
            'speedup': speedup,
            'dense_acc': dense_stats['accuracy'],
            'sparse_acc': sparse_stats['accuracy'],
            'acc_delta': acc_delta,
        }

        print(f"\n{'='*60}")
        print(f"  Benchmark: Dense vs {backend_name}")
        print(f"{'='*60}")
        print(f"  Dense:  {dense_stats['per_sample_ms']:.3f} ms/sample  "
              f"Acc: {dense_stats['accuracy']:.2f}%")
        print(f"  Sparse: {sparse_stats['per_sample_ms']:.3f} ms/sample  "
              f"Acc: {sparse_stats['accuracy']:.2f}%")
        print(f"  Speedup: {speedup:.2f}x  |  Acc delta: {acc_delta:+.2f}%")
        print(f"{'='*60}\n")

        return result


@torch.no_grad()
def profile_layer_density(model: nn.Module, dataloader, device: torch.device,
                          max_samples: int = 50, reset_fn=None) -> dict:
    """Profile per-layer input density for sparse execution planning.

    Returns:
        {module_name: {'type': str, 'density': float, 'shape': tuple}}
    """
    densities = defaultdict(list)
    shapes = {}
    types = {}

    hooks = []

    def make_hook(name):
        def hook_fn(module, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            densities[name].append(measure_density(x))
            shapes[name] = tuple(x.shape)
            types[name] = module.__class__.__name__
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    processed = 0
    for images, targets in dataloader:
        if processed >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        _ = model(images)
        if reset_fn:
            reset_fn(model)
        processed += images.size(0)

    for h in hooks:
        h.remove()

    result = {}
    for name in densities:
        import numpy as np
        result[name] = {
            'type': types[name],
            'density': float(np.mean(densities[name])),
            'shape': shapes[name],
        }

    return result
