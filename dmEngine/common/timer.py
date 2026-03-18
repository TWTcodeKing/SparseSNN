import time
import torch


class InferenceTimer:
    """Measure inference latency with proper CUDA synchronization."""

    def run(self, model, input_fn, num_warmup=10, num_iters=100, device='cuda'):
        """Time model inference.

        Args:
            model: The model to benchmark.
            input_fn: Callable that returns a single input tensor (or tuple).
            num_warmup: Number of warmup iterations (not timed).
            num_iters: Number of timed iterations.
            device: Device string.

        Returns:
            dict with keys: mean_ms, std_ms, min_ms, max_ms, num_iters
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                x = input_fn()
                _ = model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                x = input_fn()
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # ms

        latencies_t = torch.tensor(latencies)
        return {
            'mean_ms': latencies_t.mean().item(),
            'std_ms': latencies_t.std().item(),
            'min_ms': latencies_t.min().item(),
            'max_ms': latencies_t.max().item(),
            'num_iters': num_iters,
        }


def compare_latency(dense_stats, sparse_stats):
    """Compute speedup from two run() results."""
    speedup = dense_stats['mean_ms'] / max(sparse_stats['mean_ms'], 1e-6)
    return {
        'dense_mean_ms': dense_stats['mean_ms'],
        'sparse_mean_ms': sparse_stats['mean_ms'],
        'speedup': speedup,
        'overhead_ms': sparse_stats['mean_ms'] - dense_stats['mean_ms'],
    }
