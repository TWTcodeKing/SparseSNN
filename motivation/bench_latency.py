"""Benchmark TRT engine latency across batch sizes."""

import torch
import tensorrt as trt
import numpy as np
import argparse


class TRTInfer:
    """Minimal TRT inference runner with pre-allocated buffers."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def setup(self, input_shape):
        """Pre-allocate buffers for a specific input shape."""
        self.context.set_input_shape(self.input_name, input_shape)

        self.input_buf = torch.randn(
            *input_shape, dtype=torch.float32, device="cuda"
        )
        out_shape = self.context.get_tensor_shape(self.output_name)
        self.output_buf = torch.empty(
            tuple(out_shape), dtype=torch.float32, device="cuda"
        )

        self.context.set_tensor_address(self.input_name, self.input_buf.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buf.data_ptr())

    def infer(self):
        self.context.execute_async_v3(self.stream.cuda_stream)

    def benchmark(self, warmup=50, iters=200):
        for _ in range(warmup):
            self.infer()
        self.stream.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start_event.record(self.stream)
            self.infer()
            end_event.record(self.stream)
            self.stream.synchronize()
            times.append(start_event.elapsed_time(end_event))

        times = np.array(times)
        return {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p99_ms": float(np.percentile(times, 99)),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="motivation/output/spikformer_1_512.engine")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[3, 32, 32],
                        help="C H W (batch prepended automatically)")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    runner = TRTInfer(args.engine)

    print(f"\n{'Batch':>6} | {'Mean(ms)':>10} | {'Median(ms)':>10} | "
          f"{'Std(ms)':>10} | {'Min(ms)':>10} | {'P99(ms)':>10} | "
          f"{'Throughput':>12}")
    print("-" * 90)

    for batch in args.batches:
        shape = (batch, *args.input_shape)
        runner.setup(shape)
        stats = runner.benchmark(warmup=args.warmup, iters=args.iters)
        throughput = batch / (stats["mean_ms"] / 1000.0)
        print(f"{batch:>6} | {stats['mean_ms']:>10.4f} | {stats['median_ms']:>10.4f} | "
              f"{stats['std_ms']:>10.4f} | {stats['min_ms']:>10.4f} | "
              f"{stats['p99_ms']:>10.4f} | {throughput:>10.1f}/s")
    print()


if __name__ == "__main__":
    main()
