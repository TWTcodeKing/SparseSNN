"""Minimal TRT engine runner for nsys/ncu profiling.

Usage:
    python utils/profile_trt_engine.py <engine_path> [--warmup 20] [--iters 50]

Runs inference with CUDA events for timing. Designed to be wrapped by:
    nsys profile --no-cpu-sampling -o <output> python utils/profile_trt_engine.py ...
    ncu --set full -o <output> python utils/profile_trt_engine.py ...
"""

import sys, os, ctypes, argparse, torch

# Load the TRT shared lib (from the tensorrt-cu12-libs wheel) globally before import
import tensorrt_libs
ctypes.CDLL(os.path.join(os.path.dirname(tensorrt_libs.__file__), 'libnvinfer.so.10'),
            mode=ctypes.RTLD_GLOBAL)
import tensorrt as trt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', help='Path to .engine file')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(args.engine, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Discover I/O
    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    inp_shape = tuple(engine.get_tensor_shape(inp_name))
    # Replace dynamic dims with the opt shape
    inp_shape = tuple(max(1, s) for s in inp_shape)
    ctx.set_input_shape(inp_name, inp_shape)
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    print(f"Engine: {args.engine}")
    print(f"Input:  {inp_name} {inp_shape}")
    print(f"Output: {out_name} {out_shape}")

    x = torch.randn(*inp_shape, dtype=torch.float32, device='cuda:0')
    y = torch.zeros(*out_shape, dtype=torch.float32, device='cuda:0')
    ctx.set_tensor_address(inp_name, x.data_ptr())
    ctx.set_tensor_address(out_name, y.data_ptr())
    stream = torch.cuda.Stream(device=0)

    # Warmup
    for _ in range(args.warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Measured iterations
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(stream)
    for _ in range(args.iters):
        ctx.execute_async_v3(stream.cuda_stream)
    e.record(stream)
    stream.synchronize()

    ms = s.elapsed_time(e) / args.iters
    batch = inp_shape[0]
    print(f"Latency: {ms:.3f} ms/iter  |  Throughput: {batch*1000/ms:.0f} img/s")


if __name__ == '__main__':
    main()
