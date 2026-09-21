"""Minimal TRT inference script for nsys/ncu profiling.

Usage:
    # nsys
    sudo nsys profile -o out python motivation/profile_infer.py --batch 1

    # ncu
    sudo ncu --set basic --csv python motivation/profile_infer.py --batch 1 --warmup 0 --iters 1
"""

import torch
import tensorrt as trt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="motivation/output/spikformer_1_512.engine")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-shape", type=int, nargs="+", default=[3, 32, 32],
                        help="C H W (batch prepended automatically)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    shape = (args.batch, *args.input_shape)
    context.set_input_shape(input_name, shape)

    input_buf = torch.randn(*shape, dtype=torch.float32, device="cuda")
    out_shape = context.get_tensor_shape(output_name)
    output_buf = torch.empty(tuple(out_shape), dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_buf.data_ptr())
    context.set_tensor_address(output_name, output_buf.data_ptr())

    # Warmup
    for _ in range(args.warmup):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # Profiled region
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"Done: batch={args.batch}, warmup={args.warmup}, iters={args.iters}")


if __name__ == "__main__":
    main()
