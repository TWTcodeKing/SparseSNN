"""Build TensorRT engines from the motivation ONNX model."""

import os
import tensorrt as trt


def build_engine(onnx_path, engine_path, batch_sizes=(1, 4, 8, 16), fp16=True):
    """Build a TRT engine with optimization profile covering all batch sizes."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    input_name = inp.name
    dims = inp.shape  # (-1, 768)
    hidden = dims[1]

    min_b = min(batch_sizes)
    max_b = max(batch_sizes)
    opt_b = batch_sizes[len(batch_sizes) // 2]

    profile.set_shape(input_name,
                      (min_b, hidden), (opt_b, hidden), (max_b, hidden))
    config.add_optimization_profile(profile)

    print(f"Building TRT engine (FP16={fp16}, batch={min_b}-{max_b}, opt={opt_b})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Saved engine: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")
    return engine_path


if __name__ == "__main__":
    onnx_path = "experiments/motivation/output/model.onnx"
    engine_path = "experiments/motivation/output/model.engine"
    build_engine(onnx_path, engine_path)
