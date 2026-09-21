"""sengine_cpu: CPU inference engine for Spiking Neural Networks.

Self-contained CPU port of sengine: a C runtime (csrc/) with native fused
Conv+BN+IF/LIF kernels; an optional TVM backend (see tvm_env.py).

    import sengine_cpu

    engine = sengine_cpu.build("model.onnx", T=4)
    engine.save("model.sengine-cpu")

    engine = sengine_cpu.load("model.sengine-cpu")
    output = engine.infer(input_numpy)
    latency = engine.benchmark()
"""

__version__ = '0.1.0'


def build(onnx_path: str, T: int = 4, batch_size: int = 1,
          n_threads: int = 0, target=None):
    """Build CPU inference engine from ONNX model."""
    from sengine_cpu.build.engine_builder import CPUEngineBuilder
    return CPUEngineBuilder(onnx_path, T=T, batch_size=batch_size,
                             n_threads=n_threads, target=target).build()


def load(path: str, n_threads: int = 0):
    """Load a saved .sengine-cpu engine file."""
    from sengine_cpu.build.engine_builder import CPUEngineBuilder
    return CPUEngineBuilder.load(path, n_threads=n_threads)
