"""SEngine: High-performance SNN inference engine.

TensorRT-like API for Spiking Neural Networks.

    import sengine

    # Build from ONNX
    engine = sengine.build("model.onnx", T=4, batch_size=1)
    engine.save("model.sengine")

    # Load and run
    engine = sengine.load("model.sengine")
    output = engine.infer(input_numpy)
    latency = engine.benchmark()
"""

from sengine_edge.engine import SEngine, build, load

__all__ = ['SEngine', 'build', 'load']
__version__ = '0.2.0'
