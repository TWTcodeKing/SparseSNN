"""NVIDIA TensorRT backend for SNN inference.

Provides ONNX export, TRT engine building with 2:4 sparse support,
and fused neuron plugins (IPluginV3).
"""

from iengine.backends.tensorrt.export import export_onnx
from iengine.backends.tensorrt.builder import build_engine
from iengine.backends.tensorrt.runtime import TRTRunner
