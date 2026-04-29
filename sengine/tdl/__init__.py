"""Temporal Dimension Lowering (TDL) — platform-agnostic SNN graph transforms.

TDL systematically lowers the SNN temporal dimension from an explicit tensor
axis (T, B, C, H, W) into implicit operator state (T*B, C, H, W), enabling
any spatial-IR inference compiler to optimize SNN graphs as efficiently as
ANN graphs.

Three transformations:
  TDL-1: T-Axis Absorption           — stateless wrappers skip 5D reshape
  TDL-2: Stateful Operator Extraction — neurons become fused kernels
  TDL-3: Temporal Attention Decomposition — spike attention flattened to 4D

No platform dependencies — pure PyTorch + standard Python.
"""

from sengine.tdl.transforms import TDLTransform, export_with_fused_neurons
from sengine.tdl.analysis import collect_neuron_params, classify_modules
from sengine.tdl.graph_ir import OpNode, Edge, OperatorDAG, extract_dag
from sengine.tdl.cost_model import HardwareSpec, CostModel
from sengine.tdl.temporal_unroll import TemporalNode, TemporalDAG, build_temporal_dag
from sengine.tdl.slicegraph import SliceGraphDP, Partition
