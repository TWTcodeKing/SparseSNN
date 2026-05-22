"""Architecture-specific DAG extractors for SNN models.

Each module provides an extract function that takes (model, input_shape)
and returns an OperatorDAG. The top-level extract_dag() in graph_ir.py
dispatches to the appropriate extractor based on model type.
"""

from sengine_edge.tdl.model_dag.sewresnet import (
    extract_sewresnet_dag, extract_sewresnet_cifar_dag,
)
from sengine_edge.tdl.model_dag.msresnet import (
    extract_msresnet18_dag, extract_msresnet_cifar_dag, extract_msresnet104_dag,
)
from sengine_edge.tdl.model_dag.spikformer import extract_spikformer_dag
from sengine_edge.tdl.model_dag.spikingresformer import extract_spikingresformer_dag
from sengine_edge.tdl.model_dag.metaformer import extract_metaformer_dag
from sengine_edge.tdl.model_dag.qkformer import extract_qkformer_dag
from sengine_edge.tdl.model_dag.maxformer import extract_maxformer_dag
