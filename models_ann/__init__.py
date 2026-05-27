"""ANN equivalents of SNN models for TVM/compiler benchmarking.

Replaces spiking neurons with ReLU, removes temporal dimension T.
Architecture topology is identical to the SNN versions — only activations differ.

Usage:
    from models_ann import build_ann_model
    model = build_ann_model('sew_resnet18', num_classes=100, in_channels=3)
    # model accepts (B, C, H, W), no T dimension
"""

from models_ann.convert import build_ann_model, convert_snn_to_ann

__all__ = ['build_ann_model', 'convert_snn_to_ann']
