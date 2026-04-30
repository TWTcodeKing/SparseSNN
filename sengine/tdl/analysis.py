"""Module analysis for TDL transforms.

Classifies SNN model modules into stateless wrappers, stateful neurons,
attention blocks, and other. Collects neuron parameters for fused kernel
generation.

Platform-agnostic — pure PyTorch, no inference engine dependencies.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Neuron parameter collection
# ---------------------------------------------------------------------------

def collect_neuron_params(model: nn.Module) -> OrderedDict:
    """Extract neuron parameters from a PyTorch SNN model.

    Walks the model and collects parameters needed for fused neuron kernels.
    Supports: MultiStepLIFNeuron, MultiStepIFNeuron, MSNeuron.

    Returns:
        OrderedDict mapping neuron module name → param dict with keys:
        'type' ('LIF'|'IF'|'MS'), 'T', and neuron-specific params.
    """
    from models.neurons import MultiStepLIFNeuron, MultiStepIFNeuron

    # MSNeuron may not exist in all configs
    try:
        from models.msresnet import MSNeuron
    except ImportError:
        MSNeuron = None

    params = OrderedDict()
    T = getattr(model, 'T', 4)

    for name, module in model.named_modules():
        if isinstance(module, MultiStepLIFNeuron):
            inner = module.neuron
            tau = inner.tau
            if isinstance(tau, (nn.Parameter, torch.Tensor)):
                tau = tau.item()
            vth = inner.v_threshold
            if isinstance(vth, (nn.Parameter, torch.Tensor)):
                vth = vth.item()
            params[name] = {
                'type': 'LIF',
                'T': T,
                'tau': float(tau),
                'v_threshold': float(vth),
                'v_reset': float(inner.v_reset) if inner.v_reset is not None else 0.0,
                'hard_reset': inner.v_reset is not None,
            }
        elif isinstance(module, MultiStepIFNeuron):
            inner = module.neuron
            vth = inner.v_threshold
            if isinstance(vth, (nn.Parameter, torch.Tensor)):
                vth = vth.item()
            params[name] = {
                'type': 'IF',
                'T': T,
                'v_threshold': float(vth),
                'v_reset': float(inner.v_reset) if inner.v_reset is not None else 0.0,
                'hard_reset': inner.v_reset is not None,
            }
        elif MSNeuron is not None and isinstance(module, MSNeuron):
            params[name] = {
                'type': 'MS',
                'T': T,
                'decay': float(module.decay),
                'thresh': float(module.thresh),
            }

    return params


# ---------------------------------------------------------------------------
# Module classification
# ---------------------------------------------------------------------------

def is_neuron(module: nn.Module) -> bool:
    """Check if a module is a spiking neuron (stateful temporal operator)."""
    from models.neurons import MultiStepLIFNeuron, MultiStepIFNeuron
    if isinstance(module, (MultiStepLIFNeuron, MultiStepIFNeuron)):
        return True
    try:
        from models.msresnet import MSNeuron
        if isinstance(module, MSNeuron):
            return True
    except ImportError:
        pass
    return False


def is_stateless_wrapper(module: nn.Module) -> bool:
    """Check if a module is a temporal wrapper around stateless ops.

    Detects three patterns:
    A. SeqToANNContainer: has .module attr wrapping stateless children
    B. _MultiStep*: subclass of nn.Conv2d/MaxPool2d/etc with overridden forward
    C. BN wrapper: has .bn attr that is nn.BatchNorm2d
    """
    # Pattern A: SeqToANNContainer-like (has .module wrapping stateless ops)
    if hasattr(module, 'module') and not _has_neuron_children(module):
        inner = module.module
        if isinstance(inner, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
                              nn.AdaptiveAvgPool2d, nn.Linear)):
            return True
        if isinstance(inner, nn.Sequential):
            return all(_is_stateless_leaf(child) for child in inner)

    # Pattern B: _MultiStep* (subclass of standard layer with overridden forward)
    for base in (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear):
        if isinstance(module, base) and type(module) is not base:
            return True

    # Pattern C: BN wrapper (has .bn attribute)
    if hasattr(module, 'bn') and isinstance(module.bn, nn.BatchNorm2d):
        if not is_neuron(module):
            return True

    return False


def is_spike_attention(module: nn.Module) -> bool:
    """Check if a module is a spike-driven attention block.

    Detects three families:
      - DSSA (SpikingResformer): num_heads + firing_rate_x/attn buffers
      - SSA  (SpikFormer/MaxFormer): num_heads + q_lif/k_lif/v_lif neurons
      - Token_QK_Attention (MS_QKFormer): num_heads + q_lif/k_lif (no v_lif)
    """
    if not hasattr(module, 'num_heads'):
        return False
    # DSSA pattern
    if hasattr(module, 'firing_rate_x') and hasattr(module, 'firing_rate_attn'):
        return True
    # SSA / Token_QK_Attention pattern: has q_lif and k_lif neuron children
    if hasattr(module, 'q_lif') and hasattr(module, 'k_lif'):
        if is_neuron(module.q_lif) and is_neuron(module.k_lif):
            return True
    return False


def get_attention_type(module: nn.Module) -> str | None:
    """Classify which attention variant a module is.

    Returns:
        'dssa'          — SpikingResformer DSSA
        'spikformer_ssa' — SpikFormer SSA (Linear QKV, standard matmul)
        'maxformer_ssa'  — MaxFormer SSA (Conv1d QKV, reversed matmul)
        'token_qka'      — MS_QKFormer Token_QK_Attention (no V)
        None             — not an attention module
    """
    if not is_spike_attention(module):
        return None
    # DSSA: has firing_rate buffers
    if hasattr(module, 'firing_rate_x'):
        return 'dssa'
    # SpikFormer SSA: has q_linear (nn.Linear projections)
    if hasattr(module, 'q_linear'):
        return 'spikformer_ssa'
    # Distinguish MaxFormer SSA vs Token_QK_Attention:
    # SSA has v_conv/v_lif; Token_QK_Attention does not
    if hasattr(module, 'v_conv') and hasattr(module, 'v_lif'):
        return 'maxformer_ssa'
    if hasattr(module, 'q_conv') and not hasattr(module, 'v_conv'):
        return 'token_qka'
    return None


def is_spikformer_mlp(module: nn.Module) -> bool:
    """Check if a module is a SpikFormer MLP (Linear + BN1d + LIF).

    Detected by: fc1_linear + fc1_bn + fc1_lif + fc2_linear.
    """
    return (hasattr(module, 'fc1_linear') and hasattr(module, 'fc1_bn')
            and hasattr(module, 'fc1_lif') and hasattr(module, 'fc2_linear'))


def classify_modules(model: nn.Module) -> OrderedDict:
    """Classify all modules in an SNN model.

    Returns:
        OrderedDict mapping module name → category string:
        'neuron', 'stateless_wrapper', 'attention', 'other'
    """
    result = OrderedDict()
    for name, module in model.named_modules():
        if is_neuron(module):
            result[name] = 'neuron'
        elif is_spike_attention(module):
            result[name] = 'attention'
        elif is_stateless_wrapper(module):
            result[name] = 'stateless_wrapper'
        else:
            result[name] = 'other'
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_neuron_children(module: nn.Module) -> bool:
    """Check if any descendant is a spiking neuron."""
    for child in module.modules():
        if child is module:
            continue
        if is_neuron(child):
            return True
    return False


def _is_stateless_leaf(module: nn.Module) -> bool:
    """Check if a module is a stateless leaf op (Conv, BN, Pool, etc.)."""
    return isinstance(module, (
        nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d,
        nn.Linear, nn.ReLU, nn.Identity, nn.Dropout,
    ))
