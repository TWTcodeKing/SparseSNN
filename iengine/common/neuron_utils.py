"""Shared utility for fused Triton neuron replacement across all iengine backends.

Provides a single entry point that all inference modules can call to optionally
replace MultiStepLIF/IF neuron forward methods with fused Triton kernels.
"""

import torch.nn as nn


def maybe_fuse_neurons(model: nn.Module, fuse: bool = True, verbose: bool = True) -> int:
    """Conditionally replace neuron forward methods with fused Triton kernels.

    Safe to call on any model — returns 0 if fuse=False or if Triton is
    unavailable.

    Args:
        model: SNN model containing MultiStepLIFNeuron / MultiStepIFNeuron.
        fuse: If True, apply fused Triton neuron replacement.
        verbose: Print replacement details.

    Returns:
        Number of neurons replaced.
    """
    if not fuse:
        return 0

    try:
        from iengine.triton_sparse.neuron_kernel import replace_neuron_forward
    except ImportError as e:
        if verbose:
            print(f"  [Triton neurons] Not available: {e}")
        return 0

    n = replace_neuron_forward(model, verbose=verbose)
    return n
