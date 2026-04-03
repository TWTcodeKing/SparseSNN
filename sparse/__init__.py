"""Sparsification techniques for SNN models.

Submodules:
    pruning      — canonical N:M structured pruning primitives
    OBS          — SparseGPT-style OBS for N:M pruning (baseline)
    OBC          — Optimal Brain Compression: block OBS with hardware constraint in objective
    sbc          — SBC: second-order post-training pruning with SMP Hessian
    snn_sbc      — SNN-specific SBC pipeline: module-wise compression with SMP Hessian

Shared utilities in sparse.utils:
    Neuron param accessors, firing rate / membrane potential collectors.

Profiling utilities are in utils.profiling and re-exported here for convenience.
"""

from utils.profiling import (
    NeuronFiringRateProfiler,
    ChannelFiringRateProfiler,
    compute_effective_rates_conv,
    compute_enhanced_rates_linear,
    profile_neuron_firing_rates,
    profile_model_firing_rates,
)
