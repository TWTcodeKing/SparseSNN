"""Sparsification techniques for SNN models.

Submodules:
    pruning      — canonical N:M structured pruning primitives
    st_train     — SR-STE regularized training toward N:M sparsity
    permutation  — activation-aware channel permutation for N:M alignment
    comp_2_4     — dense-sparse weight factorization (W = W_24 + W_residual)
    fr_prune     — neuron-level firing-rate-aware N:M pruning (spatial/token granularity)
    convert      — element-wise N:M compensation (S2SC: Spike-to-Structured-Sparse Conversion)

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
from sparse.fr_prune import (
    prune_n_m_neuron_aware,
    apply_neuron_aware_pruning,
)
from sparse.convert import (
    compensate_n_m_pruning,
    apply_compensation,
)
