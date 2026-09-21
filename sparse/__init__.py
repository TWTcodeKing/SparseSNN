"""Post-training N:M sparsification for SNNs (SBC).

Submodules:
    pruning   canonical N:M structured pruning primitives (magnitude baselines)
    sbc       SBC core: Van Rossum Distance matrix, SMP Hessian, ExactOBS N:M pruning
    snn_sbc   end-to-end pipeline: Hessian collection, pruning, BN recalibration,
              optional channel permutation and KD fine-tuning (``python -m sparse.snn_sbc``)
    utils     neuron detection and weight reshaping helpers
"""
