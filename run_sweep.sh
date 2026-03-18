#!/bin/bash
# Hyperparameter sweep for neuron-aware N:M pruning
# Usage: bash sparse/run_sweep.sh

python -m sparse.sweep_fr_prune \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001_normal/best.pth \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets \
    --profile-batches 60 --max-spatial 16 \
    --lams 0.0,0.3,0.5,0.7,1.0,1.5,2.0 \
    --alphas 0.0,0.3,0.5,0.7,1.0 \
    --scorings multiplicative,additive \
    --output-dir sweep_results
