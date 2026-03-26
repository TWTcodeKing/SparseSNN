#!/bin/bash
# Train MetaSpikeFormer on CIFAR-100
#
# Usage:
#   bash scripts/train_metaformer_cifar100.sh [config] [gpu]
#
# Examples:
#   bash scripts/train_metaformer_cifar100.sh 384 0    # embed_dim=[96,192,384,480]
#   bash scripts/train_metaformer_cifar100.sh 512 1    # embed_dim=[128,256,512,640]
#
# Uses CIFAR-adapted architecture (reduced downsampling: 32→16→8, 64 tokens).
# No official CIFAR-100 recipe exists — this uses our best-effort adaptation.

CONFIG=${1:-384}
GPU=${2:-0}
DATA_ROOT="/home/twt/datasets/"

echo "=========================================="
echo "Training MetaSpikeFormer-${CONFIG} on CIFAR-100 (GPU ${GPU})"
echo "=========================================="
uv run tengine/train.py \
    --config configs/metaformer/meta_spikformer_8_${CONFIG}_cifar.yaml \
    --recipe configs/metaformer/recipes/cifar100.yaml \
    --dataset cifar100 \
    --data-root ${DATA_ROOT} \
    --gpu-ids ${GPU}
