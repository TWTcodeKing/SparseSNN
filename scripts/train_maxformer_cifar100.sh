#!/bin/bash
# Train MaxFormer on CIFAR-100
#
# Usage:
#   bash scripts/train_maxformer_cifar100.sh [gpu]
#
# Config: maxformer_cifar (embed=384, depths=4, EmbedOrig stem)
# Recipe: AdamW lr=1.5e-3, 400 epochs, Mixup=0.75, CutMix=0.5, T=4
# Reported: ~82.65%

GPU=${1:-0}
DATA_ROOT="${DATA_ROOT:-/data/twt/datasets}"

echo "=========================================="
echo "Training MaxFormer-CIFAR on CIFAR-100 (GPU ${GPU})"
echo "=========================================="
uv run tengine/train.py \
    --config configs/maxformer/maxformer_cifar.yaml \
    --recipe configs/maxformer/recipes/cifar100.yaml \
    --dataset cifar100 \
    --data-root ${DATA_ROOT} \
    --gpu-ids ${GPU}
