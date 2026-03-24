#!/bin/bash
# Train MS_QKFormer on CIFAR-100
#
# Usage:
#   bash scripts/train_ms_qkformer_cifar100.sh [gpu]
#
# Config: ms_qkformer_cifar (embed=384, depths=4, EmbedOrig stem, QKA+SSA)
# Recipe: AdamW lr=1.5e-3, 400 epochs, Mixup=0.75, CutMix=0.5, T=4

GPU=${1:-0}
DATA_ROOT="/home/twt/datasets/"

echo "=========================================="
echo "Training MS_QKFormer-CIFAR on CIFAR-100 (GPU ${GPU})"
echo "=========================================="
uv run tengine/train.py \
    --config configs/maxformer/ms_qkformer_cifar.yaml \
    --recipe configs/maxformer/recipes/cifar100.yaml \
    --dataset cifar100 \
    --data-root ${DATA_ROOT} \
    --gpu-ids ${GPU}
