#!/bin/bash
# Transfer-learn SpikingResformer on CIFAR-100
#
# Usage:
#   bash scripts/train_spikingresformer_cifar100.sh [variant] [gpu]
#
# Examples:
#   bash scripts/train_spikingresformer_cifar100.sh ti 0    # finetune Ti on GPU 0
#   bash scripts/train_spikingresformer_cifar100.sh s 1     # finetune S on GPU 1
#
# Original repo config: 150 epochs, lr=1e-4, cosine→1e-5, AdamW wd=0.01,
#   img_size=128, rand-m9-n1-mstd0.4-inc1, Mixup, label_smoothing=0.1

VARIANT=${1:-ti}
GPU=${2:-0}
DATA_ROOT="/home/twt/datasets/"
CKPT="checkpoints/spikingresformer/ImageNet_spikingresformer_${VARIANT}.pth"
RECIPE="configs/spikingresformer/recipes/cifar100.yaml"

echo "=========================================="
echo "Transfer SpikingResformer-${VARIANT} → CIFAR-100 (GPU ${GPU})"
echo "=========================================="
uv run tengine/transfer.py \
    --config configs/spikingresformer/spikingresformer_${VARIANT}.yaml \
    --pretrained ${CKPT} \
    --recipe ${RECIPE} \
    --dataset cifar100 \
    --data-root ${DATA_ROOT} \
    --gpu-ids ${GPU}
