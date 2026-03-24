#!/bin/bash
# Transfer-learn SpikingResformer on DVS datasets
#
# Usage:
#   bash scripts/train_spikingresformer_dvs.sh [dataset] [variant] [gpu]
#
# Examples:
#   bash scripts/train_spikingresformer_dvs.sh cifar10dvs ti 0
#   bash scripts/train_spikingresformer_dvs.sh dvs128gesture s 1
#
# Paper: finetune ImageNet pretrained model, T=10, img_size=128,
#        same settings as CIFAR-100 transfer + neuromorphic data augmentation
# Note: transfer.py is single-GPU. For multi-GPU, adapt to train.py with DDP.

DATASET=${1:-cifar10dvs}
VARIANT=${2:-ti}
GPU=${3:-0}

if [ "$DATASET" = "cifar10dvs" ]; then
    DATA_ROOT="/home/twt/datasets/cifar10-dvs"
    RECIPE="configs/spikingresformer/recipes/cifar10dvs.yaml"
elif [ "$DATASET" = "dvs128gesture" ]; then
    DATA_ROOT="/home/twt/datasets/dvs128gesture"
    RECIPE="configs/spikingresformer/recipes/dvs128gesture.yaml"
else
    echo "Unknown dataset: $DATASET (use cifar10dvs or dvs128gesture)"
    exit 1
fi

CKPT="checkpoints/spikingresformer/ImageNet_spikingresformer_${VARIANT}.pth"

echo "=========================================="
echo "Transfer SpikingResformer-${VARIANT} → ${DATASET} (GPU ${GPU})"
echo "=========================================="
uv run tengine/transfer.py \
    --config configs/spikingresformer/spikingresformer_${VARIANT}.yaml \
    --pretrained ${CKPT} \
    --recipe ${RECIPE} \
    --dataset ${DATASET} \
    --data-root ${DATA_ROOT} \
    --gpu-ids ${GPU}
