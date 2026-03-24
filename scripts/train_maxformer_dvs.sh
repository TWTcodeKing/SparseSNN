#!/bin/bash
# Train MaxFormer on DVS datasets (from scratch, multi-GPU)
#
# Usage:
#   bash scripts/train_maxformer_dvs.sh [dataset] [gpus]
#
# Examples:
#   bash scripts/train_maxformer_dvs.sh cifar10dvs 0,1         # 2 GPUs
#   bash scripts/train_maxformer_dvs.sh dvs128gesture 0,1,2,3  # 4 GPUs
#   bash scripts/train_maxformer_dvs.sh cifar10dvs 0            # single GPU
#
# Trained from scratch. Reported: CIFAR10-DVS 84.2%, DVS Gesture 98.6%

DATASET=${1:-cifar10dvs}
GPUS=${2:-0,1}

if [ "$DATASET" = "cifar10dvs" ]; then
    DATA_ROOT="/home/twt/datasets/cifar10-dvs"
    RECIPE="configs/maxformer/recipes/cifar10dvs.yaml"
elif [ "$DATASET" = "dvs128gesture" ]; then
    DATA_ROOT="/home/twt/datasets/dvs128gesture"
    RECIPE="configs/maxformer/recipes/dvs128gesture.yaml"
else
    echo "Unknown dataset: $DATASET (use cifar10dvs or dvs128gesture)"
    exit 1
fi

NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Training MaxFormer-DVS on ${DATASET} (GPUs: ${GPUS}, nproc=${NPROC})"
echo "=========================================="

if [ "$NPROC" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} \
        tengine/train.py \
        --config configs/maxformer/maxformer_dvs.yaml \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS}
else
    uv run tengine/train.py \
        --config configs/maxformer/maxformer_dvs.yaml \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS}
fi
