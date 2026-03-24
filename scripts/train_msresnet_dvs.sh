#!/bin/bash
# Train MS-ResNet on DVS datasets (multi-GPU supported)
#
# Usage:
#   bash scripts/train_msresnet_dvs.sh [dataset] [gpus]
#
# Examples:
#   bash scripts/train_msresnet_dvs.sh cifar10dvs 0,1        # 2 GPUs
#   bash scripts/train_msresnet_dvs.sh cifar10dvs 0           # single GPU
#
# Model: ms_resnet_dvs20 (ResNet-20, stem_stride=2, 0.27M params)
# Paper: 75.56% on CIFAR10-DVS, T=20, 1024 epochs

DATASET=${1:-cifar10dvs}
GPUS=${2:-0,1}

if [ "$DATASET" = "cifar10dvs" ]; then
    DATA_ROOT="/home/twt/datasets/cifar10-dvs"
    RECIPE="configs/ms_resnet/recipes/cifar10dvs.yaml"
else
    echo "Unknown dataset: $DATASET (paper only evaluates on cifar10dvs)"
    exit 1
fi

NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Training ms_resnet_dvs20 on ${DATASET} (GPUs: ${GPUS}, nproc=${NPROC})"
echo "=========================================="

if [ "$NPROC" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} \
        tengine/train.py \
        --model ms_resnet_dvs20 \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS}
else
    uv run tengine/train.py \
        --model ms_resnet_dvs20 \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS}
fi
