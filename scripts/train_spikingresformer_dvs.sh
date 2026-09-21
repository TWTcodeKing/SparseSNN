#!/bin/bash
# Transfer-learn SpikingResformer on DVS datasets (multi-GPU supported)
#
# Usage:
#   bash scripts/train_spikingresformer_dvs.sh [dataset] [variant] [gpus]
#
# Examples:
#   bash scripts/train_spikingresformer_dvs.sh cifar10dvs ti 0,1     # 2 GPUs
#   bash scripts/train_spikingresformer_dvs.sh dvs128gesture s 0     # single GPU
#
# Paper: finetune ImageNet pretrained model, T=10, img_size=128

DATASET=${1:-cifar10dvs}
VARIANT=${2:-ti}
GPUS=${3:-0,1}

if [ "$DATASET" = "cifar10dvs" ]; then
    DATA_ROOT="${DATA_ROOT:-/data/twt/datasets}/cifar10-dvs"
    RECIPE="configs/spikingresformer/recipes/cifar10dvs.yaml"
elif [ "$DATASET" = "dvs128gesture" ]; then
    DATA_ROOT="${DATA_ROOT:-/data/twt/datasets}/dvs128gesture"
    RECIPE="configs/spikingresformer/recipes/dvs128gesture.yaml"
else
    echo "Unknown dataset: $DATASET (use cifar10dvs or dvs128gesture)"
    exit 1
fi

CKPT="checkpoints/spikingresformer/ImageNet_spikingresformer_${VARIANT}.pth"
NPROC=$(echo $GPUS | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Transfer SpikingResformer-${VARIANT} → ${DATASET} (GPUs: ${GPUS}, nproc=${NPROC})"
echo "=========================================="

if [ "$NPROC" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} \
        tengine/transfer.py \
        --config configs/spikingresformer/spikingresformer_${VARIANT}.yaml \
        --pretrained ${CKPT} \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS} \
        --batch-size 64
else
    uv run tengine/transfer.py \
        --config configs/spikingresformer/spikingresformer_${VARIANT}.yaml \
        --pretrained ${CKPT} \
        --recipe ${RECIPE} \
        --dataset ${DATASET} \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${GPUS}
fi
