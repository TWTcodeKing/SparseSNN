#!/bin/bash
# Train MS-ResNet CIFAR variants on CIFAR-100
#
# Usage:
#   bash scripts/train_msresnet_cifar100.sh [depth] [gpu]
#
# Examples:
#   bash scripts/train_msresnet_cifar100.sh 110 0    # depth=110 on GPU 0
#   bash scripts/train_msresnet_cifar100.sh 56 1     # depth=56 on GPU 1
#   bash scripts/train_msresnet_cifar100.sh all 0     # train all depths sequentially
#
# Paper reported accuracy (Table VI):
#   depth 20: ~59%  |  depth 32: 61.35%  |  depth 44: 63.84%
#   depth 56: 65.24%  |  depth 110: 66.83%

DEPTH=${1:-110}
GPU=${2:-0}
DATA_ROOT="/home/twt/datasets/"
RECIPE="configs/ms_resnet/recipes/cifar100_cifar_arch.yaml"

train_one() {
    local depth=$1
    local gpu=$2
    echo "=========================================="
    echo "Training ms_resnet_cifar${depth} on CIFAR-100 (GPU ${gpu})"
    echo "=========================================="
    uv run tengine/train.py \
        --model ms_resnet_cifar${depth} \
        --recipe ${RECIPE} \
        --dataset cifar100 \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${gpu}
}

if [ "$DEPTH" = "all" ]; then
    for d in 20 32 44 56 110; do
        train_one $d $GPU
    done
else
    train_one $DEPTH $GPU
fi
