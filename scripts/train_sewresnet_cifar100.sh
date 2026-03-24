#!/bin/bash
# Train SEW-ResNet CIFAR variants on CIFAR-100
#
# Usage:
#   bash scripts/train_sewresnet_cifar100.sh [depth] [gpu] [connect_f]
#
# Examples:
#   bash scripts/train_sewresnet_cifar100.sh 110 0          # depth=110, GPU 0, ADD
#   bash scripts/train_sewresnet_cifar100.sh 56 1 ADD       # depth=56, GPU 1, ADD
#   bash scripts/train_sewresnet_cifar100.sh all 0          # train all depths sequentially
#
# Note: neuron_type defaults to IF (hardcoded in SEWResNetCifar constructor).
# To change, modify the model constructor default or pass via code.

DEPTH=${1:-110}
GPU=${2:-0}
CONNECT_F=${3:-ADD}
DATA_ROOT="/home/twt/datasets/"
RECIPE="configs/sew_resnet/recipes/cifar100.yaml"

train_one() {
    local depth=$1
    local gpu=$2
    echo "=========================================="
    echo "Training sew_resnet_cifar${depth} on CIFAR-100 (GPU ${gpu}, ${CONNECT_F})"
    echo "=========================================="
    uv run tengine/train.py \
        --model sew_resnet_cifar${depth} \
        --recipe ${RECIPE} \
        --dataset cifar100 \
        --data-root ${DATA_ROOT} \
        --gpu-ids ${gpu} \
        --connect-f ${CONNECT_F}
}

if [ "$DEPTH" = "all" ]; then
    for d in 20 32 44 56 110; do
        train_one $d $GPU
    done
else
    train_one $DEPTH $GPU
fi
