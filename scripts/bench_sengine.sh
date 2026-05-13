#!/bin/bash
# Benchmark sengine inference latency (C++ executor) for SNN models.
#
# Usage:
#   bash scripts/bench_sengine.sh [gpu_id]
#
# Edit the MODEL / CONFIG / DATASET / BATCH_SIZES variables below
# to benchmark different models.

set -e

GPU_ID=${1:-0}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"

T=4
WARMUP=200
ITERS=1000
EXPORT_DIR="sengine/exports"

# ── ResNet models ──
# python scripts/bench_sengine_latency.py \
#     --model sew_resnet101 --dataset imagenet \
#     --T $T --batch-sizes 4,8,16,32 --fusion slicer --autotune --precision fp32 \
#     --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID
python scripts/bench_sengine_latency.py \
    --config ./configs/maxformer/maxformer_10_512.yaml --dataset imagenet \
    --T $T --batch-sizes 4,8,16,32 --fusion slicer --autotune --precision fp32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID

    
python scripts/bench_sengine_latency.py \
    --config ./configs/spikformer/spikformer_4_512.yaml --dataset imagenet \
    --T $T --batch-sizes 1,2,4,8 --fusion slicer --autotune --precision fp32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID





python scripts/bench_sengine_latency.py \
 --model snn_vgg9 --dataset cifar10dvs \
 --T 16 --batch-sizes 16,32 --fusion slicer --autotune --precision fp32 \
 --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID

python scripts/bench_sengine_latency.py \
 --config ./configs/maxformer/maxformer_dvs.yaml --dataset cifar10dvs \
 --T 16 --batch-sizes 32 --fusion slicer --autotune --precision fp32 \
 --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID


python scripts/bench_sengine_latency.py \
 --config ./configs/maxformer/ms_qkformer_dvs.yaml --dataset cifar10dvs \
 --T 16 --batch-sizes 32 --fusion slicer --autotune --precision fp32 \
 --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID


python scripts/bench_sengine_latency.py \
    --config ./configs/spikingresformer/spikingresformer_m.yaml --dataset imagenet \
    --T 4 --batch-sizes 4,8,16,32 --fusion slicer --autotune \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID

python scripts/bench_sengine_latency.py \
    --config ./configs/spikingresformer/spikingresformer_s.yaml --dataset cifar10dvs \
    --T 16 --batch-sizes 4,8,16,32 --fusion slicer --autotune \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID


python scripts/bench_sengine_latency.py \
    --config ./configs/spikingresformer/spikingresformer_m.yaml --dataset imagenet \
    --T 4 --batch-sizes 4,8,16,32 --fusion slicer --autotune --precision fp32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID

python scripts/bench_sengine_latency.py \
    --config ./configs/spikingresformer/spikingresformer_s.yaml --dataset cifar10dvs \
    --T 16 --batch-sizes 4,8,16,32 --fusion slicer --autotune --precision fp32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR --gpu-ids $GPU_ID

