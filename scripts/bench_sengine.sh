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
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"

T=4
WARMUP=200
ITERS=1000
EXPORT_DIR="sengine/exports"

# ── ResNet models ──
python scripts/bench_sengine_latency.py \
    --model sew_resnet18 --dataset cifar100 \
    --T $T --batch-sizes 1,4,8,16,32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR

python scripts/bench_sengine_latency.py \
    --model sew_resnet34 --dataset cifar100 \
    --T $T --batch-sizes 1,4,8,16,32 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR

# ── Transformer models ──
python scripts/bench_sengine_latency.py \
    --config configs/spikformer/spikformer_8_384.yaml --dataset cifar100 \
    --T $T --batch-sizes 1,4,8,16 \
    --warmup $WARMUP --iters $ITERS --export-dir $EXPORT_DIR
