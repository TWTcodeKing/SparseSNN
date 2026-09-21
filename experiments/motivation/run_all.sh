#!/bin/bash
# Motivation experiment: profile TRT on Spikformer-1-512 (CIFAR-100)
# Usage: bash experiments/motivation/run_all.sh [gpu_id]
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
OUT=experiments/motivation/output
PYTHON=/home/twt/SparseSNN/.venv/bin/python
NSYS=/usr/local/cuda-12.8/bin/nsys
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
ENGINE=$OUT/spikformer_1_512.engine

echo "============================================"
echo "  Motivation: TRT on Spikformer-1-512"
echo "  GPU: $GPU_ID"
echo "============================================"

mkdir -p $OUT

# Step 1: Export ONNX + Build TRT
echo -e "\n>>> Step 1: Export ONNX + Build TRT engine"
$PYTHON experiments/motivation/prepare_spikformer.py

# Step 2: Latency benchmark
echo -e "\n>>> Step 2: Latency benchmark"
$PYTHON experiments/motivation/bench_latency.py --engine $ENGINE --batches 1 2 4 8

# Step 3: nsys profiling
echo -e "\n>>> Step 3: nsys profiling"
for B in 1 2 4 8; do
    echo "  nsys: batch=$B"
    sudo $NSYS profile \
        --capture-range=cudaProfilerApi --stats=true --force-overwrite=true \
        -o $OUT/nsys_spk_b${B} \
        $PYTHON experiments/motivation/profile_infer.py --engine $ENGINE --batch $B \
            --warmup 10 --iters 20 \
        2>&1 | tee $OUT/nsys_spk_b${B}_stats.txt
done

# Step 4: ncu profiling
echo -e "\n>>> Step 4: ncu profiling"
for B in 1 8; do
    echo "  ncu: batch=$B"
    sudo $NCU --set basic --csv --launch-count 500 --target-processes all \
        $PYTHON experiments/motivation/profile_infer.py --engine $ENGINE --batch $B \
            --warmup 0 --iters 1 \
        > $OUT/ncu_spk_b${B}.csv 2>&1
done

# Step 5: Parse results
echo -e "\n>>> Step 5: Analysis"
$PYTHON experiments/motivation/parse_ncu.py $OUT/ncu_spk_b1.csv 1
$PYTHON experiments/motivation/parse_ncu.py $OUT/ncu_spk_b8.csv 8

echo -e "\n============================================"
echo "  Done. Results in $OUT/"
echo "============================================"
