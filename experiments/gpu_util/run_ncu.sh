#!/bin/bash
# Phase 2: Run ncu profiling for all model × batch × backend combinations.
#
# Usage:
#   sudo bash experiments/gpu_util/run_ncu.sh [gpu_id]
#
# Requires sudo for ncu access to GPU performance counters.
# Estimated runtime: 4-8 hours total (16 runs × 15-30 min each).

set -e

GPU_ID=${1:-3}
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
PYTHON=.venv/bin/python
REPORT_DIR=experiments/gpu_util/ncu_reports

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

mkdir -p "$REPORT_DIR"

# ─── ncu metrics (one comma-separated string) ───
METRICS="\
gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor.sum,\
smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_alu.sum,\
smsp__inst_executed_pipe_lsu.sum,\
launch__occupancy_per_register_count,\
launch__block_size,\
launch__grid_size,\
sm__ctas_launched.sum"

# Warmup config: 10 warmup iters in the runner script.
# ncu --launch-skip skips warmup kernel launches.
# We use generous counts to capture at least one full inference pass.
WARMUP_ITERS=10
MEASURE_ITERS=1

MODELS=(maxformer_10_512 sew_resnet101 spikingresformer_m spikformer_4_512)
BATCHES=(16 32)

total=$((${#MODELS[@]} * ${#BATCHES[@]} * 2))
count=0

for model in "${MODELS[@]}"; do
    for batch in "${BATCHES[@]}"; do
        # ─── sengine profiling ───
        count=$((count + 1))
        report="${REPORT_DIR}/${model}_B${batch}_sengine"
        echo ""
        echo "================================================================"
        echo "  [$count/$total] ncu: sengine | $model | B=$batch | GPU=$GPU_ID"
        echo "================================================================"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
        $NCU \
            --target-processes all \
            --profile-from-start off \
            --metrics "$METRICS" \
            -o "$report" \
            --force-overwrite \
            $PYTHON experiments/gpu_util/run_sengine_profile.py \
                --model "$model" --batch "$batch" \
                --warmup $WARMUP_ITERS --iters $MEASURE_ITERS

        echo "  Report: ${report}.ncu-rep"

        # ─── TRT profiling ───
        count=$((count + 1))
        report="${REPORT_DIR}/${model}_B${batch}_trt"
        echo ""
        echo "================================================================"
        echo "  [$count/$total] ncu: TRT | $model | B=$batch | GPU=$GPU_ID"
        echo "================================================================"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
        $NCU \
            --target-processes all \
            --profile-from-start off \
            --metrics "$METRICS" \
            -o "$report" \
            --force-overwrite \
            $PYTHON experiments/gpu_util/run_trt_profile.py \
                --model "$model" --batch "$batch" \
                --warmup $WARMUP_ITERS --iters $MEASURE_ITERS

        echo "  Report: ${report}.ncu-rep"
    done
done

echo ""
echo "================================================================"
echo "  All $total profiling runs completed."
echo "  Reports in: $REPORT_DIR/"
echo "  Next: python experiments/gpu_util/parse_ncu.py"
echo "================================================================"
