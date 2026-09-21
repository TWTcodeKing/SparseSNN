#!/bin/bash
# Profile ONE model + batch + backend with ncu.
#
# Usage:
#   # Build + profile sengine for maxformer B=16 on GPU 3
#   sudo bash experiments/gpu_util/profile_one.sh maxformer_10_512 16 sengine 3
#
#   # Profile TRT only (assumes engine already built)
#   sudo bash experiments/gpu_util/profile_one.sh sew_resnet101 32 trt 3
#
#   # Profile both backends
#   sudo bash experiments/gpu_util/profile_one.sh spikformer_4_512 16 both 3
#
# Arguments:
#   $1 = model key (maxformer_10_512 | sew_resnet101 | spikingresformer_m | spikformer_4_512)
#   $2 = batch size (16 | 32)
#   $3 = backend (sengine | trt | both)
#   $4 = GPU ID (default: 3)

set -e

MODEL=${1:?Usage: $0 <model> <batch> <backend> [gpu_id]}
BATCH=${2:?Usage: $0 <model> <batch> <backend> [gpu_id]}
BACKEND=${3:?Usage: $0 <model> <batch> <backend> [gpu_id]}
GPU_ID=${4:-3}

NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
PYTHON=.venv/bin/python
REPORT_DIR=experiments/gpu_util/ncu_reports

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

mkdir -p "$REPORT_DIR"

# ─── ncu metrics ───
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

WARMUP=10
ITERS=1

profile_backend() {
    local be=$1
    local report="${REPORT_DIR}/${MODEL}_B${BATCH}_${be}"

    echo ""
    echo "================================================================"
    echo "  ncu profiling: $be | $MODEL | B=$BATCH | GPU=$GPU_ID"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    $NCU \
        --target-processes all \
        --profile-from-start off \
        --metrics "$METRICS" \
        -o "$report" \
        --force-overwrite \
        $PYTHON experiments/gpu_util/run_${be}_profile.py \
            --model "$MODEL" --batch "$BATCH" \
            --warmup $WARMUP --iters $ITERS

    echo ""
    echo "  Report saved: ${report}.ncu-rep"
    echo ""

    # Quick summary: extract kernel count
    $NCU --import "${report}.ncu-rep" --csv --page raw 2>/dev/null | wc -l | \
        xargs -I{} echo "  Kernel rows in CSV: {}"
}

# ─── Main ───
echo "Model:   $MODEL"
echo "Batch:   $BATCH"
echo "Backend: $BACKEND"
echo "GPU:     $GPU_ID"

if [ "$BACKEND" = "sengine" ] || [ "$BACKEND" = "both" ]; then
    profile_backend "sengine"
fi

if [ "$BACKEND" = "trt" ] || [ "$BACKEND" = "both" ]; then
    profile_backend "trt"
fi

echo ""
echo "================================================================"
echo "  Done. Parse with: python experiments/gpu_util/parse_ncu.py"
echo "================================================================"
