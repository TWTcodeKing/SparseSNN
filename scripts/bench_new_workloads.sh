#!/bin/bash
# sengine vs TensorRT vs torch.compile(Inductor) on the two new workloads
# (SNN-VGG16 / UT-HAR, SNN-VGG9 / UrbanSound8K), B = 4,8,16,32, T = 4.
# All runs go to ONE GPU sequentially so the three tools see the same device state.
#
# Usage: bash scripts/bench_new_workloads.sh [gpu_id]      # logs -> output/bench_new_workloads/
set -u
GPU_ID=${1:-1}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=$GPU_ID
PY=.venv/bin/python
OUT=output/bench_new_workloads
BS=4,8,16,32
T=4
mkdir -p $OUT

declare -A CKPT=( [snn_vgg16]=output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth
                  [snn_vgg9]=output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth )
declare -A DS=( [snn_vgg16]=ut_har [snn_vgg9]=urbansound8k )

run() {  # run <log-name> <cmd...>
    local log=$OUT/$1.log; shift
    echo "[$(date +%H:%M:%S)] START $log"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU_ID > $log
    "$@" >> $log 2>&1; echo "[$(date +%H:%M:%S)] END   $log (exit $?)"
}

for M in snn_vgg16 snn_vgg9; do
    D=${DS[$M]}; C=${CKPT[$M]}
    run sengine_fp16_${M} $PY scripts/bench_sengine_latency.py --model $M --dataset $D --T $T \
        --batch-sizes $BS --fusion slicer --autotune --precision fp16 --checkpoint $C --gpu-ids 0
done
for M in snn_vgg16 snn_vgg9; do
    D=${DS[$M]}; C=${CKPT[$M]}
    run trt_fp16_${M} $PY scripts/bench_trt_latency.py --model $M --dataset $D --T $T --batch-sizes $BS --fp16 --checkpoint $C --gpu-ids 0
    run trt_fp32_${M} $PY scripts/bench_trt_latency.py --model $M --dataset $D --T $T --batch-sizes $BS --checkpoint $C --gpu-ids 0
done
for M in snn_vgg16 snn_vgg9; do
    D=${DS[$M]}; C=${CKPT[$M]}
    run inductor_fp32_${M} $PY scripts/bench_inductor_latency.py --model $M --dataset $D --T $T --batch-sizes $BS --backend inductor --mode reduce-overhead --checkpoint $C --gpu-ids 0
    run inductor_fp16_${M} $PY scripts/bench_inductor_latency.py --model $M --dataset $D --T $T --batch-sizes $BS --backend inductor --mode reduce-overhead --fp16 --checkpoint $C --gpu-ids 0
done
echo "[$(date +%H:%M:%S)] ALL DONE"
