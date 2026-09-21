#!/bin/bash
# ONNX Runtime (CUDA EP) and torch.compile max-autotune baselines on the UT-HAR / UrbanSound8K
# workloads, B=4..32, fp16 + fp32, interleaved per batch on one idle GPU.
# Usage: bash scripts/bench_extra_baselines.sh [gpu_id]   -> output/bench_new_workloads/extra_gpu<id>.log
set -u
GPU_ID=${1:-6}
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH; export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=$GPU_ID
PY=.venv/bin/python; OUT=output/bench_new_workloads; LOG=$OUT/extra_gpu${GPU_ID}.log; : > $LOG
declare -A DS=( [snn_vgg16]=ut_har [snn_vgg9]=urbansound8k )
declare -A CKPT=( [snn_vgg16]=output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth [snn_vgg9]=output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth )
util() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i $GPU_ID; }
for B in 4 8 16 32; do
  for M in snn_vgg16 snn_vgg9; do
    D=${DS[$M]}; C=${CKPT[$M]}
    echo "[$(date +%H:%M:%S)] B=$B $M gpu-before: $(util)" >> $LOG
    for P in fp32 fp16; do
      F=""; [ $P = fp16 ] && F="--fp16"
      $PY scripts/bench_onnxruntime_latency.py --model $M --dataset $D --T 4 --batch-sizes $B $F --checkpoint $C --gpu-ids 0 > $OUT/ort_${P}_${M}_b${B}.log 2>&1
      grep -E "^\s+B=$B\s+[0-9.]+ms" $OUT/ort_${P}_${M}_b${B}.log | sed "s/^/RESULT ort_$P $M /" >> $LOG || echo "RESULT ort_$P $M B=$B FAILED" >> $LOG
      $PY scripts/bench_inductor_latency.py --model $M --dataset $D --T 4 --batch-sizes $B --backend inductor --mode max-autotune $F --checkpoint $C --gpu-ids 0 > $OUT/inductor_maxautotune_${P}_${M}_b${B}.log 2>&1
      grep -E "^\s+B=$B\s+[0-9.]+ms" $OUT/inductor_maxautotune_${P}_${M}_b${B}.log | sed "s/^/RESULT inductor_maxautotune_$P $M /" >> $LOG || echo "RESULT inductor_maxautotune_$P $M B=$B FAILED" >> $LOG
    done
  done
done
echo "[$(date +%H:%M:%S)] ALL DONE" >> $LOG
