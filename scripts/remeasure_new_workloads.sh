#!/bin/bash
# Clean back-to-back re-measurement (interleaved per batch size) of sengine fp16 (cached
# kernels + validated fusion), TensorRT fp16 (prebuilt engines) and Inductor fp16 on one GPU,
# followed by nsys per-kernel traces of the sengine CUDA graphs.
# Usage: bash scripts/remeasure_new_workloads.sh [gpu_id]
set -u
GPU_ID=${1:-1}
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH; export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=$GPU_ID
PY=.venv/bin/python; OUT=output/bench_new_workloads; SCR=/tmp/claude-1024/-home-twt-SparseSNN/af89380d-d85e-4c5c-9416-b9d49f1fd655/scratchpad
LOG=$OUT/${REMEASURE_LOG:-remeasure.log}; : > $LOG
declare -A DS=( [snn_vgg16]=ut_har [snn_vgg9]=urbansound8k )
declare -A CKPT=( [snn_vgg16]=output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth [snn_vgg9]=output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth )
util() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i $GPU_ID; }
for B in 4 8 16 32; do
  for M in snn_vgg16 snn_vgg9; do
    D=${DS[$M]}
    echo "[$(date +%H:%M:%S)] B=$B $M gpu-before: $(util)" >> $LOG
    $PY $SCR/sengine_bench.py sengine/exports/${M}_${D}_plugin.onnx $B 2>&1 | grep RESULT >> $LOG
    $PY scripts/bench_trt_latency.py --model $M --dataset $D --T 4 --batch-sizes $B --fp16 --checkpoint ${CKPT[$M]} --gpu-ids 0 2>&1 \
        | grep -E "^\s+B=$B " | sed "s/^/RESULT trt_fp16 $M /" >> $LOG
    $PY scripts/bench_inductor_latency.py --model $M --dataset $D --T 4 --batch-sizes $B --backend inductor --mode reduce-overhead --fp16 --checkpoint ${CKPT[$M]} --gpu-ids 0 2>&1 \
        | grep -E "^\s+B=$B\s+[0-9.]+ms" | sed "s/^/RESULT inductor_fp16 $M /" >> $LOG
  done
done
for spec in $( [ "${NO_NSYS:-0}" = "1" ] && echo "" || echo "snn_vgg16_ut_har:4 snn_vgg16_ut_har:32 snn_vgg9_urbansound8k:4 snn_vgg9_urbansound8k:16" ); do set -- ${spec//:/ }
  nsys profile -o profiles/sengine_$1_b$2 --trace=cuda --cuda-graph-trace=node --force-overwrite=true $PY $SCR/sengine_bench.py sengine/exports/$1_plugin.onnx $2 20 50 > $OUT/nsys_sengine_$1_b$2.log 2>&1
done
echo "[$(date +%H:%M:%S)] ALL DONE" >> $LOG
