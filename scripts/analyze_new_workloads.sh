#!/bin/bash
# Follow-up analysis for bench_new_workloads.sh: TRT nsys traces, sengine per-kernel breakdown
# vs TRT, and a real TensorRT FP32 run (separate engine dir).  Usage: bash scripts/analyze_new_workloads.sh [gpu_id]
set -u
GPU_ID=${1:-1}
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH; export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=$GPU_ID
PY=.venv/bin/python; OUT=output/bench_new_workloads; mkdir -p $OUT profiles
run() { local log=$OUT/$1.log; shift; echo "[$(date +%H:%M:%S)] START $log"; "$@" > $log 2>&1; echo "[$(date +%H:%M:%S)] END   $log (exit $?)"; }
for spec in "snn_vgg16 ut_har 4" "snn_vgg16 ut_har 32" "snn_vgg9 urbansound8k 4" "snn_vgg9 urbansound8k 16"; do
    set -- $spec; M=$1; D=$2; B=$3
    run nsys_trt_${M}_b${B} nsys profile -o profiles/trt_${M}_b${B} --trace=cuda,nvtx --force-overwrite=true \
        $PY utils/profile_trt_engine.py trt_engines/${M}_b${B}.engine --warmup 10 --iters 10
    run kernels_${M}_b${B} $PY scripts/analyze_kernel_latency.py --onnx sengine/exports/${M}_${D}_plugin.onnx --T 4 --batch $B \
        --trt-nsys profiles/trt_${M}_b${B}.nsys-rep
done
run trt_fp32real_snn_vgg16 $PY scripts/bench_trt_latency.py --model snn_vgg16 --dataset ut_har --T 4 --batch-sizes 4,8,16,32 \
    --checkpoint output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth --engine-dir trt_engines_fp32 --gpu-ids 0
run trt_fp32real_snn_vgg9 $PY scripts/bench_trt_latency.py --model snn_vgg9 --dataset urbansound8k --T 4 --batch-sizes 4,8,16,32 \
    --checkpoint output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth --engine-dir trt_engines_fp32 --gpu-ids 0
echo "[$(date +%H:%M:%S)] ALL DONE"
