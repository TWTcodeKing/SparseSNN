"""Shared constants for GPU utilization profiling experiments."""

import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Experiment Matrix ───

T = 4
BATCH_SIZES = [2, 8, 16, 32]
PRECISION = "fp16"
DATASET = "imagenet"
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 1000

# model_key → (type, spec, plugin_onnx_basename)
# type: "config" or "factory"
MODELS = {
    "maxformer_10_512": {
        "type": "config",
        "spec": "configs/maxformer/maxformer_10_512.yaml",
        "plugin_onnx": "sengine/exports/maxformer_10_512_imagenet_plugin.onnx",
    },
    "sew_resnet101": {
        "type": "factory",
        "spec": "sew_resnet101",
        "plugin_onnx": "sengine/exports/sew_resnet101_imagenet_plugin.onnx",
    },
    "spikingresformer_m": {
        "type": "config",
        "spec": "configs/spikingresformer/spikingresformer_m.yaml",
        "plugin_onnx": "sengine/exports/spikingresformer_m_imagenet_plugin.onnx",
    },
    "spikformer_4_512": {
        "type": "config",
        "spec": "configs/spikformer/spikformer_4_512.yaml",
        "plugin_onnx": "sengine/exports/spikformer_4_512_imagenet_plugin.onnx",
    },
}

# ─── Paths ───

NCU_PATH = "/opt/nvidia/nsight-compute/2025.1.1/ncu"
CUDA_HOME = "/usr/local/cuda-12.8"

GPUTIL_DIR = os.path.dirname(os.path.abspath(__file__))
NCU_REPORTS_DIR = os.path.join(GPUTIL_DIR, "ncu_reports")
RESULTS_DIR = os.path.join(GPUTIL_DIR, "results")
SENGINE_DIR = os.path.join(GPUTIL_DIR, "engines")
TRT_ENGINES_DIR = os.path.join(GPUTIL_DIR, "trt_engines")

# ─── ncu Metrics ───

NCU_METRICS = [
    "gpu__time_duration.sum",
    # DRAM / HBM
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    # Shared memory
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
    # SM occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Tensor Core
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_tensor.sum",
    # FMA / ALU
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_alu.sum",
    # LSU (load/store)
    "smsp__inst_executed_pipe_lsu.sum",
    # Launch config
    "launch__occupancy_per_register_count",
    "launch__block_size",
    "launch__grid_size",
    "sm__ctas_launched.sum",
]

# Human-readable names for the key utilization metrics (for tables)
METRIC_LABELS = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput (% peak)",
    "dram__bytes.sum.per_second": "DRAM Bandwidth (GB/s)",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed": "Shared Mem Util (% peak)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "SM Occupancy (% peak)",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "Tensor Core Util (% peak)",
    "smsp__inst_executed_pipe_tensor.sum": "Tensor Core Insts",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "FMA Util (% peak)",
    "smsp__inst_executed_pipe_fma.sum": "FMA Insts",
    "smsp__inst_executed_pipe_alu.sum": "ALU Insts",
    "smsp__inst_executed_pipe_lsu.sum": "LSU Insts",
    "launch__occupancy_per_register_count": "Occupancy (per register)",
    "gpu__time_duration.sum": "Kernel Duration (ns)",
}


def select_gpu():
    """Find the GPU with most free memory and 0% utilization."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,nounits,noheader"],
            text=True)
    except Exception:
        print("WARNING: nvidia-smi failed, defaulting to GPU 0")
        return 0

    best_gpu, best_free = 0, -1
    for line in out.strip().split('\n'):
        parts = [x.strip() for x in line.split(',')]
        idx, free_mb, util_pct = int(parts[0]), int(parts[1]), int(parts[2])
        if util_pct == 0 and free_mb > best_free:
            best_gpu, best_free = idx, free_mb
    # Fallback: if no idle GPU, pick the one with most free memory
    if best_free < 0:
        for line in out.strip().split('\n'):
            parts = [x.strip() for x in line.split(',')]
            idx, free_mb = int(parts[0]), int(parts[1])
            if free_mb > best_free:
                best_gpu, best_free = idx, free_mb
    return best_gpu


def plugin_onnx_path(model_key):
    return os.path.join(PROJECT_ROOT, MODELS[model_key]["plugin_onnx"])


def sengine_path(model_key, batch):
    return os.path.join(SENGINE_DIR, f"{model_key}_B{batch}.sengine")


def trt_engine_path(model_key, batch):
    return os.path.join(TRT_ENGINES_DIR, f"{model_key}_B{batch}.engine")


def trt_onnx_path(model_key, batch):
    return os.path.join(TRT_ENGINES_DIR, f"{model_key}_B{batch}.onnx")


def ncu_report_path(model_key, batch, backend):
    return os.path.join(NCU_REPORTS_DIR, f"{model_key}_B{batch}_{backend}")
