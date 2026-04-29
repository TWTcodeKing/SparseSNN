# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is a research framework for **post-training N:M structured sparsity** on Spiking Neural Networks (SNNs). The core contribution is **Spiking Brain Compression (SBC)**: second-order (OBS-based) weight pruning that uses a Surrogate Membrane Potential (SMP) Hessian encoding LIF temporal dynamics via the Van Rossum Distance convolution matrix, producing TensorRT-compatible 2:4 sparse models without retraining.

## Common Commands

### Training
```bash
# ResNet models (direct factory)
uv run tengine/train.py --model sew_resnet_cifar56 --dataset cifar100 \
    --data-root /data/twt/datasets --gpu-ids 0

# Transformer models (config-based)
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --dataset cifar100 --data-root /data/twt/datasets --gpu-ids 0

# Multi-GPU DDP
torchrun --nproc_per_node=4 tengine/train.py --config <yaml> \
    --dataset imagenet --data-root /data/twt/datasets --gpu-ids 0,1,2,3

# Training with recipe (YAML defaults, CLI overrides)
uv run tengine/train.py --model ms_resnet_cifar110 \
    --recipe configs/ms_resnet/recipes/cifar100_cifar_arch.yaml \
    --dataset cifar100 --data-root /data/twt/datasets
```

### SBC Pruning (post-training 2:4 sparsification)
```bash
python -m sparse.snn_sbc \
    --model sew_resnet_cifar56 \
    --dense-checkpoint output/.../best.pth \
    --dataset cifar100 --data-root /data/twt/datasets \
    --T 4 --nm 2 4 --evaluate

# Batch all CIFAR-100 models
bash scripts/run_sbc_global_cifar100.sh [gpu_id]
```

### SBC + Channel Permutation + KD Fine-tuning
```bash
# Full pipeline: SBC 2:4 → channel permutation → 1-epoch KD fine-tune → save to obc_pt_ft/
bash scripts/finetune_all.sh [gpu_id]
```

### Evaluation
```bash
python tengine/test.py --model sew_resnet_cifar56 --dataset cifar100 \
    --data-root /data/twt/datasets --checkpoint output/.../best.pth

python tengine/test.py --config configs/maxformer/maxformer_cifar.yaml \
    --dataset cifar100 --data-root /data/twt/datasets --checkpoint output/.../best.pth
```

### Transfer Learning
```bash
uv run tengine/transfer.py --config configs/spikingresformer/spikingresformer_ti.yaml \
    --pretrained checkpoints/spikingresformer/ImageNet_spikingresformer_ti.pth \
    --dataset cifar100 --data-root /data/twt/datasets --img-size 128
```

### Dense Inference Benchmark
```bash
python -m iengine.dense --config <yaml> --dataset cifar100 \
    --data-root /data/twt/datasets --checkpoint <path> --compile --fuse-neurons
```

### CUTLASS Backend Compile + Benchmark
```bash
python -m iengine.backends.cutlass.compile --model sew_resnet34 --T 4 --batch 1 --benchmark
```

### TensorRT Backend Benchmark
```bash
python -m iengine.backends.tensorrt.benchmark --model sew_resnet34 --mode {dense|sparse|compare}
```

### WaveFuse Tests
```bash
pytest wavefuse/tests/                         # IR correctness, kernel tests, e2e pipeline (79 tests)
```

### TileLang Kernel Tests (legacy)
```bash
pytest iengine/backends/tilelang/tests/       # fused Conv2d/Linear+BN+Neuron correctness
```

### SEngine Pipeline
```bash
python sengine/scripts/test_sengine.py        # Phase A: Python IR pipeline test
```

## Architecture

### Two model specification paths
- **ResNet models**: specified by `--model <name>` (e.g., `sew_resnet_cifar56`, `ms_resnet104`). Factory functions live in `models/sewresnet.py` and `models/msresnet.py`, registered in `tengine/utils.py:_RESNET_REGISTRY`.
- **Transformer models**: specified by `--config <yaml>` (e.g., `configs/spikformer/spikformer_8_384.yaml`). Builder functions registered in `models/__init__.py:ARCH_BUILDERS` dict.

### Key directories
- **`models/`** — SNN model zoo: neurons (`neurons.py`), layers (`layers.py`: `SeqToANNContainer`), ResNets (`sewresnet.py`, `msresnet.py`, `dvs_sewresnet.py`), Transformers (`spikformer.py`, `metaformer.py`, `qkformer.py`, `maxformer.py`, `spikingresformer.py`)
- **`sparse/`** — Pruning algorithms: `sbc.py` (SBC core: VRD matrix, ExactOBS N:M), `snn_sbc.py` (full pipeline: Hessian collection + pruning + BN recalibration), `pruning.py` (magnitude pruning primitives), `utils.py` (neuron detection, weight reshaping, firing rate collection)
- **`tengine/`** — Training engine: `train.py`, `test.py`, `transfer.py`, `dist.py` (DDP), `logger.py`, `utils.py` (model/dataset builders, checkpointing, recipes)
- **`iengine/`** — Inference engine: `dense.py` (PyTorch baseline), `tdl/` (Temporal Dimension Lowering graph transforms), `backends/` (TensorRT, TileLang, CUTLASS, plus CoreML/DeepSparse stubs)
- **`datasets/`** — Data loaders for CIFAR-10/100, ImageNet, CIFAR10-DVS, DVS128Gesture
- **`utils/`** — BN fusion (`fuse.py`: absorb BN into Conv/Linear post-pruning), spike pattern collection (`patterns.py`), TRT engine profiling (`profile_trt_engine.py`)
- **`configs/`** — Per-architecture YAML configs and training recipes
- **`sengine/`** — Unified SNN inference engine (successor to spike_engine + wavefuse). ONNX frontend → EngineIR → STAKG partitioning → CUTLASS/cuSPARSELt codegen → C++/CUDA runtime. `sengine/python/sengine/` (Python IR, parser, optimizer, memory planner), `sengine/csrc/` (C++17/CUDA kernels), `sengine/codegen/` (template renderer)
- **`wavefuse/`** — PyTorch-to-CUDA co-scheduling compiler. FX trace → WaveGraph IR → topological pairing → fused Conv+BN+Neuron kernels with warp-level compute/memory interleaving (PTX bar.sync). Entry: `wavefuse.compile(model, example_input)` → `CompiledSNN`
- **`spike_engine/`** — Original C++/CUDA inference engine (being absorbed by sengine). Hand-written fused kernels, binary `.spkengine` serialized engine format. Baseline for TRT comparison benchmarks
- **`obc_pt/`** — SBC-pruned 2:4 sparse checkpoints (`.pth`)
- **`obc_pt_ft/`** — SBC-pruned + channel permutation + 1-epoch KD fine-tuned checkpoints (suffix `_perm_ft1`)
- **`third-party/`** — Vendored TensorRT headers and nnfusion (Rammer) reference (for compilation only)
- **`docs/`** — Reference papers (PDFs), weekly design decisions (`week*_design_decisions.md`), results (`week*_results.md`)

### SNN-specific patterns
- **Temporal dimension**: all models process T timesteps internally. Input is `(B, C, H, W)`; models reshape to `(T, B, C, H, W)` or `(T*B, C, H, W)` internally.
- **Neuron reset**: `reset_net(model)` must be called after every forward pass to clear membrane state. This is already wired into all training/evaluation loops.
- **Neuron types**: `LIFNeuron`/`IFNeuron` (single-step) wrapped in `MultiStepLIFNeuron`/`MultiStepIFNeuron` (multi-step). MS-ResNet uses its own `MSNeuron`. The tuple of all types is obtained via `sparse.utils._get_neuron_types()`.
- **SeqToANNContainer**: wrapper that merges T and B dims for stateless ops (Conv2d, BN, Linear), then reshapes back. This is the mechanism TDL formalizes.

### SBC pruning flow (sparse/)
1. `collect_smp_hessians()` — forward hooks capture layer inputs, apply VRD matrix M, accumulate `H = 2*(MX)^T*(MX)` per layer
2. `sbc_prune_layer_nm_global()` — ExactOBS: per-row H^-1, element-wise greedy with N:M capacity constraint, OBS compensation + rank-1 H^-1 update
3. Conv2d uses im2col Hessian with columnslast permutation `(K, C*R*S) -> (K, R*S*C)` for TRT-compatible 2:4 along input channels
4. BN recalibration via `utils.fuse.recalibrate_bn()`

### Inference engine (iengine/) — baseline reference
- **TDL** (`iengine/tdl/`): Temporal Dimension Lowering transforms SNN graphs. Three transforms: T-Axis Absorption (stateless ops), Stateful Extraction (neurons), Temporal Attention Decomposition. Produces `OperatorDAG` → `TemporalDAG` → `SliceGraphDP` for partitioning. Includes cost model (`cost_model.py`) and DSSA 4D analysis (`dssa_4d.py`).
- **Backends** (`iengine/backends/`):
  - **TensorRT** — ONNX export (`export.py`), engine build (`builder.py`) with 2:4 sparse support, INT8 calibration (`calibrator.py`), runtime inference (`runtime.py`), benchmarking (`benchmark.py`)
  - **TileLang** — Legacy reference: fused Conv2d+BN+Neuron kernels (dense and 2:4 sparse variants). Tests pass but not part of active development
  - **CUTLASS** — Conv2d+BN+IF fused kernels (`kernels/conv2d_if_kernel.cuh`, `kernels/snn_epilogue.h`) using CUTLASS implicit GEMM with custom epilogue functor; `compile.py` builds and benchmarks, `backend.py` provides runtime interface

### SEngine — unified inference engine (sengine/)
- **Goal**: Hardware-optimized SNN inference with dual-side sparsity (weight 2:4 sparse + activation tile-skip) targeting RTX 4090
- **Pipeline**: ONNX → EngineIR (parser) → STAKG partition (fusion selection via roofline cost model + greedy 1-step lookahead) → codegen (C++/CUDA with CUTLASS sparse GEMM) → binary `.spkengine` → runtime
- **Key algorithm**: STAKG (Spatial-Temporal DAG Aware Kernel Grouping) in `sengine/python/sengine/stakg.py` — selects Conv+BN+Neuron fusion groups, schedules for resource conflict avoidance
- **Backend**: CUTLASS for Conv2d, cuSPARSELt for weight compression + sparse Linear only (not Conv)
- **Scripts**: `sengine/scripts/` for pipeline tests, benchmarks, profiling
- **Design docs**: `SYSTEM_DESIGN.md` (full architecture), `CODING_PLAN.md` (implementation roadmap)

### WaveFuse — co-scheduling compiler (wavefuse/)
- **Goal**: Fused Conv+BN+Neuron kernel generation with warp-level interleaving between compute (Conv) and memory (LIF) ops
- **Pipeline**: PyTorch FX trace (`wavefuse.frontend.trace_snn()`) → WaveGraph IR (topological layers, persistent state tracking) → DAG-independent pairing → CUDA codegen (Jinja2 templates) → CUDA graph capture → `CompiledSNN`
- **API**: `wavefuse.compile(model, example_input)` → `CompiledSNN` with `.reset_state()` and `.__call__(x)`
- **Results**: 14-18% speedup on 128+ channel workloads via PTX bar.sync barriers; 79/79 tests passing

## Environment Notes

- Python virtualenv at `.venv/`, use `uv pip install` for packages (no sudo access)
- Datasets at `/data/twt/datasets/` (not under home)
- GPU server; CUDA required for all model work
- Use `uv run` instead of `python` when running training scripts to ensure venv
- No formal build system, linting, or CI at repo level — research codebase
- `scripts/` contains shell templates (`train_*.sh`, `eval_*.sh`, `sparse_*.sh`, `profile.sh`) that invoke the Python entry points with model-specific args
