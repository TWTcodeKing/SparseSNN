# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is a research framework for **post-training N:M structured sparsity** on Spiking Neural Networks (SNNs). The core contribution is **Spiking Brain Compression (SBC)**: second-order (OBS-based) weight pruning that uses a Surrogate Membrane Potential (SMP) Hessian encoding LIF temporal dynamics via the Van Rossum Distance convolution matrix, producing TensorRT-compatible 2:4 sparse models without retraining.

The project has two major subsystems:
1. **Training + Pruning** (`tengine/` + `sparse/`): Train dense SNN models, apply SBC 2:4 pruning, optionally fine-tune with KD
2. **Inference Engine** (`sengine/`): TileLang-based fused kernels + BA-MTTS scheduling + CUDA Graph execution, targeting RTX 4090

## Common Commands

### Training
```bash
# ResNet models (direct factory via --model)
uv run tengine/train.py --model sew_resnet_cifar56 --dataset cifar100 \
    --data-root /data/twt/datasets --gpu-ids 0

# Transformer models (config-based via --config)
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --dataset cifar100 --data-root /data/twt/datasets --gpu-ids 0

# Multi-GPU DDP
torchrun --nproc_per_node=4 tengine/train.py --config <yaml> \
    --dataset imagenet --data-root /data/twt/datasets --gpu-ids 0,1,2,3

# Training with recipe (YAML defaults, CLI overrides take priority)
uv run tengine/train.py --model ms_resnet_cifar110 \
    --recipe configs/ms_resnet/recipes/cifar100_cifar_arch.yaml \
    --dataset cifar100 --data-root /data/twt/datasets
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
# Full pipeline: SBC 2:4 -> channel permutation -> 1-epoch KD fine-tune -> save to obc_pt_ft/
bash scripts/finetune_all.sh [gpu_id]
```

### End-to-End Inference Pipeline
```bash
# 1. Export ONNX (plugin-mode with FusedIFNeuron/FusedLIFNeuron custom ops)
python -m sengine.scripts.export_onnx --model sew_resnet18 --dataset cifar100 \
    --data-root /data/twt/datasets --checkpoint output/.../best.pth

# 2. Build SEngine + benchmark
python -m sengine.scripts.bench --onnx model_plugin.onnx --T 4 --batch 1

# 3. Benchmark a pre-built .sengine file
python -m sengine.scripts.bench --sengine model.sengine

# 4. Compare sengine vs TensorRT
python -m sengine.scripts.bench --sengine model.sengine --trt model.engine --trt-input-shape 4,3,224,224

# 5. Full comparison across batch sizes
bash scripts/bench_sengine_vs_trt.sh [gpu_id]

# 6. Verify numerical correctness against PyTorch reference
python -m sengine.scripts.verify_correctness --onnx model_plugin.onnx --T 4
```

### SEngine Programmatic API
```python
import sengine

# Build from ONNX
engine = sengine.build("model.onnx", T=4, batch_size=1)
engine.save("model.sengine")

# Load and run
engine = sengine.load("model.sengine")
output = engine.infer(input_numpy)     # numpy in -> numpy out
ms = engine.benchmark()                # latency measurement
```

### TensorRT Baseline Benchmark
```bash
python -m iengine.backends.tensorrt.benchmark --model sew_resnet34 --mode {dense|sparse|compare}
```

## Architecture

### Two model specification paths
- **ResNet models**: `--model <name>` (e.g., `sew_resnet_cifar56`). Factory functions in `models/sewresnet.py`, `models/msresnet.py`, registered in `tengine/utils.py:_RESNET_REGISTRY`. Available names: `sew_resnet{18,34,50,101,152}`, `sew_resnet_cifar{20,32,44,56,110}`, `ms_resnet{18,34,50,104}`, `ms_resnet_cifar{20,32,44,56,110}`, `ms_resnet_dvs20`, `dvs_sew_resnet`, `snn_vgg{9,11,16,19}`.
- **Transformer models**: `--config <yaml>`. Builder functions in `models/__init__.py:ARCH_BUILDERS`. Supported archs: `spikformer`, `metaformer`, `qkformer`, `maxformer`, `ms_qkformer`, `spikingresformer`.

### Config and recipe system
- **Config YAML** (transformers): defines architecture params (`arch`, `embed_dims`, `num_heads`, `depths`, `patch_size`, etc.)
- **Recipe YAML**: defines training hyperparams (`optimizer`, `scheduler`, `augmentation`, `regularization`, `snn.T`, `epochs`, `batch_size`). Loaded via `tengine/utils.py:load_training_recipe()`. Recipe defaults are overridden by explicit CLI args.
- **Dataset config**: hardcoded in `tengine/utils.py:_DATASET_CONFIG` — `cifar10` (32x32, 10cls), `cifar100` (32x32, 100cls), `imagenet` (224x224, 1000cls), `cifar10dvs` (128x128, 2ch, 10cls), `dvs128gesture` (128x128, 2ch, 11cls).

### Key directories
- **`models/`** — SNN model zoo: neurons (`neurons.py`), layers (`layers.py`: `SeqToANNContainer`), ResNets (`sewresnet.py`, `msresnet.py`, `dvs_sewresnet.py`), Transformers (`spikformer.py`, `metaformer.py`, `qkformer.py`, `maxformer.py`, `spikingresformer.py`)
- **`sparse/`** — Pruning: `sbc.py` (VRD matrix, ExactOBS N:M), `snn_sbc.py` (full pipeline: Hessian collection + pruning + BN recalibration), `pruning.py` (magnitude primitives), `utils.py` (neuron detection, weight reshaping)
- **`tengine/`** — Training engine: `train.py`, `test.py`, `transfer.py`, `dist.py` (DDP), `logger.py`, `utils.py` (model/dataset builders, checkpointing, recipes)
- **`iengine/`** — TensorRT baseline: ONNX export, engine build with 2:4 sparse support, INT8 calibration, benchmarking
- **`sengine/`** — Custom SNN inference engine (see section below)
- **`datasets/`** — Data loaders for all supported datasets
- **`utils/`** — BN fusion (`fuse.py`), TRT engine profiling (`profile_trt_engine.py`)
- **`configs/`** — Per-architecture YAML configs and training recipes
- **`obc_pt/`** — SBC-pruned 2:4 sparse checkpoints; **`obc_pt_ft/`** — pruned + permuted + KD fine-tuned checkpoints
- **`scripts/`** — Shell templates: `train_*.sh`, `eval_*.sh`, `sparse_*.sh`, `bench_*.sh`, `finetune_all.sh`
- **`docs/`** — System design (`SYSTEM_DESIGN.md`), reference papers

### SNN-specific patterns
- **Temporal dimension**: all models process T timesteps. Input is `(B, C, H, W)`; models reshape to `(T, B, C, H, W)` or `(T*B, C, H, W)` internally.
- **Neuron reset**: `reset_net(model)` must be called after every forward pass to clear membrane state. Already wired into all training/evaluation loops.
- **Neuron types**: `LIFNeuron`/`IFNeuron` (single-step) wrapped in `MultiStepLIFNeuron`/`MultiStepIFNeuron` (multi-step). MS-ResNet uses `MSNeuron`. All types obtained via `sparse.utils._get_neuron_types()`.
- **SeqToANNContainer**: wrapper that merges T and B dims for stateless ops (Conv2d, BN, Linear), then reshapes back. This is the mechanism TDL formalizes.

### SBC pruning flow (sparse/)
1. `collect_smp_hessians()` — forward hooks capture layer inputs, apply VRD matrix M, accumulate `H = 2*(MX)^T*(MX)` per layer
2. `sbc_prune_layer_nm_global()` — ExactOBS: per-row H^-1, element-wise greedy with N:M capacity constraint, OBS compensation + rank-1 H^-1 update
3. Conv2d uses im2col Hessian with columnslast permutation `(K, C*R*S) -> (K, R*S*C)` for TRT-compatible 2:4 along input channels
4. BN recalibration via `utils.fuse.recalibrate_bn()`

### SEngine — custom SNN inference engine (sengine/)

**Pipeline**: PyTorch model -> TDL transforms (5D->4D) -> ONNX export (plugin-mode with FusedIF/LIF/MS custom ops) -> `ONNXParser` -> `EngineIR` -> optimization passes -> `TileLangCompiler` (kernel compilation with autotuning cache) -> BA-MTTS scheduling -> memory planning -> CUDA Graph capture -> `.sengine` serialized engine

**Key components**:
- **`engine.py`** — `SEngine` class: main orchestrator; wraps full pipeline; C++ CUDA Graph execution path (zero Python in inference hot loop)
- **`ir.py`** — Graph IR: `OpType` enum, `KernelVariant` enum, `BoundType` (COMPUTE/MEMORY), `Node`/`Edge`/`FusionGroup`/`EngineIR` classes
- **`parser.py`** — ONNX parser: handles plugin-mode custom ops, shape inference, attention op detection
- **`optimizer.py`** — 7 IR passes (in order): BN folding, dead node elimination, fusion group detection, 2:4 sparsity validation, NHWC layout annotation, kernel variant selection, shape propagation + bound classification
- **`build/`** — `engine_builder.py` (orchestration), `sengine_io.py` (.sengine serialization), `tilelang_compiler.py` (kernel factory), `schedule_builder.py` (BA-MTTS), `tuning_cache.py`
- **`bound_aware_scheduler.py`** — BA-MTTS: classifies ops as compute-bound (C) or memory-bound (M), finds topological order maximizing C<->M transitions to exploit GPU hardware overlap
- **`kernels/`** — TileLang fused kernels: `conv2d_bn_if_t4.py` (Conv+BN+IF for T=4), `spikformer_kernels.py` (transformer ops), `dwconv_bn.py` (depthwise), `grouped_conv_bn.py` (grouped), `fused_attention_kernels.py`
- **`tdl/`** — Temporal Dimension Lowering: `transforms.py` (TDL-1/2/3 graph transforms), `neuron_ops.py` (FusedIFOp/FusedLIFOp/FusedMSOp), `ssa_4d.py`/`dssa_4d.py` (4D attention variants), `model_dag/` (DAG tracing)
- **`csrc/cpp_executor.cu`** — C++ CUDA executor: zero external deps beyond CUDA runtime, implements native IF/LIF/Add/Pool/TemporalMean kernels, loads TileLang `.so` via `dlopen`, CUDA Graph capture/replay. Exposed to Python via ctypes in `runtime/cpp_executor.py`.
- **`scripts/`** — `bench.py` (build + benchmark + TRT comparison), `export_onnx.py` (ONNX export), `verify_correctness.py` (numerical validation)

### TDL (Temporal Dimension Lowering) — bridge between training models and sengine
TDL converts the 5D SNN execution model `(T, B, C, H, W)` to 4D `(T*B, C, H, W)` via three graph transforms:
- **TDL-1 (T-Axis Absorption)**: patches `SeqToANNContainer` wrappers so stateless ops process all timesteps at once
- **TDL-2 (Stateful Extraction)**: replaces spiking neurons with `FusedIFOp`/`FusedLIFOp`/`FusedMSOp` custom ops that loop over T internally
- **TDL-3 (Attention Decomposition)**: replaces attention blocks with 4D variants (`SpikformerSSA4D`, `MaxFormerSSA4D`, `TokenQKA4D`, DSSA)

## Environment Notes

- Python virtualenv at `.venv/`, use `uv pip install` for packages (no sudo access)
- CUDA 12.8 at `/usr/local/cuda-12.8`, SM arch 8.9 (RTX 4090)
- Datasets at `/data/twt/datasets/` (not under home)
- GPU server; CUDA required for all model work
- Use `uv run` instead of `python` when running training scripts to ensure venv
- No formal test suite, linting, or CI — research codebase. Use `sengine/scripts/verify_correctness.py` for numerical validation.
- Key dependencies: `spikingjelly==0.0.0.0.14` (SNN framework), `tilelang>=0.1.8` (kernel compiler), `tensorrt-cu12>=10.7.0` (baseline)
- `scripts/` contains shell templates that invoke Python entry points with model-specific args
- C++ executor compiled as `sengine/csrc/libsengine_exec.so`; rebuilt via CUDA compiler when source changes
