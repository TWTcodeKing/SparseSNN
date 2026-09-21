# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is a research framework for **post-training N:M structured sparsity** on Spiking Neural Networks (SNNs). The core contribution is **Spiking Brain Compression (SBC)**: second-order (OBS-based) weight pruning that uses a Surrogate Membrane Potential (SMP) Hessian encoding LIF temporal dynamics via the Van Rossum Distance convolution matrix, producing TensorRT-compatible 2:4 sparse models without retraining.

Two major subsystems:
1. **Training + Pruning** (`tengine/` + `sparse/`): train dense SNN models, apply SBC 2:4 pruning, optionally fine-tune with KD
2. **Inference Engine** (`sengine/`): TileLang fused kernels + BA-MTTS scheduling + CUDA Graph execution on RTX 4090 (tuning auto-adapts to A100), with forks for Jetson AGX Orin (`sengine_edge/`) and CPU (`sengine-cpu/`), plus TensorRT / TVM / torch.compile / ONNX Runtime baselines (`iengine/`, `scripts/bench_*_latency.py`)

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

# Training-time 2:4 regularization (SR-STE): --structured-sparse --sr-lambda 0.01 --sr-start-epoch/--sr-end-epoch,
# or the configs/spikformer/recipes/{structured_sparse,dynamic_sparse}.yaml recipes. The recipe comments
# mention iengine.structured_sparse.semi_structured_path and /home/twt/datasets; neither exists anymore.

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
# Optional: --permute-channels (helps 3x3 Conv, hurts 1x1) and --finetune <epochs> (KD)

# Batch all CIFAR-100 models -> obc_pt/
bash scripts/run_sbc_global_cifar100.sh [gpu_id]

# Full pipeline: SBC 2:4 -> channel permutation -> 1-epoch KD fine-tune -> obc_pt_ft/
# (architecture-aware: ResNets get perm+ft, transformers SBC only)
bash scripts/finetune_all.sh [gpu_id]
```

### SEngine: build + benchmark
The unified driver runs three phases: export plugin ONNX (TDL) -> fusion-validation pre-pass -> build + benchmark per batch size.
```bash
# --model (ResNet/factory) or --config (transformer); --fusion none|slicer (comma list = ablation)
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet \
    --T 4 --batch-sizes 1,4 --fusion slicer --autotune --precision fp32 --gpu-ids 0
# Quick run: drop --autotune (skips validation + tuning). Export only: --export-only

# Batch drivers (edit the model list inside)
bash scripts/bench_sengine.sh [gpu_id]
bash scripts/bench_transformer.sh configs/maxformer/maxformer_10_512.yaml [gpu_id]   # sengine vs TRT; SENGINE_ONLY=1 to skip TRT
```
Lower-level steps:
```bash
# 1. Export ONNX (plugin-mode with FusedIFNeuron/FusedLIFNeuron custom ops) -> sengine/exports/{model}_{dataset}_plugin.onnx
python -m sengine.scripts.export_onnx --model sew_resnet18 --dataset cifar100 --checkpoint output/.../best.pth

# 2. Build + benchmark (optionally --save model.sengine, --fusion slicer, --autotune, --precision fp32, --fusion-rec <json from step 4>)
python -m sengine.scripts.bench --onnx model_plugin.onnx --T 4 --batch 1

# 3. Benchmark a pre-built .sengine, optionally vs a TRT engine
python -m sengine.scripts.bench --sengine model.sengine [--trt model.engine --trt-input-shape 4,3,224,224]

# 4. Fusion validation pre-pass (profile fused vs decomposed per shape -> JSON recommendations)
python -m sengine.build.fusion_validator --onnx model_plugin.onnx --T 4 --batch 4 --output .cache/fusion_rec.json

# 5. Numerical correctness vs PyTorch reference
python -m sengine.scripts.verify_correctness --model sew_resnet18 --dataset cifar100 --T 4

# 6. C++ executor benchmark (bypasses Python entirely)
python -m sengine.scripts.bench_cpp --onnx model_plugin.onnx --T 4 --batch 1
```

### SEngine Programmatic API
```python
import sengine
engine = sengine.build("model.onnx", T=4, batch_size=1)
engine.save("model.sengine")
engine = sengine.load("model.sengine")
output = engine.infer(input_numpy)     # numpy in -> numpy out
ms = engine.benchmark()
```

### Baseline Benchmarks
```bash
# TensorRT latency: standard ONNX (no TDL) -> TRT engine, across batch sizes.
# --sparse for a 2:4 checkpoint; --export-only / --onnx for cross-platform (export on x86, bench on Jetson)
python scripts/bench_trt_latency.py --model sew_resnet18 --dataset cifar100 --batch-sizes 1,4,8,16
bash scripts/bench_trt.sh [gpu_id]

# TensorRT dense vs SBC-sparse accuracy + latency (needs dataset)
python -m iengine.backends.tensorrt.benchmark --model sew_resnet34 --dataset cifar100 \
    --data-root /data/twt/datasets --mode {dense|sparse|compare}

# torch.compile (Inductor) and ONNX Runtime CUDA EP
python scripts/bench_inductor_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4
python scripts/bench_onnxruntime_latency.py --config configs/spike_bert/spike_bert_small.yaml --dataset sst2 --T 4 --batch-sizes 1

# sengine vs TRT vs Inductor on the UT-HAR / UrbanSound8K workloads, B=4..32 (one idle GPU, tools interleaved per B)
bash scripts/bench_new_workloads.sh [gpu_id]        # full pipeline incl. autotune (slow); logs -> output/bench_new_workloads/
bash scripts/remeasure_new_workloads.sh [gpu_id]    # clean re-measurement with cached kernels/engines (+ nsys graph traces)
python scripts/parse_bench_new_workloads.py         # tabulate the logs
bash scripts/bench_extra_baselines.sh [gpu_id]       # ONNX Runtime CUDA EP + Inductor max-autotune, fp16/fp32
python scripts/export_new_workload_results.py       # -> output/bench_new_workloads/{latency,correctness,kernel_breakdown}_4090.csv
bash scripts/analyze_new_workloads.sh [gpu_id]      # follow-up: TRT nsys traces, per-kernel breakdown, real TRT fp32 engines -> trt_engines_fp32/
                                                    # (its last compute-sanitizer step hardcodes a stale scratchpad path; drop or repoint it)
# Check nvidia-smi utilization before trusting any number: this host is shared and a busy GPU inflates latency 1.5x+.

# TVM via torch.compile(backend='tvm') on the SNN->ANN converted model (TVM cannot run SNN ops;
# ANN batch = B*T to match compute). Needs a TVM install, see Environment Notes.
python scripts/bench_tvm_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4 --gpu-ids 0
```

### Profiling
```bash
# Per-kernel latency breakdown of a built sengine (TileLang .so kernels), optionally vs a TRT nsys trace
python scripts/analyze_kernel_latency.py --onnx sengine/exports/maxformer_10_768_imagenet_plugin.onnx \
    --T 4 --batch 4 [--trt-nsys trt_b4.nsys-rep]

# nsys / ncu on a TRT engine -> profiles/
bash scripts/profile.sh {nsys|ncu} trt_engines/<model>/<engine>.engine [output_name]

# GPU-utilization matrix (sengine vs TRT, ncu metrics; models/batches in GPUtil/config.py)
python GPUtil/build_engines.py --gpu-id 3                     # phase 1: build engines -> GPUtil/engines, GPUtil/trt_engines
sudo bash GPUtil/profile_one.sh maxformer_10_512 16 both 3    # phase 2 (ncu needs sudo); run_ncu.sh does the full matrix
python GPUtil/parse_ncu.py                                    # phase 3 -> GPUtil/results

# Fused-kernel speedup decomposition (DRAM-traffic reduction vs latency hiding)
python breakdown/bench_progressive.py --sengine GPUtil/engines/sew_resnet101_B32.sengine --gpu-id 2
```

### Kernel Smoke Tests
There is no test suite, lint, or CI. These scripts are the closest thing for kernels (each autotunes and compares against cuDNN):
```bash
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
    python scripts/bench_grouped_conv_smoke.py        # grouped Conv+BN (SpikingResFormer GWFFN shapes)
python scripts/bench_winograd_fused_smoke.py           # fused implicit Winograd vs im2col vs cuDNN
python scripts/bench_winograd_vs_imcol.py              # im2col vs Winograd across 3x3 s1 shapes
```
For end-to-end numerics use `sengine/scripts/verify_correctness.py`.

### Engine Correctness vs PyTorch (all three engines)
```bash
# Exports plugin ONNX with the checkpoint weights, builds the engine, compares logits on real test
# samples against PyTorch FP32 and FP16 (model.half()), optionally the whole test split (--eval-acc).
python scripts/verify_workloads.py --engine sengine --model snn_vgg16 --dataset ut_har --T 4 \
    --checkpoint output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth --num-samples 100 --eval-acc --gpu-ids 1
python scripts/verify_workloads.py --engine sengine_cpu --model snn_vgg9 --dataset urbansound8k --T 4 \
    --checkpoint output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth --eval-acc --threads 32
python scripts/verify_workloads.py --engine sengine_edge --model snn_vgg16 --dataset ut_har --T 4 --checkpoint ... --eval-acc
# Strict default: top-1 agreement >= 0.99 vs PyTorch FP16, mean cosine >= 0.99, |acc - acc_fp16| <= 0.5%.
# --fp16-tolerance judges relative to PyTorch's own FP16-vs-FP32 disagreement (low-confidence models).
```

### SEngine Variants (Jetson / CPU)
```bash
# Jetson AGX Orin (sengine_edge): export plugin ONNX on x86, build + bench on the Orin
python scripts/bench_sengine_edge_latency.py --model sew_resnet18 --dataset imagenet --T 4 --export-only
python scripts/bench_sengine_edge_latency.py --model sew_resnet18 --dataset imagenet --fusion none,slicer --autotune --batch-sizes 4

# CPU: SNN-VGG-9 on NTU-Fi HumanID, sengine-cpu MLAS kernel vs ONNX Runtime / OpenVINO / ncnn
python scripts/bench_snn_vgg9_cpu.py --threads 1,4,8 [--frameworks sengine,ort] [--numa 0]
# MLAS kernel build (needs a local MLAS checkout): see sengine-cpu/BUILD.md
cd sengine-cpu/csrc && make NO_BLAS=1   # C runtime libsengine_cpu.so (native conv path resolves scipy's OpenBLAS at runtime)
# Full CPU engine (native conv backend, no TVM needed): import sengine_cpu (repo-root symlink -> sengine-cpu)
python -c "import sengine_cpu; e = sengine_cpu.build('sengine/exports/snn_vgg9_urbansound8k_plugin.onnx', T=4, n_threads=32); print(e.infer(x).shape)"
```

### C++ Executor Build
```bash
# Standalone binary (sengine_exec.cu, via Makefile)
cd sengine/csrc && make clean && make

# Shared library (cpp_executor.cu, loaded via ctypes by sengine/runtime/cpp_executor.py)
cd sengine/csrc && nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC \
    -gencode=arch=compute_89,code=sm_89 \
    -o libsengine_exec.so cpp_executor.cu \
    -lcudart -ldl -lcublas
```
Both hardcode `sm_89`; change `-gencode` when building for another GPU (sengine_edge uses `sm_87` and `/usr/local/cuda`). The shared-library link needs cuDNN too: append `-L/usr/lib/x86_64-linux-gnu -lcudnn` (system cuDNN 8) or the command above fails at load with an undefined `cudnn*` symbol. The TileLang kernel `.so` export auto-detects the arch, so only the executor needs a manual rebuild.

## Architecture

### Two model specification paths
- **ResNet/factory models**: `--model <name>` (e.g., `sew_resnet_cifar56`). Factory functions in `models/sewresnet.py`, `models/msresnet.py`, registered in `tengine/utils.py:_RESNET_REGISTRY`. Available names: `sew_resnet{18,34,50,101,152}`, `sew_resnet_cifar{20,32,44,56,110}`, `ms_resnet{18,34,50,104}`, `ms_resnet_cifar{20,32,44,56,110}`, `ms_resnet_dvs20`, `dvs_sew_resnet`, `snn_vgg{9,11,16,19}`, `ems_yolo_res34`, `spike_yolo_{n,s,m}`.
- **Transformer/config models**: `--config <yaml>`. Builder functions in `models/__init__.py:ARCH_BUILDERS`. Supported archs: `spikformer`, `metaformer`, `qkformer`, `maxformer`, `ms_qkformer`, `spikingresformer`, `spike_yolo`, `ems_yolo`, `spike_bert`.
- Every driver (train/test/sbc/export/bench_*) accepts exactly one of `--model` / `--config`; benchmark scripts name outputs `{model or config basename}_{dataset}`.

### Config and recipe system
- **Config YAML** (transformers): architecture params (`arch`, `embed_dims`, `num_heads`, `depths`, `patch_size`, ...)
- **Recipe YAML**: training hyperparams (`optimizer`, `scheduler`, `augmentation`, `regularization`, `snn.T`, `epochs`, `batch_size`). Loaded via `tengine/utils.py:load_training_recipe()`; explicit CLI args override recipe defaults.
- **Dataset config**: hardcoded in `tengine/utils.py:_DATASET_CONFIG`: `cifar10` (32x32, 10cls), `cifar100` (32x32, 100cls), `imagenet` (224x224, 1000cls), `cifar10dvs` (128x128, 2ch, 10cls), `dvs128gesture` (128x128, 2ch, 11cls), `coco` (640x640, 80cls, detection), `gen1` (320x320, 2cls, detection), `sst2`/`mrpc` (NLP, seq_len=128), `ntufi_humanid` (WiFi CSI, 3x112x128 non-square, 14cls), `ut_har` (WiFi CSI, SenseFi preprocessed, 1x250x90, 7cls), `urbansound8k` (log-mel 1x64x176 of 4s clips, 10cls, folds 1-9 train / fold 10 test).

### Key directories
- **`models/`**: SNN model zoo. Neurons (`neurons.py`), layers (`layers.py`: `SeqToANNContainer`), ResNets (`sewresnet.py`, `msresnet.py`, `dvs_sewresnet.py`), VGG (`snn_vgg.py`), Transformers (`spikformer.py`, `metaformer.py`, `qkformer.py`, `maxformer.py`, `spikingresformer.py`), Detection (`spike_yolo.py`, `ems_yolo.py`, `detection_head.py`), NLP (`spike_bert.py`)
- **`models_ann/`**: SNN->ANN conversion (`build_ann_model`, `convert_snn_to_ann`): neurons -> ReLU, unwraps `SeqToANNContainer`, drops the T dim; monkey-patches per-arch blocks (MS-ResNet, SpikingResformer `_MultiStep*` ops and DSSA). Gives topology-identical ANN baselines for TVM / torch.compile / ORT
- **`sparse/`**: `sbc.py` (VRD matrix, ExactOBS N:M), `snn_sbc.py` (full pipeline: Hessian collection + pruning + BN recalibration + optional permutation/KD), `pruning.py` (magnitude primitives), `utils.py` (neuron detection, weight reshaping)
- **`tengine/`**: `train.py`, `test.py`, `transfer.py`, `dist.py` (DDP), `logger.py`, `utils.py` (model/dataset builders, checkpointing, recipes)
- **`iengine/`**: baseline backends. `backends/tensorrt/` (ONNX export, 2:4 sparse engine build, INT8 calibration, benchmark, `TRTRunner`), `backends/tvm/` (Relax frontend, MetaSchedule tuning)
- **`sengine/`**: custom SNN inference engine (below); **`sengine_edge/`**, **`sengine-cpu/`**: forks (below)
- **`datasets/`**: CIFAR, ImageNet, DVS (cifar10dvs, dvs128gesture), COCO/Gen1 (detection), GLUE (NLP), NTU-Fi HumanID and UT-HAR (WiFi CSI; UT-HAR `.csv` files are numpy `.npy`), UrbanSound8K (`urbansound8k.py` computes log-mel features once and caches `.npz` per fold under `UrbanSound8K/cache_logmel{n_mels}x{n_frames}/`)
- **`utils/`**: BN fusion (`fuse.py`), TRT engine profiling (`profile_trt_engine.py`)
- **`configs/`**: per-architecture YAML configs and `recipes/`
- **`scripts/`**: `train_*.sh`, `eval_*.sh`, `sparse_*.sh`, `finetune_all.sh`, and the Python benchmark drivers `bench_{sengine,sengine_edge,trt,inductor,onnxruntime,tvm}_latency.py`, `bench_snn_vgg9_cpu.py`, `analyze_kernel_latency.py`, kernel smoke tests
- **`GPUtil/`**: ncu GPU-utilization study (sengine vs TRT); experiment matrix in `config.py`, reports in `ncu_reports/`
- **`breakdown/`**: `bench_progressive.py` decomposes fused-kernel speedup (decomposed / fused-PS / interleaved production variants)
- **`motivation/`**: TensorRT nsys/ncu experiments on Spikformer showing launch overhead and temporal load imbalance; `run_all.sh` orchestrates, results in `motivation/output/`
- **`obc_pt/`**: SBC-pruned 2:4 checkpoints; **`obc_pt_ft/`**: pruned + permuted + KD fine-tuned; **`output/`**: training runs (`{model}_{dataset}_bs{B}_lr{lr}/`)
- **`docs/`**: `*.md` is gitignored (local-only). `SYSTEM_DESIGN.md` (cuSPARSELt / `.spkengine` era) and `open_questions.md` (WaveFuse-era API notes) are early designs and do not describe the current TileLang implementation; trust the code

### SNN-specific patterns
- **Temporal dimension**: all models process T timesteps. Input is `(B, C, H, W)`; models reshape to `(T, B, C, H, W)` or `(T*B, C, H, W)` internally.
- **Neuron reset**: `reset_net(model)` must be called after every forward pass to clear membrane state. Already wired into all training/evaluation loops.
- **Neuron types**: `LIFNeuron`/`IFNeuron`/`ILIFNeuron` (single-step) wrapped in `MultiStepLIFNeuron`/`MultiStepIFNeuron`/`MultiStepILIFNeuron` (multi-step). MS-ResNet uses `MSNeuron`. SpikeYOLO uses `ILIFNeuron` (integer LIF with multi-level quantization). SpikingResformer has its own `LIF` subclass. All types obtained via `sparse.utils._get_neuron_types()`.
- **SeqToANNContainer**: merges T and B dims for stateless ops (Conv2d, BN, Linear), then reshapes back. This is the mechanism TDL formalizes.

### SBC pruning flow (sparse/)
1. `collect_smp_hessians()`: forward hooks capture layer inputs, apply VRD matrix M, accumulate `H = 2*(MX)^T*(MX)` per layer
2. `sbc_prune_layer_nm_global()`: ExactOBS, per-row H^-1, element-wise greedy with N:M capacity constraint, OBS compensation + rank-1 H^-1 update
3. Conv2d uses im2col Hessian with columnslast permutation `(K, C*R*S) -> (K, R*S*C)` for TRT-compatible 2:4 along input channels
4. BN recalibration via `utils.fuse.recalibrate_bn()`

### SEngine: custom SNN inference engine (sengine/)

**Pipeline**: PyTorch model -> TDL transforms (5D->4D) -> ONNX export (plugin-mode with FusedIF/LIF/MS/ILIF custom ops) -> `ONNXParser` -> `EngineIR` -> optimization passes -> fusion strategy -> `TileLangCompiler` (kernel compilation with autotuning cache) -> BA-MTTS scheduling -> memory planning -> buffer planning -> standalone kernel `.so` export -> C++ executor + CUDA Graph capture -> `.sengine` serialized engine

**Key components**:
- **`engine.py`**: `SEngine` orchestrator; C++ CUDA Graph execution path (zero Python in the inference hot loop); `_detect_arch()` picks the SM arch from torch, `_detect_nvcc()` prefers `/usr/local/cuda-12.8`
- **`ir.py`**: `OpType`, `KernelVariant`, `BoundType` (COMPUTE/MEMORY), `NeuronType`, `AttentionParams`, `Node`/`Edge`/`FusionGroup`/`EngineIR`
- **`parser.py`**: ONNX parser for plugin-mode custom ops, shape inference, attention op detection
- **`optimizer.py`**: IR passes in order: BN folding, dead node elimination, fusion group detection, 2:4 sparsity validation, NHWC layout annotation, kernel variant selection, shape propagation + bound classification, edge layout propagation, layout reformat insertion
- **`fusion_strategy.py`**: pluggable fusion decisions decoupled from IR transforms: `none` (all decomposed, ablation baseline), `slicer` (greedy compute-anchored fusion)
- **`graph_slicer.py`**: each slice is anchored by a COMPUTE-bound op and absorbs reachable MEMORY-bound successors via BFS
- **`kernel_codegen.py`**: looks up epilogue patterns in `kernels/interleaved_templates.py` and compiles fused kernels with concrete shapes
- **`memory.py`**: tensor lifetime analysis, greedy first-fit pool allocation with 256-byte alignment
- **`cuda_graph_runtime.py`**: `CUDAGraphEngine` captures the BA-MTTS-scheduled kernel sequence into a CUDA Graph; all tensors pre-allocated
- **`bound_aware_scheduler.py`**: BA-MTTS: classifies ops as compute-bound (C) or memory-bound (M) and finds a topological order maximizing C<->M transitions to exploit hardware overlap
- **`build/`**: `engine_builder.py` (orchestration), `sengine_io.py` (.sengine serialization; BN params stored as named weight blobs, stem conv C_in padded to 16), `tilelang_compiler.py` (kernel factory; arch-adaptive tile search: A100+ gets larger tiles/deeper pipeline), `schedule_builder.py` (BA-MTTS), `tuning_cache.py`, `buffer_planner.py` (GPU buffer allocation + kernel bindings for the C++ executor), `fusion_validator.py` (profile fused vs decomposed pre-pass), `export_standalone.py` (TileLang kernels -> standalone `.so`, no TVM runtime deps)
- **`kernels/`**: TileLang kernels: `conv2d_bn_if_t4.py` (Conv+BN+IF, interleaved per-T epilogue), `spikformer_kernels.py`, `dwconv_bn.py`, `grouped_conv_bn.py`, `fused_attention_kernels.py`, `add_lif_fused.py`, `pool_lif_fused.py`, `winograd_conv.py` (F(2,3) for small 3x3), `interleaved_templates.py` (parameterized epilogue templates for codegen)
- **`runtime/`**: `plan_executor.py` (Python dispatch via CUDA Graph engine) and `cpp_executor.py` (ctypes wrapper for `csrc/libsengine_exec.so`)
- **`tdl/`**: Temporal Dimension Lowering: `transforms.py` (TDL-1/2/3; `export_with_fused_neurons()`), `neuron_ops.py`, `attention_ops.py`, `ssa_4d.py`/`dssa_4d.py`, `analysis.py`, `cost_model.py`, `temporal_unroll.py`, `slicegraph.py`, `graph_ir.py`, `model_dag/` (one DAG tracer per architecture: sewresnet, msresnet, spikformer, metaformer, qkformer, maxformer, spikingresformer)
- **`tuning/`**: `analytical.py` (cost model), `hw_calibrate.py` (hardware calibration), `roofline.py` (roofline-based tile configs)
- **`csrc/`**: `cpp_executor.cu` (library `.so`), `sengine_exec.cu` (standalone binary), `kernels.cuh` (native IF/LIF/Add/Pool/TemporalMean kernels). Loads TileLang `.so` via `dlopen`
- **`scripts/`**: `bench.py`, `bench_cpp.py`, `export_onnx.py`, `verify_correctness.py`
- **`exports/`**: plugin/standard ONNX and `.sengine` files (`{model}_{dataset}_plugin.onnx`)

**Caches**:
- Autotuning: `~/.cache/sengine/tuning_cache.json`, keyed by (shape, gpu_name, gpu_arch, T, B). Delete it (or the entries) to force retuning.
- Project `.cache/` (gitignored): standalone kernel `.so` build dirs (`sengine_B{batch}/`, `sengine_edge_B{batch}/`; files are `kern_{nid}_{sha1(source)}.so`, so different models or forks can share a dir) and fusion recommendations (`fusion_rec_*.json`). `.sengine-cpu.cache/` is the sengine-cpu TVM kernel build dir (`DEFAULT_BUILD_DIR` in `sengine-cpu/build/tvm_compiler.py`).

### TDL (Temporal Dimension Lowering)
Converts the 5D SNN execution model `(T, B, C, H, W)` to 4D `(T*B, C, H, W)` via three graph transforms:
- **TDL-1 (T-Axis Absorption)**: patches `SeqToANNContainer` wrappers so stateless ops process all timesteps at once
- **TDL-2 (Stateful Extraction)**: replaces spiking neurons with `FusedIFOp`/`FusedLIFOp`/`FusedMSOp`/`FusedILIFOp` custom ops that loop over T internally
- **TDL-3 (Attention Decomposition)**: replaces attention blocks with 4D variants (`SpikformerSSA4D`, `MaxFormerSSA4D`, `TokenQKA4D`, DSSA4D)

### SEngine variants (forks of sengine/)
Both mirror sengine's module layout. A change to sengine core does NOT propagate; patch each fork separately.
- **`sengine_edge/`**: Jetson AGX Orin (sm_87, 16 SMs, 100KB smem, 4MB L2). Differences: Orin-specific tile-config generation in `tuning/roofline.py` (occupancy-weighted to hide LPDDR5 latency), a T-unrolled 3x3 fused kernel (one im2col tile per timestep, pipelining disabled), L1-carveout tweaks and an L2 persistence API in `csrc/cpp_executor.cu` (`sengine_setup_l2_persistence()` pins weights in L2; only enabled when the device name contains "Orin", `SENGINE_EDGE_L2_PERSIST=1` forces it), `csrc/Makefile` uses `/usr/local/cuda` (JetPack) + sm_87. Export dir `sengine_edge/exports/`, kernel build dir `.cache/sengine_edge_B{batch}`. For functional testing on this x86 host the executor `.so` was rebuilt for sm_89 (`csrc/libsengine_exec.sm87.so` is the old Orin binary; rebuild with `make`/nvcc on the Orin). `--fusion none` currently segfaults in the edge fork.
- **`sengine-cpu/`**: CPU engine (hyphenated dir; imported as `sengine_cpu` via the repo-root symlink). FP32 throughout, NHWC activations. Conv backends: **native** (`csrc/native_conv.c`: NHWC im2col + cblas_sgemm resolved at runtime from scipy's bundled OpenBLAS, fused BN + IF/LIF T-loop; the default whenever the TVM interpreter is absent, or `SENGINE_CPU_BACKEND=native`) and TVM TE kernels (`build/tvm_compiler.py`, subprocess into a TVM venv, `SENGINE_CPU_BACKEND=tvm`). Native ops in `csrc/native_kernels.c` (IF/LIF, Add, MaxPool, GAP, TemporalMean, GEMM+bias, Softmax); the C executor (`csrc/cpu_executor.c`) dispatches a BA-MTTS schedule whose dependencies are traced through zero-cost/absorbed nodes. `engine.infer(x)` takes NCHW (B,C,H,W), tiles T times into the Tile node buffer, and returns (B, classes). Standalone fused MLAS kernel `csrc/mlas_conv_bn_lif.cpp` is only used by `scripts/bench_snn_vgg9_cpu.py`; `kernels_exo/` holds an Exo-generated 6x8 SGEMM. ONNX Runtime/OpenVINO/ncnn are comparison baselines. Build docs: `sengine-cpu/BUILD.md`.

## Environment Notes

- Python virtualenv at `.venv/` (Python 3.12). Use `uv pip install` for packages (no sudo). There is no `requirements.txt`; key installed versions: torch 2.5.1+cu121, tilelang 0.1.8, spikingjelly 0.0.0.0.14, tensorrt-cu12 10.7.0, onnx 1.21.0, onnxsim 0.6.2, onnxruntime-gpu 1.25.1, cupy-cuda12x 14, openvino 2026.1, ncnn. Use `uv run` (or `.venv/bin/python`) for scripts.
- CUDA 12.8 at `/usr/local/cuda-12.8`; this host has 8x RTX 4090 (sm_89), shared with other users (check `nvidia-smi` for free memory and utilization and pick an idle GPU before benchmarking; a busy GPU inflates latency 1.5x+). Benchmark scripts export `CUDA_HOME=/usr/local/cuda-12.8`, prepend its `bin` to `PATH`, and set `TORCH_CUDA_ARCH_LIST=8.9`; do the same when running TileLang code by hand.
- GPU arch: TileLang kernel export and the tuning cache auto-detect the arch, and tile search adapts to A100 (>=160KB smem). The C++ executor `.so`/Makefiles hardcode `sm_89` (see C++ Executor Build).
- Datasets at `/data/twt/datasets/` (`UT_HAR/` from SenseFi's `UT_HAR.zip`, `UrbanSound8K/UrbanSound8K/` extracted from the Zenodo tarball, `UCI-HAR/` is a different smartphone-IMU dataset).
- Nsight Compute at `/opt/nvidia/nsight-compute/2025.1.1/ncu`; the ncu scripts require sudo. `compute-sanitizer` must be run from `/usr/local/cuda-12.8/bin/` (the `/usr/bin` one lacks its injection library).
- TVM: `iengine/backends/tvm/`, `sengine-cpu/build/tvm_compiler.py`, `sengine-cpu/kernels/*.py`, and `sengine-cpu/runtime/py_executor.py` hardcode an isolated interpreter at `/home/twt/tvm_build/tvm_venv/bin/python` (kept separate because tilelang bundles its own TVM whose `tvm_ffi` conflicts). That path does not exist on this host at the moment; `sengine-cpu/env_setup.sh` builds a venv at `sengine-cpu/.tvm-env` (update the hardcoded path if you use it). `scripts/bench_tvm_latency.py` needs `tvm` importable by the interpreter that runs it.
- Generated outputs (all gitignored): `sengine/exports/`, `trt_engines/` (fp16), `trt_engines_fp32/`, `onnxrt_exports/`, `profiles/`, `output/`, `obc_pt/`, `obc_pt_ft/`, `.cache/`, `.sengine-cpu.cache/`, `third-party/`, `*.so`, `*.onnx`, `*.engine`, `*.sengine`, `*.pth`.
- `main.py` is a scratch file, not an entry point.
- Known gaps: `sengine --precision fp32` does not match PyTorch on SNN-VGG16/UT-HAR (fp16 does); the sengine `fusion=none` ablation path is untested on the new workloads.
