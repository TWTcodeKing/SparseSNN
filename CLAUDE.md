# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is a research framework for **post-training N:M structured sparsity** on Spiking Neural Networks (SNNs). The core contribution is **Spiking Brain Compression (SBC)**: second-order (OBS-based) weight pruning that uses a Surrogate Membrane Potential (SMP) Hessian encoding LIF temporal dynamics via the Van Rossum Distance convolution matrix, producing TensorRT-compatible 2:4 sparse models without retraining.

Two major subsystems:
1. **Training + Pruning** (`tengine/` + `sparse/`): train dense SNN models, apply SBC 2:4 pruning, optionally fine-tune with KD
2. **Inference Engine** (`sengine/`): TileLang fused kernels + BA-MTTS scheduling + CUDA Graph execution. One package with **target profiles** for RTX 4090 (`ada`, default), A100 (`a100`) and Jetson AGX Orin (`orin`); `sengine_cpu/` is the CPU port. TensorRT / TVM / torch.compile / ONNX Runtime baselines live in `iengine/` and `scripts/bench_*_latency.py`.

`README.md` is the reader-facing entry point (setup, data layout, reproduction chain, results); `docs/ARCHITECTURE.md` describes the design. Keep all three in sync when commands or layout change.

## Common Commands

### Training
```bash
# ResNet/factory models (--model)
uv run tengine/train.py --model sew_resnet_cifar56 --dataset cifar100 \
    --data-root /data/twt/datasets --gpu-ids 0

# Transformer models (--config)
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --dataset cifar100 --data-root /data/twt/datasets --gpu-ids 0

# Multi-GPU DDP
torchrun --nproc_per_node=4 tengine/train.py --config <yaml> \
    --dataset imagenet --data-root /data/twt/datasets --gpu-ids 0,1,2,3

# Recipe (YAML defaults, CLI overrides take priority)
uv run tengine/train.py --model ms_resnet_cifar110 \
    --recipe configs/ms_resnet/recipes/cifar100_cifar_arch.yaml \
    --dataset cifar100 --data-root /data/twt/datasets

# Family wrappers (DATA_ROOT env var overrides /data/twt/datasets)
bash scripts/train_sewresnet_cifar100.sh 56 0
# Training-time SR-STE 2:4 regularization: --structured-sparse --sr-lambda 0.01 --sr-start-epoch/--sr-end-epoch,
# or configs/spikformer/recipes/{structured_sparse,dynamic_sparse}.yaml (no reported results use it)
```

### Evaluation / transfer
```bash
python tengine/test.py --model sew_resnet_cifar56 --dataset cifar100 --data-root /data/twt/datasets --checkpoint output/.../best.pth
python tengine/test.py --config configs/maxformer/maxformer_cifar.yaml --dataset cifar100 --data-root /data/twt/datasets --checkpoint output/.../best.pth
uv run tengine/transfer.py --config configs/spikingresformer/spikingresformer_ti.yaml \
    --pretrained checkpoints/spikingresformer/ImageNet_spikingresformer_ti.pth \
    --dataset cifar100 --data-root /data/twt/datasets --img-size 128
```

### SBC Pruning (post-training 2:4 sparsification)
```bash
python -m sparse.snn_sbc --model sew_resnet_cifar56 --dense-checkpoint output/.../best.pth \
    --dataset cifar100 --data-root /data/twt/datasets --T 4 --nm 2 4 --evaluate
# Optional: --permute-channels (helps 3x3 Conv, hurts 1x1) and --finetune <epochs> (KD); --config for transformers
# ImageNet: --dataset imagenet --img-size 224 --batch-size 8..32 --calib-batches 400..1600 from the released dense checkpoints (checkpoints/, gitignored, not on this host)

bash scripts/run_sbc_global_cifar100.sh [gpu_id]   # all CIFAR-100 models in output/ -> obc_pt/
bash scripts/finetune_all.sh [gpu_id]              # SBC -> channel permutation -> 1-epoch KD (ResNets), SBC only (transformers) -> obc_pt_ft/
```

### SEngine: build + benchmark
The unified driver runs three phases: export plugin ONNX (TDL) -> fusion-validation pre-pass -> build + benchmark per batch size.
```bash
# --model (factory) or --config (transformer); --fusion none|slicer (comma list = ablation);
# --target auto|ada|a100|orin (default auto-detect); --gpu-ids N masks CUDA_VISIBLE_DEVICES to that GPU
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet \
    --T 4 --batch-sizes 1,4 --fusion slicer --autotune --precision fp16 --gpu-ids 0
# Quick run: drop --autotune (skips validation + tuning). Export only: --export-only

bash scripts/bench_sengine.sh [gpu_id]                                              # model list inside
bash scripts/bench_transformer.sh configs/maxformer/maxformer_10_512.yaml [gpu_id]   # sengine vs TRT; SENGINE_ONLY=1 to skip TRT
```
Lower-level steps:
```bash
python -m sengine.scripts.export_onnx --model sew_resnet18 --dataset cifar100 --checkpoint output/.../best.pth   # -> sengine/exports/{model}_{dataset}_plugin.onnx
python -m sengine.scripts.bench --onnx model_plugin.onnx --T 4 --batch 1 [--save model.sengine --fusion slicer --autotune --precision fp32 --fusion-rec <json>]
python -m sengine.scripts.bench --sengine model.sengine [--trt model.engine --trt-input-shape 4,3,224,224]
python -m sengine.build.fusion_validator --onnx model_plugin.onnx --T 4 --batch 4 --output .cache/fusion_rec.json
python -m sengine.scripts.verify_correctness --model sew_resnet18 --dataset cifar100 --T 4       # random input vs PyTorch
python -m sengine.scripts.bench_cpp --onnx model_plugin.onnx --T 4 --batch 1                       # C++ executor only
python scripts/bench_sengine_cached.py sengine/exports/snn_vgg16_ut_har_plugin.onnx 4 [warmup iters]  # cached kernels + fusion rec, prints a RESULT line
```
Programmatic: `sengine.build(onnx, T=4, batch_size=1, fusion='slicer', autotune=False, fusion_rec=None, precision='fp16', target=None)`, `engine.infer(numpy)`, `engine.benchmark()`, `engine.save()` / `sengine.load()`.

### Baseline Benchmarks
```bash
python scripts/bench_trt_latency.py --model sew_resnet18 --dataset cifar100 --batch-sizes 1,4,8,16   # --sparse for 2:4 ckpt; --export-only / --onnx for cross-platform; --engine-dir
bash scripts/bench_trt.sh [gpu_id]
python -m iengine.backends.tensorrt.benchmark --model sew_resnet34 --dataset cifar100 --data-root /data/twt/datasets --mode {dense|sparse|compare}
python scripts/bench_inductor_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4
python scripts/bench_onnxruntime_latency.py --config configs/spike_bert/spike_bert_small.yaml --dataset sst2 --T 4 --batch-sizes 1
python scripts/bench_tvm_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4 --gpu-ids 0   # needs TVM (not installed); ANN twin, batch B*T
python scripts/acc_compare.py --backend all --checkpoint <dense.pth> --sengine <.sengine> --trt <.engine> --num-images 1600   # ImageNet top-1/5 on identical images

# UT-HAR / UrbanSound8K study (one idle GPU; tools interleaved per B); logs -> output/bench_new_workloads/, CSVs -> results/
bash scripts/bench_new_workloads.sh [gpu_id]        # full pipeline incl. autotune (slow)
bash scripts/remeasure_new_workloads.sh [gpu_id]    # clean re-measurement with cached kernels/engines (+ nsys graph traces)
bash scripts/bench_extra_baselines.sh [gpu_id]      # ONNX Runtime CUDA EP + Inductor max-autotune, fp16/fp32
bash scripts/analyze_new_workloads.sh [gpu_id]      # TRT nsys traces, per-kernel breakdown, TRT fp32 engines -> trt_engines_fp32/
python scripts/parse_bench_new_workloads.py && python scripts/export_new_workload_results.py   # -> output/bench_new_workloads/*_4090.csv (copy to results/)
# Check nvidia-smi utilization before trusting any number: this host is shared and a busy GPU inflates latency 1.5x+.
```

### Engine Correctness vs PyTorch
```bash
# Exports plugin ONNX with the checkpoint weights, builds the engine, compares logits on real test
# samples against PyTorch FP32 and FP16 (model.half()), optionally the whole test split (--eval-acc).
python scripts/verify_workloads.py --engine sengine --model snn_vgg16 --dataset ut_har --T 4 \
    --checkpoint output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth --num-samples 100 --eval-acc --gpu-ids 1
python scripts/verify_workloads.py --engine sengine --target orin --model snn_vgg16 --dataset ut_har --T 4 --checkpoint ... --eval-acc   # Orin profile (functional on x86)
python scripts/verify_workloads.py --engine sengine_cpu --model snn_vgg9 --dataset urbansound8k --T 4 \
    --checkpoint output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth --eval-acc --threads 32
# Strict default: top-1 agreement >= 0.99 vs PyTorch FP16, mean cosine >= 0.99, |acc - acc_fp16| <= 0.5%.
# --fp16-tolerance judges relative to PyTorch's own FP16-vs-FP32 disagreement (low-confidence models such as VGG9/UrbanSound8K).
```

### Jetson / CPU
```bash
# Jetson AGX Orin: export plugin ONNX on x86, build + bench on the Orin with the orin profile
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet --T 4 --export-only
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet --target orin --fusion none,slicer --autotune --batch-sizes 4

# CPU engine (native C backend; NCHW in, (B, classes) out)
python -c "import sengine_cpu; e = sengine_cpu.build('sengine/exports/snn_vgg9_urbansound8k_plugin.onnx', T=4, n_threads=32); print(e.infer(x).shape)"
cd sengine_cpu/csrc && make NO_BLAS=1      # libsengine_cpu.so (cblas_sgemm resolved from scipy's OpenBLAS at runtime)
python scripts/bench_snn_vgg9_cpu.py --threads 1,4,8 [--frameworks sengine,ort] [--numa 0]   # MLAS kernel vs ORT / OpenVINO / ncnn on NTU-Fi (dataset absent here); build: sengine_cpu/BUILD.md
```

### C++ Executor Build
```bash
cd sengine/csrc && make clean && make          # sengine_exec + libsengine_exec.so; CUDA_HOME=/usr/local/cuda-12.8, sm_89 by default; `make lib` = library only
make lib ARCH="-gencode arch=compute_87,code=sm_87" CUDA_HOME=/usr/local/cuda CUDNN_LIB="-L/usr/lib/aarch64-linux-gnu -lcudnn"   # Jetson (JetPack)
```
The Makefile links `-lcudart` from `$(CUDA_HOME)/lib64` explicitly: the ldconfig'd CUDA 11.8 cudart otherwise gets picked up and TileLang aborts at load with "libcudart symbols not found globally" (`ldd libsengine_exec.so` must show `libcudart.so.12`). The library also links system cuDNN 8 (`CUDNN_LIB`). Executor binaries are per-device build products and are never committed; `runtime/cpp_executor.py` loads `csrc/libsengine_exec.so` and otherwise raises with the `make lib` command for the detected arch. TileLang kernel `.so` export auto-detects the arch.

### Profiling
```bash
python scripts/analyze_kernel_latency.py --onnx sengine/exports/maxformer_10_768_imagenet_plugin.onnx --T 4 --batch 4 [--trt-nsys trt_b4.nsys-rep]
bash scripts/profile.sh {nsys|ncu} trt_engines/<model>/<engine>.engine [output_name]        # -> profiles/
python experiments/gpu_util/build_engines.py --gpu-id 3                                     # phase 1 -> experiments/gpu_util/{engines,trt_engines}
sudo bash experiments/gpu_util/profile_one.sh maxformer_10_512 16 both 3                    # phase 2 (ncu needs sudo); run_ncu.sh = full matrix (experiments/gpu_util/config.py)
python experiments/gpu_util/parse_ncu.py                                                    # phase 3 -> experiments/gpu_util/results/
python experiments/breakdown/bench_progressive.py --sengine experiments/gpu_util/engines/sew_resnet101_B32.sengine --gpu-id 2   # fused-kernel speedup decomposition
bash experiments/motivation/run_all.sh [gpu_id]                                             # TRT launch-overhead / temporal-imbalance study -> experiments/motivation/output/
```

### Kernel Smoke Tests
There is no test suite, lint, or CI. These are the closest thing for kernels (each autotunes and compares against cuDNN):
```bash
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 python scripts/bench_grouped_conv_smoke.py
python scripts/bench_winograd_fused_smoke.py
python scripts/bench_winograd_vs_imcol.py
```
For end-to-end numerics use `scripts/verify_workloads.py` (real data) or `sengine/scripts/verify_correctness.py` (random input). After touching `sengine/`, run `verify_workloads.py --engine sengine` on `snn_vgg9`/`urbansound8k` with `--fusion slicer` (fast, cached kernels) before anything else.

## Architecture

### Two model specification paths
- **Factory models**: `--model <name>`. Factory functions in `models/sewresnet.py`, `models/msresnet.py`, `models/snn_vgg.py`, ..., registered in `tengine/utils.py:_RESNET_REGISTRY`: `sew_resnet{18,34,50,101,152}`, `sew_resnet_cifar{20,32,44,56,110}`, `ms_resnet{18,34,50,104}`, `ms_resnet_cifar{20,32,44,56,110}`, `ms_resnet_dvs20`, `dvs_sew_resnet`, `snn_vgg{9,11,16,19}`, `ems_yolo_res34`, `spike_yolo_{n,s,m}`.
- **Config models**: `--config <yaml>`. Builders in `models/__init__.py:ARCH_BUILDERS`: `spikformer`, `metaformer`, `qkformer`, `maxformer`, `ms_qkformer`, `spikingresformer`, `spike_yolo`, `ems_yolo`, `spike_bert`.
- Every driver accepts exactly one of `--model` / `--config`; benchmark outputs are named `{model or config basename}_{dataset}`.

### Config and recipe system
- **Config YAML** (transformers): architecture params (`arch`, `embed_dims`, `num_heads`, `depths`, `patch_size`, ...)
- **Recipe YAML**: training hyperparams (`optimizer`, `scheduler`, `augmentation`, `regularization`, `snn.T`, `epochs`, `batch_size`), loaded via `tengine/utils.py:load_training_recipe()`; explicit CLI args override.
- **Dataset config**: `tengine/utils.py:_DATASET_CONFIG`: `cifar10` (32x32, 10cls), `cifar100`, `imagenet` (224, 1000cls), `cifar10dvs` (128, 2ch, 10cls), `dvs128gesture` (128, 2ch, 11cls), `coco` (640, 80cls, detection), `gen1` (320, 2cls, detection), `sst2`/`mrpc` (NLP, seq_len 128), `ntufi_humanid` (WiFi CSI 3x112x128, 14cls), `ut_har` (WiFi CSI 1x250x90, 7cls), `urbansound8k` (log-mel 1x64x176, 10cls, folds 1-9 train / 10 test).

### Key directories
- **`models/`**: SNN model zoo: neurons (`neurons.py`), layers (`layers.py`: `SeqToANNContainer`), ResNets, VGG (`snn_vgg.py`), Transformers, detection (`spike_yolo.py`, `ems_yolo.py`, `detection_head.py`), NLP (`spike_bert.py`)
- **`models_ann/`**: SNN->ANN conversion (`build_ann_model`, `convert_snn_to_ann`) for topology-identical ANN baselines (TVM / torch.compile / ORT)
- **`snn_datasets/`**: loaders (named to avoid shadowing HuggingFace `datasets`, which `glue.py` imports). UT-HAR `.csv` files are numpy `.npy`; `urbansound8k.py` caches log-mel `.npz` per fold under `UrbanSound8K/cache_logmel{n_mels}x{n_frames}/`
- **`sparse/`**: `sbc.py` (VRD matrix, ExactOBS N:M), `snn_sbc.py` (pipeline), `pruning.py` (magnitude primitives), `utils.py` (neuron detection, weight reshaping)
- **`tengine/`**: `train.py`, `test.py`, `transfer.py`, `dist.py` (DDP), `logger.py`, `utils.py` (model/dataset builders, checkpointing, recipes)
- **`iengine/`**: `backends/tensorrt/` (ONNX export, 2:4 sparse engine build, INT8 calibration, benchmark, `TRTRunner`), `backends/tvm/` (Relax frontend, MetaSchedule)
- **`sengine/`**: GPU inference engine (below); **`sengine_cpu/`**: CPU port (below)
- **`utils/`**: BN fusion / recalibration (`fuse.py`), TRT engine runner for nsys/ncu (`profile_trt_engine.py`)
- **`configs/`**: per-architecture YAML configs and `recipes/`
- **`scripts/`**: `train_*.sh`, `run_sbc_global_cifar100.sh`, `finetune_all.sh`, benchmark drivers `bench_{sengine,trt,inductor,onnxruntime,tvm}_latency.py`, `bench_snn_vgg9_cpu.py`, `verify_workloads.py`, `acc_compare.py`, the new-workload study scripts, `analyze_kernel_latency.py`, `profile.sh`, kernel smoke tests
- **`experiments/`**: `gpu_util/` (ncu utilization matrix; `config.py` holds the model/batch matrix, `results/` the CSVs), `breakdown/` (`bench_progressive.py`: decomposed / fused-PS / interleaved variants), `motivation/` (TensorRT nsys/ncu experiments on Spikformer; `run_all.sh` orchestrates, the other `.py` are ad-hoc probes/plots)
- **`results/`**: tracked CSVs with provenance (`results/README.md`)
- **`docs/`**: `ARCHITECTURE.md` (tracked); `archive/` and PDFs are gitignored local notes (`SYSTEM_DESIGN.md` and `open_questions.md` are early designs that do not describe the current implementation)
- **`output/`**: training runs (`{model}_{dataset}_bs{B}_lr{lr}/best.pth`); **`obc_pt/`**: SBC checkpoints (`*_sbc_2_4_global.pth`, `*_perm_ft1.pth`); **`obc_pt_ft/`**: finetune_all.sh output dir. All gitignored.

### SNN-specific patterns
- **Temporal dimension**: input `(B, C, H, W)`; models reshape to `(T, B, C, H, W)` or `(T*B, C, H, W)` internally.
- **Neuron reset**: `reset_net(model)` after every forward pass; already wired into all loops.
- **Neuron types**: `LIFNeuron`/`IFNeuron`/`ILIFNeuron` wrapped in `MultiStep*` containers; MS-ResNet uses `MSNeuron`; SpikeYOLO uses `ILIFNeuron`; SpikingResformer has its own `LIF`. All obtained via `sparse.utils._get_neuron_types()`.
- **SeqToANNContainer**: merges T and B for stateless ops; the mechanism TDL formalizes.

### SBC pruning flow (sparse/)
1. `collect_smp_hessians()`: forward hooks capture layer inputs, apply VRD matrix M, accumulate `H = 2*(MX)^T*(MX)` per layer
2. `sbc_prune_layer_nm_global()`: ExactOBS, per-row H^-1, element-wise greedy with N:M capacity constraint, OBS compensation + rank-1 H^-1 update
3. Conv2d uses im2col Hessian with columns-last permutation `(K, C*R*S) -> (K, R*S*C)` for TRT-compatible 2:4 along input channels
4. BN recalibration via `utils.fuse.recalibrate_bn()`

### SEngine (sengine/)

**Pipeline**: PyTorch model -> TDL transforms (5D->4D) -> ONNX export (plugin-mode with FusedIF/LIF/MS/ILIF custom ops) -> `ONNXParser` -> `EngineIR` -> optimization passes -> fusion strategy -> `TileLangCompiler` (autotuning cache) -> BA-MTTS scheduling -> memory planning -> buffer planning -> standalone kernel `.so` export -> C++ executor + CUDA Graph capture -> `.sengine` file

**Key components**:
- **`engine.py`**: `SEngine` orchestrator, `build()` / `load()`; C++ CUDA Graph execution path (zero Python in the hot loop); `_detect_arch()` from torch, `_detect_nvcc()` prefers `/usr/local/cuda-12.8`
- **`targets/`**: `TargetProfile` (base.py) + `ada.py` / `a100.py` / `orin.py`; `get_target(name)` resolves explicit name > `SENGINE_TARGET` env > device auto-detect (`Orin`/sm_87 -> orin; sm_80, smem >= 160KB or `A100` -> a100; else ada); `set_active_target()`/`active_target()` registry. Env overrides: `SENGINE_L2_PERSIST=1`, `SENGINE_PREFER_L1=1`. Everything hardware-specific reads from the profile: hw fallback table, tile search spaces and ranking weights (`tuning/roofline.py`, `tuning/analytical.py`, `build/tilelang_compiler.py`), `CalibrationFactors` (`tdl/cost_model.py`), fused-per-T threshold (`optimizer.py`), 3x3 interleaved kernel variant (`t_loop` vs the T=4 hand-unrolled `t_unrolled4` with num_stages=1), L2 persistence + L1 carveout (Orin), kernel cache subdir. Add a GPU by adding a profile, not by branching on device names.
- **`ir.py`**: `OpType`, `KernelVariant`, `BoundType` (COMPUTE/MEMORY), `NeuronType`, `AttentionParams`, `Node`/`Edge`/`FusionGroup`/`EngineIR`
- **`parser.py`**: ONNX parser for plugin-mode custom ops, shape inference, attention op detection
- **`optimizer.py`**: passes in order: BN folding, dead node elimination, fusion group detection, 2:4 sparsity validation, NHWC layout annotation, kernel variant selection, shape propagation + bound classification, edge layout propagation, layout reformat insertion. Stem conv (C_in<4, C_out>=8) -> `TileLangStemConvBN` (IF-free Conv+BN kernel on the unfused path, interleaved Conv+BN+LIF via the slicer)
- **`fusion_strategy.py`** (`none` / `slicer`), **`graph_slicer.py`** (compute-anchored slices absorbing reachable memory-bound successors), **`kernel_codegen.py`** (epilogue templates from `kernels/interleaved_templates.py`)
- **`memory.py`** (lifetime analysis, first-fit pool, 256-byte alignment), **`cuda_graph_runtime.py`** (Python-side CUDA Graph capture; legacy reference for buffer management), **`bound_aware_scheduler.py`** (BA-MTTS)
- **`build/`**: `engine_builder.py`, `sengine_io.py` (.sengine serialization; BN params as named weight blobs, stem C_in padded to 16, attention params), `tilelang_compiler.py` (kernel factory + autotune), `schedule_builder.py`, `tuning_cache.py`, `buffer_planner.py` (buffers + kernel bindings for the C++ executor; `tilelang_5` = Conv/Linear+BN, `tilelang_6` = fused +neuron), `fusion_validator.py`, `export_standalone.py`
- **`kernels/`**: `conv2d_bn_if_t4.py` (Conv+BN+IF interleaved per-T epilogue; stem kernels; T-unrolled 3x3), `spikformer_kernels.py`, `dwconv_bn.py`, `grouped_conv_bn.py`, `fused_attention_kernels.py`, `add_lif_fused.py`, `pool_lif_fused.py`, `winograd_conv.py`, `interleaved_templates.py`
- **`runtime/`**: `plan_executor.py` (dispatch through the C++ executor, L2/L1 setup per profile) and `cpp_executor.py` (ctypes wrapper for `csrc/libsengine_exec.so`)
- **`tdl/`**: `transforms.py` (TDL-1/2/3; `export_with_fused_neurons()`), `neuron_ops.py`, `attention_ops.py`, `ssa_4d.py`/`dssa_4d.py`, `analysis.py`, `cost_model.py`, `temporal_unroll.py`, `slicegraph.py`, `graph_ir.py`, `model_dag/` (one DAG tracer per architecture)
- **`tuning/`**: `analytical.py` (cost model), `hw_calibrate.py` (hardware calibration), `roofline.py` (tile configs)
- **`csrc/`**: `cpp_executor.cu` (library; exports `sengine_setup_l2_persistence`, `sengine_set_prefer_l1`), `sengine_exec.cu` (standalone binary), `kernels.cuh` (native IF/LIF/Add/Pool/TemporalMean/bias kernels), `Makefile` (`CUDA_HOME`, `ARCH`, `CUDA_LIB`, `CUDNN_LIB` overridable; targets `all|lib|clean`)
- **`scripts/`**: `bench.py`, `bench_cpp.py`, `export_onnx.py`, `verify_correctness.py`; **`exports/`**: ONNX and `.sengine` files (gitignored)

**Caches**: `~/.cache/sengine/tuning_cache.json` keyed by (shape, gpu_name, gpu_arch, T, B); `.cache/sengine_B{batch}/` (Orin profile: `.cache/sengine_edge_B{batch}/`) standalone kernel `.so` build dirs (`kern_{nid}_{sha1(source)}.so`, so models share a dir); `.cache/fusion_rec_*.json` recommendations (embed gpu_name/arch and tuning configs).

### TDL (Temporal Dimension Lowering)
- **TDL-1 (T-Axis Absorption)**: patches `SeqToANNContainer` so stateless ops process all timesteps at once
- **TDL-2 (Stateful Extraction)**: replaces neurons with `FusedIFOp`/`FusedLIFOp`/`FusedMSOp`/`FusedILIFOp` custom ops that loop over T internally
- **TDL-3 (Attention Decomposition)**: replaces attention blocks with 4D variants (`SpikformerSSA4D`, `MaxFormerSSA4D`, `TokenQKA4D`, DSSA4D)

### sengine_cpu (sengine_cpu/)
Independent rewrite of the front end for CPUs (own `ir.py` with `CPUKernelVariant`, `optimizer.py`, `scheduler.py`, `build/engine_builder.py`, `.sengine-cpu` file format in `build/sengine_io.py`). FP32, NHWC. Conv backends: **native** (`csrc/native_conv.c`, im2col + `cblas_sgemm` from scipy's OpenBLAS, fused BN + neuron T-loop; default) and optional **TVM** (`build/tvm_compiler.py` + `kernels/{conv_bn_if,add_lif}.py` run in the interpreter from `SENGINE_CPU_TVM_PYTHON`, default `sengine_cpu/.tvm-env/bin/python` created by `env_setup.sh`; only conv1x1 and Add+LIF kernels exist; `SENGINE_CPU_BACKEND=native|tvm`). C executor `csrc/cpu_executor.c` dispatches a BA-MTTS schedule; native ops in `csrc/native_kernels.c`. `engine.infer(x)` takes NCHW and returns `(B, classes)`. `csrc/mlas_conv_bn_lif.cpp` is only for `scripts/bench_snn_vgg9_cpu.py`. Build docs: `sengine_cpu/BUILD.md`.

## Environment Notes

- Python virtualenv at `.venv/` (3.12); `uv pip install` for packages (no sudo). `requirements.txt` pins the stack: torch 2.5.1+cu121 (installed separately), tilelang 0.1.8, spikingjelly 0.0.0.0.14, tensorrt-cu12 10.7, onnx 1.21.0, onnxsim 0.6.2, onnxruntime-gpu 1.25.1, cupy-cuda12x, openvino, ncnn. Use `uv run` or `.venv/bin/python`.
- CUDA 12.8 at `/usr/local/cuda-12.8`; 8x RTX 4090 (sm_89) shared with other users. Check `nvidia-smi` for free memory and utilization and pick an idle GPU before benchmarking; a busy GPU inflates latency 1.5x+. Benchmark scripts export `CUDA_HOME=/usr/local/cuda-12.8`, prepend its `bin` to `PATH`, set `TORCH_CUDA_ARCH_LIST=8.9`; do the same by hand for TileLang code.
- Datasets at `/data/twt/datasets/` (present: cifar10, cifar-100-python, cifar10-dvs, dvs128gesture, imagenet, coco, UT_HAR, UrbanSound8K; absent: NTU-Fi-HumanID, gen1; `UCI-HAR/` is unrelated).
- Nsight Compute at `/opt/nvidia/nsight-compute/2025.1.1/ncu` (ncu scripts need sudo). `compute-sanitizer` must be run from `/usr/local/cuda-12.8/bin/`.
- TVM is not installed on this host (tilelang bundles its own TVM whose `tvm_ffi` conflicts). `scripts/bench_tvm_latency.py` and `iengine/backends/tvm/` need an interpreter with TVM; `sengine_cpu` uses `SENGINE_CPU_TVM_PYTHON`.
- Repo layout changed on 2026-09-21 (branch `consolidate`): `sengine_edge/` was folded into `sengine/targets/orin.py`, `sengine-cpu/` became `sengine_cpu/` (symlink gone), `datasets/` became `snn_datasets/`, `GPUtil/` `breakdown/` `motivation/` moved under `experiments/`. Backups of the pre-consolidation trees: `/home/twt/backup/`.
- Generated outputs (all gitignored): `sengine/exports/`, `trt_engines/`, `trt_engines_fp32/`, `onnxrt_exports/`, `profiles/`, `output/`, `obc_pt/`, `obc_pt_ft/`, `.cache/`, `.sengine_cpu.cache/`, `experiments/gpu_util/{engines,trt_engines,ncu_reports}/`, `third-party/`, executor binaries, `*.so *.onnx *.engine *.sengine *.pth *.nsys-rep *.ncu-rep`.
- Known gaps: `sengine --precision fp32` does not match PyTorch on SNN-VGG16/UT-HAR (fp16 does); `--fusion none` passes only with `--fp16-tolerance` on VGG9/UrbanSound8K; detection and Spike-BERT have no trained checkpoints; the Orin profile is validated functionally on x86 only.
