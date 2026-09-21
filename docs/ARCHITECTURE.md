# Architecture

## Model specification

Two paths, accepted by every driver (train / test / SBC / export / bench):

- **Factory models**, `--model <name>`: factory functions in `models/sewresnet.py`,
  `models/msresnet.py`, `models/snn_vgg.py`, ... registered in `tengine/utils.py:_RESNET_REGISTRY`
  (`sew_resnet{18,34,50,101,152}`, `sew_resnet_cifar{20,32,44,56,110}`, `ms_resnet{18,34,50,104}`,
  `ms_resnet_cifar{20,...,110}`, `ms_resnet_dvs20`, `dvs_sew_resnet`, `snn_vgg{9,11,16,19}`,
  `ems_yolo_res34`, `spike_yolo_{n,s,m}`).
- **Config models**, `--config <yaml>`: builders in `models/__init__.py:ARCH_BUILDERS`
  (`spikformer`, `metaformer`, `qkformer`, `maxformer`, `ms_qkformer`, `spikingresformer`,
  `spike_yolo`, `ems_yolo`, `spike_bert`). The YAML carries architecture parameters; a **recipe**
  YAML (`configs/*/recipes/`) carries training hyper-parameters and is loaded by
  `tengine/utils.py:load_training_recipe()`, with explicit CLI flags taking priority.
- Dataset shapes and class counts are hard-coded in `tengine/utils.py:_DATASET_CONFIG`.

SNN conventions: input `(B, C, H, W)` is expanded to `T` timesteps inside the model
(`(T, B, C, H, W)` or `(T*B, C, H, W)`). Stateless layers are wrapped in
`models/layers.py:SeqToANNContainer`, which merges T and B. Neurons are `LIFNeuron` / `IFNeuron` /
`ILIFNeuron` wrapped in `MultiStep*` containers (MS-ResNet uses `MSNeuron`, SpikingResFormer its
own `LIF`); `reset_net(model)` clears membrane state after each forward and is wired into every
loop. `models_ann/` converts an SNN into a topology-identical ANN (neurons to ReLU, T dropped)
for the TVM / torch.compile / ONNX Runtime baselines.

## SBC pruning (`sparse/`)

1. `collect_smp_hessians()`: forward hooks capture each layer's input, apply the Van Rossum
   Distance matrix `M` along T and accumulate `H = 2 (MX)^T (MX)` per layer.
2. `sbc_prune_layer_nm_global()`: ExactOBS with a per-row `H^-1`, greedy element selection under
   the N:M capacity constraint, OBS weight compensation and a rank-1 `H^-1` update.
3. Conv2d weights use the im2col Hessian with a columns-last permutation `(K, C*R*S) -> (K, R*S*C)`
   so that 2:4 groups run along input channels, as TensorRT requires.
4. BatchNorm statistics are recalibrated (`utils/fuse.py:recalibrate_bn()`); optional channel
   permutation (`--permute-channels`) and a KD fine-tune (`--finetune`) follow.

## sengine: GPU inference engine (`sengine/`)

Pipeline:

```
PyTorch model -> TDL transforms -> plugin ONNX -> ONNXParser -> EngineIR -> optimizer passes
  -> fusion strategy -> TileLangCompiler (autotuning cache) -> BA-MTTS schedule -> memory plan
  -> buffer plan -> standalone kernel .so -> C++ executor + CUDA Graph -> .sengine file
```

- **TDL, Temporal Dimension Lowering** (`tdl/`): TDL-1 absorbs the T axis into stateless ops
  (patches `SeqToANNContainer`), TDL-2 extracts stateful neurons into `FusedIFOp` /
  `FusedLIFOp` / `FusedMSOp` / `FusedILIFOp` custom ops that loop over T internally, TDL-3
  replaces attention blocks by 4-D variants (`ssa_4d.py`, `dssa_4d.py`). `model_dag/` holds one
  DAG tracer per architecture; `transforms.py:export_with_fused_neurons()` produces the ONNX.
- **IR** (`ir.py`): `OpType`, `KernelVariant`, `BoundType` (COMPUTE / MEMORY), `NeuronType`,
  `Node` / `Edge` / `FusionGroup` / `EngineIR`. `parser.py` builds it from the plugin ONNX.
- **Optimizer** (`optimizer.py`), in order: BN folding, dead-node elimination, fusion-group
  detection, 2:4 sparsity validation, NHWC layout annotation, kernel-variant selection, shape
  propagation and bound classification, per-edge layout propagation, reformat insertion.
- **Fusion** (`fusion_strategy.py`, `graph_slicer.py`, `kernel_codegen.py`): `none` keeps every
  op decomposed (ablation); `slicer` anchors each slice on a compute-bound op and absorbs the
  memory-bound successors reachable from it, then instantiates an epilogue template from
  `kernels/interleaved_templates.py`. `build/fusion_validator.py` profiles fused against
  decomposed per shape and writes a recommendation JSON that the build reuses.
- **Kernels** (`kernels/`): TileLang Conv+BN+neuron with an interleaved per-timestep epilogue
  (`conv2d_bn_if_t4.py`, including the padded-C_in stem and a T-unrolled 3x3 variant),
  1x1 / depthwise / grouped conv, Linear+BN(+LIF), attention (`fused_attention_kernels.py`,
  `spikformer_kernels.py`), Add+LIF, Pool+LIF, Winograd F(2,3) for small 3x3 convs.
- **Scheduling** (`bound_aware_scheduler.py`, `build/schedule_builder.py`): BA-MTTS classifies
  ops as compute- or memory-bound and picks a topological order that maximizes C/M transitions so
  consecutive kernels overlap on the hardware.
- **Memory and execution** (`memory.py`, `build/buffer_planner.py`, `build/export_standalone.py`,
  `csrc/`, `runtime/`): lifetime analysis with first-fit pooling, GPU buffer bindings for the
  C++ executor, kernels exported as standalone `.so` files (no TVM runtime dependency), native
  CUDA kernels for IF / LIF / Add / Pool / TemporalMean in `csrc/kernels.cuh`, CUDA Graph capture
  and replay in `csrc/cpp_executor.cu` (loaded through ctypes by `runtime/cpp_executor.py`;
  `runtime/plan_executor.py` is the Python-dispatch reference path). `build/sengine_io.py`
  serializes everything into a `.sengine` file.
- **Tuning** (`tuning/`): analytical cost model, hardware calibration, roofline-driven tile
  configurations; results are cached in `~/.cache/sengine/tuning_cache.json`.
- **Targets** (`targets/`): a `TargetProfile` carries everything that differs between GPUs:
  SM arch, hardware fallback table, tile search space and ranking weights, cost-model calibration,
  the 3x3 kernel variant, and the Orin-only L2 persistence / L1 carveout switches. Profiles exist
  for RTX 4090 (`ada`, default), A100 (`a100`) and Jetson AGX Orin (`orin`); selection is
  explicit (`target=` / `--target`), by `SENGINE_TARGET`, or auto-detected from the device.
  `SENGINE_L2_PERSIST=1` / `SENGINE_PREFER_L1=1` force the Orin runtime switches on any target.

Caches: compiled kernel `.so` files in `.cache/sengine_B{batch}/` (Orin profile:
`.cache/sengine_edge_B{batch}/`), fusion recommendations in `.cache/fusion_rec_*.json`.

## sengine_cpu (`sengine_cpu/`)

Same front end (plugin ONNX, IR, optimizer, slicer, BA-MTTS) rewritten for CPUs: FP32
throughout, NHWC activations, a C executor (`csrc/cpu_executor.c`) dispatching native kernels
(`csrc/native_kernels.c`: IF/LIF, Add, MaxPool, GAP, TemporalMean, GEMM+bias, Softmax) and a
native conv backend (`csrc/native_conv.c`: NHWC im2col + `cblas_sgemm` resolved at runtime from
scipy's bundled OpenBLAS, fused BN + neuron T-loop). An optional TVM backend generates conv1x1 and
Add+LIF kernels in a separate interpreter (`SENGINE_CPU_TVM_PYTHON`, `SENGINE_CPU_BACKEND=tvm`).
`csrc/mlas_conv_bn_lif.cpp` is a standalone fused MLAS kernel used only by
`scripts/bench_snn_vgg9_cpu.py`. Build notes: `sengine_cpu/BUILD.md`.

## Baselines (`iengine/`, `scripts/`)

- `iengine/backends/tensorrt/`: standard ONNX export (no TDL), engine build with 2:4 sparsity and
  INT8 calibration, `TRTRunner`, dense-vs-sparse accuracy benchmark.
- `iengine/backends/tvm/`: Relax front end and MetaSchedule tuning for the ANN twin.
- `scripts/bench_{trt,inductor,onnxruntime,tvm}_latency.py`: latency drivers with the same
  `--model/--config`, `--dataset`, `--T`, `--batch-sizes` interface as the sengine driver.
