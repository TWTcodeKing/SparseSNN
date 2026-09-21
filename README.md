# SparseSNN

Post-training 2:4 structured sparsity and a fused inference engine for Spiking Neural Networks.

Two parts:

1. **Spiking Brain Compression (SBC)** (`sparse/`): second-order (OBS-based) N:M pruning
   whose Hessian is built from a Surrogate Membrane Potential that encodes LIF temporal dynamics
   through the Van Rossum Distance convolution matrix. It produces TensorRT-compatible 2:4 sparse
   checkpoints from a trained dense SNN without retraining (optional channel permutation and a
   1-epoch KD fine-tune are available).
2. **sengine** (`sengine/`): an SNN inference engine. Temporal Dimension Lowering (TDL) turns the
   5-D `(T, B, C, H, W)` execution model into a 4-D graph with stateful neuron ops, TileLang
   generates Conv/Linear kernels whose epilogue runs the BN + neuron update per timestep, a
   bound-aware scheduler (BA-MTTS) orders compute- and memory-bound kernels to overlap, and the
   whole schedule is captured in a CUDA Graph driven by a C++ executor. One code base targets
   RTX 4090 (default), A100 and Jetson AGX Orin through target profiles; `sengine_cpu/` is a
   CPU port with native C kernels.

The training stack (`tengine/`, `models/`, `snn_datasets/`) covers spiking ResNets, VGGs and
Transformers (Spikformer, MetaFormer, QKFormer, MaxFormer, SpikingResFormer) on CIFAR, ImageNet,
DVS, WiFi-CSI and audio datasets. TensorRT, torch.compile, ONNX Runtime and TVM baselines are in
`iengine/` and `scripts/`.

## Repository layout

| Path | Content |
|---|---|
| `models/`, `models_ann/` | SNN model zoo (neurons, ResNets, VGG, Transformers, detection, NLP); SNN-to-ANN converter used by the TVM / torch.compile baselines |
| `snn_datasets/` | dataset loaders (CIFAR, ImageNet, CIFAR10-DVS, DVS128 Gesture, COCO, Gen1, GLUE, NTU-Fi HumanID, UT-HAR, UrbanSound8K) |
| `tengine/` | training (`train.py`), evaluation (`test.py`), transfer learning (`transfer.py`), DDP, recipes |
| `configs/` | architecture YAMLs for transformer models and training recipes per dataset |
| `sparse/` | SBC pruning (`python -m sparse.snn_sbc`) |
| `sengine/` | the GPU inference engine (see `docs/ARCHITECTURE.md`) |
| `sengine_cpu/` | CPU engine, native C runtime (`sengine_cpu/BUILD.md`) |
| `iengine/` | TensorRT backend (ONNX export, 2:4 sparse engine build, INT8 calibration) and TVM Relax backend |
| `scripts/` | training wrappers, SBC batch drivers, benchmark drivers for every engine and baseline, correctness checks, kernel smoke tests |
| `experiments/` | profiling studies: `gpu_util/` (ncu utilization matrix), `breakdown/` (fused-kernel speedup decomposition), `motivation/` (TensorRT launch-overhead and temporal-imbalance traces) |
| `results/` | measured CSVs with their provenance |
| `utils/` | BN folding / recalibration, a TensorRT engine runner for nsys/ncu |

Generated artifacts stay out of git: `output/` (training runs), `obc_pt/` (SBC checkpoints),
`sengine/exports/` (ONNX and `.sengine` files), `trt_engines*/`, `onnxrt_exports/`, `profiles/`,
`.cache/` (compiled kernels, fusion recommendations).

## Setup

Python 3.12, CUDA 12.8, an NVIDIA GPU (the kernels were developed on RTX 4090, sm_89).

```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# C++ executor for sengine (sm_89 by default; see the Makefile for other archs)
cd sengine/csrc && make && cd ../..
# CPU engine runtime
cd sengine_cpu/csrc && make NO_BLAS=1 && cd ../..
```

Every benchmark script exports `CUDA_HOME=/usr/local/cuda-12.8`, puts its `bin` on `PATH` and
sets `TORCH_CUDA_ARCH_LIST=8.9`; do the same when calling TileLang code by hand. Nsight Systems
and Nsight Compute are only needed for the profiling scripts.

TVM (`scripts/bench_tvm_latency.py`, `iengine/backends/tvm/`, the optional `sengine_cpu` TVM
backend) is not pip-installable alongside TileLang's bundled TVM; build it in a separate
interpreter and point `SENGINE_CPU_TVM_PYTHON` at it (`sengine_cpu/env_setup.sh` creates one).

### Data

Pass the parent directory as `--data-root`; loaders expect these subdirectories:

| Dataset | Directory |
|---|---|
| CIFAR-10 / CIFAR-100 | `cifar10/`, `cifar-100-python/` (torchvision auto-download) |
| ImageNet | `imagenet/{train,val}` |
| CIFAR10-DVS, DVS128 Gesture | `cifar10-dvs/`, `dvs128gesture/` (SpikingJelly layout; pass the dataset dir itself as `--data-root`) |
| UT-HAR (WiFi CSI) | `UT_HAR/{data,label}` from SenseFi's `UT_HAR.zip` |
| UrbanSound8K | `UrbanSound8K/{metadata,audio}`; log-mel features are cached under `cache_logmel64x176/` on first use |
| NTU-Fi HumanID | `NTU-Fi-HumanID/{train_amp,test_amp}` |
| COCO, Gen1 | `coco/{annotations,train2017,val2017}`, `gen1/{train,val,test}/{images,labels}` |

## Reproduction

Models are named in two ways: factory models with `--model` (`sew_resnet18`,
`sew_resnet_cifar56`, `ms_resnet_cifar110`, `snn_vgg9`, ...) and config models with
`--config <yaml>` (all transformers). Every driver takes exactly one of the two.

### 1. Train a dense model

```bash
uv run tengine/train.py --model sew_resnet_cifar56 --dataset cifar100 --data-root /data/twt/datasets --gpu-ids 0
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml --dataset cifar100 --data-root /data/twt/datasets --gpu-ids 0
uv run tengine/train.py --model ms_resnet_cifar110 --recipe configs/ms_resnet/recipes/cifar100_cifar_arch.yaml --dataset cifar100 --data-root /data/twt/datasets
torchrun --nproc_per_node=4 tengine/train.py --config <yaml> --dataset imagenet --data-root /data/twt/datasets --gpu-ids 0,1,2,3
```

Runs land in `output/{model}_{dataset}_bs{B}_lr{lr}/best.pth`. The `scripts/train_*.sh` wrappers
train whole model families (`DATA_ROOT` env var overrides the default root). Evaluate with
`tengine/test.py`, transfer ImageNet checkpoints with `tengine/transfer.py`.

### 2. SBC 2:4 pruning

```bash
python -m sparse.snn_sbc --model sew_resnet_cifar56 --dense-checkpoint output/.../best.pth \
    --dataset cifar100 --data-root /data/twt/datasets --T 4 --nm 2 4 --evaluate
# --permute-channels (helps 3x3 convs, hurts 1x1), --finetune <epochs> (KD), --config for transformers
bash scripts/run_sbc_global_cifar100.sh [gpu]   # every CIFAR-100 model in output/ -> obc_pt/
bash scripts/finetune_all.sh [gpu]              # SBC -> permutation -> 1-epoch KD (ResNets), SBC only (transformers)
```

ImageNet models were pruned with the same command and `--dataset imagenet --img-size 224
--calib-batches 400..1600` from the authors' released dense checkpoints (SEW-ResNet, MS-ResNet,
MaxFormer, SpikingResFormer), which are not redistributed here.

### 3. Build and benchmark sengine

```bash
# export plugin ONNX (TDL) -> fusion-validation pre-pass -> build + benchmark per batch size
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet --T 4 \
    --batch-sizes 1,4 --fusion slicer --autotune --precision fp16 --gpu-ids 0
# --target {auto,ada,a100,orin} selects the tuning/kernel profile (auto-detected by default)
# --fusion none is the unfused ablation; drop --autotune for a quick build; --export-only stops after ONNX
bash scripts/bench_sengine.sh [gpu]                                            # model list inside
bash scripts/bench_transformer.sh configs/maxformer/maxformer_10_512.yaml [gpu] # sengine vs TensorRT
```

Lower level: `python -m sengine.scripts.export_onnx`, `python -m sengine.scripts.bench --onnx ...
[--save model.sengine]`, `python -m sengine.build.fusion_validator`, `python -m
sengine.scripts.bench_cpp`. Programmatic use:

```python
import sengine
e = sengine.build("sengine/exports/sew_resnet18_imagenet_plugin.onnx", T=4, batch_size=4, fusion="slicer")
y = e.infer(x)          # numpy (B, C, H, W) -> (B, classes)
ms = e.benchmark()
e.save("model.sengine"); e = sengine.load("model.sengine")
```

Compiled kernels are cached in `.cache/sengine_B{batch}/` and tuning results in
`~/.cache/sengine/tuning_cache.json` (keyed by shape, GPU and T, B); delete them to force retuning.

### 4. Correctness against PyTorch

```bash
python scripts/verify_workloads.py --engine sengine --model snn_vgg16 --dataset ut_har --T 4 \
    --checkpoint output/snn_vgg16_ut_har_bs16_lr0.0005/best.pth --num-samples 100 --eval-acc --gpu-ids 0
python scripts/verify_workloads.py --engine sengine_cpu --model snn_vgg9 --dataset urbansound8k --T 4 \
    --checkpoint output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth --eval-acc --threads 32
python -m sengine.scripts.verify_correctness --model sew_resnet18 --dataset cifar100 --T 4   # random input, logits only
```

Strict pass: top-1 agreement >= 0.99 and mean cosine >= 0.99 against PyTorch FP16, test accuracy
within 0.5 points. `--fp16-tolerance` judges relative to PyTorch's own FP16-vs-FP32 disagreement.

### 5. Baselines

```bash
python scripts/bench_trt_latency.py --model sew_resnet18 --dataset cifar100 --batch-sizes 1,4,8,16 [--sparse]
python -m iengine.backends.tensorrt.benchmark --model sew_resnet34 --dataset cifar100 --data-root ... --mode compare  # dense vs 2:4 accuracy + latency
python scripts/bench_inductor_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4
python scripts/bench_onnxruntime_latency.py --config configs/spike_bert/spike_bert_small.yaml --dataset sst2 --T 4 --batch-sizes 1
python scripts/bench_tvm_latency.py --model sew_resnet18 --dataset cifar100 --T 4 --batch-sizes 1,4   # needs TVM; runs the ANN twin with batch B*T
python scripts/acc_compare.py --backend all --checkpoint <dense.pth> --sengine <.sengine> --trt <.engine>  # ImageNet top-1/5, three backends on identical images
```

The UT-HAR / UrbanSound8K study is scripted end to end: `scripts/bench_new_workloads.sh` (full
pipeline with autotuning), `scripts/remeasure_new_workloads.sh` (clean re-measurement with cached
kernels and engines), `scripts/bench_extra_baselines.sh` (ONNX Runtime, Inductor max-autotune),
`scripts/analyze_new_workloads.sh` (nsys traces, per-kernel breakdown, TensorRT FP32), then
`scripts/parse_bench_new_workloads.py` and `scripts/export_new_workload_results.py` produce the
CSVs in `results/`. Check `nvidia-smi` first: a busy GPU inflates every latency.

### 6. CPU and Jetson

```bash
python -c "import sengine_cpu; e = sengine_cpu.build('sengine/exports/snn_vgg9_urbansound8k_plugin.onnx', T=4, n_threads=16)"
python scripts/bench_snn_vgg9_cpu.py --threads 1,4,8 [--frameworks sengine,ort]   # MLAS kernel vs ONNX Runtime / OpenVINO / ncnn
# Jetson AGX Orin: export on x86, build on the device with the orin profile
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet --T 4 --export-only
python scripts/bench_sengine_latency.py --model sew_resnet18 --dataset imagenet --target orin --fusion none,slicer --autotune --batch-sizes 4
```

The Orin needs its own executor build (`make ARCH="-gencode=arch=compute_87,code=sm_87"
CUDA_HOME=/usr/local/cuda` in `sengine/csrc`); a prebuilt sm_87 binary is kept in
`sengine/csrc/prebuilt/`.

### 7. Profiling

```bash
python scripts/analyze_kernel_latency.py --onnx sengine/exports/maxformer_10_768_imagenet_plugin.onnx --T 4 --batch 4 [--trt-nsys trt_b4.nsys-rep]
bash scripts/profile.sh {nsys|ncu} trt_engines/<engine>.engine [name]
python experiments/gpu_util/build_engines.py --gpu-id 0 && sudo bash experiments/gpu_util/run_ncu.sh 0 && python experiments/gpu_util/parse_ncu.py
python experiments/breakdown/bench_progressive.py --sengine experiments/gpu_util/engines/sew_resnet101_B32.sengine --gpu-id 0
bash experiments/motivation/run_all.sh [gpu]
```

Kernel smoke tests (each autotunes and checks against cuDNN): `scripts/bench_grouped_conv_smoke.py`,
`scripts/bench_winograd_fused_smoke.py`, `scripts/bench_winograd_vs_imcol.py`.

## Results

FP16 latency in milliseconds on an idle RTX 4090, T=4 (`results/latency_4090.csv`; TensorRT and
Inductor are the strongest baselines there, ONNX Runtime and FP32 rows are in the file):

| Model / dataset | Batch | sengine | TensorRT | torch.compile |
|---|---|---|---|---|
| SNN-VGG16 / UT-HAR (1x250x90) | 4 | 2.27 | 2.16 | 3.07 |
| | 8 | 4.01 | 4.74 | 6.51 |
| | 16 | 7.28 | 10.21 | 12.89 |
| | 32 | 15.12 | 21.71 | 26.41 |
| SNN-VGG9 / UrbanSound8K (1x64x176) | 4 | 0.61 | 0.52 | 0.65 |
| | 8 | 0.95 | 1.02 | 1.19 |
| | 16 | 1.93 | 2.33 | 2.52 |
| | 32 | 3.65 | 5.39 | 5.49 |

Engine-vs-PyTorch agreement and test accuracy for the same models are in
`results/correctness_4090.csv`; the ncu utilization matrix for MaxFormer-10-512, SEW-ResNet-101
and Spikformer-4-512 is in `experiments/gpu_util/results/`.

## Trained checkpoints on this host

`output/` holds dense checkpoints for CIFAR-100 (SEW/MS-ResNet-32/56/110, MaxFormer, MS-QKFormer,
MetaFormer-8-384), CIFAR10-DVS and DVS128 Gesture (MaxFormer, MS-QKFormer, MS-ResNet-20),
SpikingResFormer-Ti/S transfers, SNN-VGG16 on UT-HAR and SNN-VGG9 on UrbanSound8K. `obc_pt/`
holds the SBC 2:4 outputs (`{model}_{dataset}_sbc_2_4_global.pth`, `_perm_ft1` for permuted +
KD-fine-tuned) including the ImageNet models. Neither directory is in git.

## Known limitations

- `sengine --precision fp32` does not match PyTorch on SNN-VGG16/UT-HAR (FP16 does).
- The `--fusion none` ablation is functional but only validated with `--fp16-tolerance` on
  low-confidence models.
- Detection (SpikeYOLO, EMS-YOLO on COCO/Gen1) and NLP (Spike-BERT on GLUE, needs `transformers`
  and HuggingFace `datasets`) are supported by the model/export stack but have no trained
  checkpoints or reported numbers in this repository.
- The Orin profile is validated functionally on x86 (kernels compile and match PyTorch); latency
  must be measured on the device.
