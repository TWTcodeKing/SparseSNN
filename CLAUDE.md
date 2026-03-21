# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals

SparseSNN targets three key optimizations for SNN inference acceleration:

1. **Accelerate LIF/IF neuron computation** — Spiking neurons (membrane potential update + threshold + reset) are the dominant bottleneck (50-93% of inference latency). Custom fused CUDA/Triton kernels can eliminate per-op overhead by fusing the entire neuron dynamics into a single kernel.

2. **Accelerate SSA (Spiking Self-Attention)** — SSA variants (in `models/spikformer.py`, `models/qkformer.py`, `models/metaformer.py`, `models/maxformer.py`) account for 24-37% of Spikformer inference. Binary spike Q/K/V enable sparse attention optimizations (block-sparse, activation-sparse matmul).

3. **Lossless 2:4 structured weight sparsification** — Convert dense SNN weights to NVIDIA 2:4 Sparse Tensor Core format with minimal accuracy loss. Uses OBS (Optimal Brain Surgeon) with second-order Hessian information for calibration-only conversion (no retraining). Achieves 1.11-1.17x end-to-end speedup; scales with model size and batch size.

## Project Overview

SparseSNN is a Spiking Neural Network (SNN) training and inference framework implementing multiple SNN architectures with standalone spiking neuron primitives (no spikingjelly dependency). It supports single-GPU and multi-GPU (DDP) training on CIFAR-10, CIFAR-100, ImageNet, CIFAR-10-DVS, and DVS128 Gesture.

## Common Commands

### Training — Transformer models (config-based)
```bash
# Single GPU
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --dataset cifar100 --data-root ./data --gpu-ids 0

# Multi-GPU DDP
torchrun --nproc_per_node=4 tengine/train.py \
    --config configs/spikformer/spikformer_8_384.yaml \
    --dataset imagenet --data-root /data/imagenet --gpu-ids 0,1,2,3
```

### Training — ResNet models (direct)
```bash
uv run tengine/train.py --model sew_resnet18 --dataset cifar10 --data-root ./data --gpu-ids 0
```

### Training with structured sparsity
```bash
# SR-STE (static 2:4 regularization)
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --recipe configs/spikformer/recipes/structured_sparse.yaml \
    --dataset cifar100 --data-root ./data --gpu-ids 0

# Dynamic N:M-ceiling sparse training
uv run tengine/train.py --config configs/spikformer/spikformer_8_384.yaml \
    --recipe configs/spikformer/recipes/dynamic_sparse.yaml \
    --dataset cifar100 --data-root ./data --gpu-ids 0
```

### Post-training tools
```bash
# Profile per-channel firing rates (needed for channel permutation)
uv run utils/firing_rate_profile.py --config <yaml> --checkpoint <path> \
    --dataset <name> --data-root <path> --output firing_rates.pt

# Apply channel permutation + optional N:M pruning
uv run sparse/permutation.py --config <yaml> --checkpoint <path> \
    --rates firing_rates.pt --output permuted_model.pth

# Neuron-level firing-rate-aware N:M pruning (finer granularity than channel-level)
uv run sparse/fr_prune.py --config <yaml> --checkpoint <path> \
    --dataset <name> --data-root <path> --lam 0.8 --alpha 0.5 \
    --scoring multiplicative --output pruned_neuron.pth

# Sweep hyperparameters for neuron-aware pruning (profiles once, sweeps lam/alpha/scoring)
uv run sparse/sweep_fr_prune.py --config <yaml> --checkpoint <path> \
    --dataset <name> --data-root <path> --output sweep_results/

# Benchmark weight-sparse inference (dmEngine)
uv run dmEngine/benchmark.py
```

### Evaluation
```bash
uv run tengine/test.py --config configs/spikformer/spikformer_8_384.yaml \
    --dataset cifar10 --data-root ./data --checkpoint ./output/.../best.pth --gpu-ids 0
```

### Environment
Uses `uv` for dependency management. Python virtualenv at `.venv/`.

## Architecture

### Package structure

- **`models/`** — All SNN model implementations. Every model uses standalone neurons from `models/neurons.py` and utility layers from `models/layers.py`.
- **`sparse/`** — All sparsification techniques. `pruning.py` is the canonical N:M pruning module (all pruning imports should come from here). `st_train.py` implements SR-STE regularized training. `permutation.py` provides activation-aware channel permutation for N:M alignment. `comp_2_4.py` implements dense-sparse weight factorization (W = W_24 + W_residual). `fr_prune.py` provides neuron-level (spatial/token-level) firing-rate-aware N:M pruning (pruning logic only — profiling utilities are imported from `utils/profiling.py`); `sweep_fr_prune.py` sweeps its hyperparameters.
- **`datasets/`** — Dataset loaders returning `(train_loader, val_loader)` tuples. Each dataset has a config in `tengine/utils.py:_DATASET_CONFIG` (includes `num_classes`, `img_size`, `in_channels`). Supports both static image datasets and neuromorphic DVS event datasets.
- **`tengine/`** — Training engine: training loop (`train.py`), evaluation (`test.py`), distributed utils (`dist.py`), logging (`logger.py`), and shared utilities (`utils.py`).
- **`configs/`** — YAML config files for transformer model variants, organized by architecture family. Each arch folder also has a `recipes/` subfolder with per-dataset training recipes.
- **`utils/`** — Shared utilities and visualization tools. `profiling.py` is the single source for all firing rate profiling: `ChannelFiringRateProfiler` (channel-level, used for permutation), `NeuronFiringRateProfiler` (spatial/token-level, used for fr_prune scoring), and their convenience wrappers `profile_model_firing_rates` / `profile_neuron_firing_rates`. `density_hooks.py` registers forward hooks on Conv/Linear/SSA layers to track spike firing rates as computation density proxies. `firing_rate_profile.py` is a CLI wrapper around `utils.profiling` that saves firing rates + permutations to `.pt`.
- **`iengine/`** — Inference acceleration backends (semi_structured, torch_sparse, sputnik_sparse, triton_sparse) plus `structured_sparse/semi_structured_path.py` for SR-STE→SparseSemiStructuredTensor conversion and benchmarking. `gather_accumulate/triton_gather_acc.py` exploits SNN activation sparsity: binary spikes reduce `x @ W^T` to gathering/summing weight columns at non-zero positions.
- **`dmEngine/`** — Weight-sparse inference benchmarking framework with pluggable backends (CSR, 2:4 semi-structured, Triton, Sputnik). Uses `dmModels/` for standard ANN ResNet models and random sparsification utilities.

### Model building — two paths

1. **Transformer models (config-based)**: Use `--config <yaml>`. YAML defines architecture params (embed_dims, num_heads, depths, etc.). Runtime params (num_classes, T, img_size, in_channels) are merged from `--dataset` and CLI args. Each model file exposes a single `build_<arch>(config)` function. The `models.ARCH_BUILDERS` dict maps arch names to builders.

2. **ResNet models (direct)**: Use `--model <name>`. Factory functions in `tengine/utils.py:_RESNET_REGISTRY`. Available: `sew_resnet{18,34,50,101,152}` (uses `T` param) and `ms_resnet{18,34,104}` (uses `time_window`).

### YAML config structure
```
configs/<arch>/<variant>.yaml
```
Each YAML has an `arch` field (spikformer, metaformer, qkformer, maxformer) plus model-specific hyperparameters. Dataset-dependent params are NOT in the YAML — they come from `_DATASET_CONFIG` at runtime.

### Training recipes (`configs/<arch>/recipes/`)
Each architecture has per-dataset recipe YAMLs (e.g. `recipes/cifar100.yaml`) defining optimizer, scheduler, augmentation, regularization, and SNN params. `load_training_recipe()` in `tengine/utils.py` flattens the nested YAML into CLI-compatible defaults. Override chain: CLI args > recipe > argparse defaults.

Example recipe structure:
```yaml
optimizer:
  type: adamw
  lr: 1e-4
  weight_decay: 0.06
scheduler:
  type: cosine          # cosine | step | multistep
  warmup_epochs: 10
  min_lr: 1e-6
epochs: 150
batch_size: 128
augmentation:
  auto_aug: true
  mixup_alpha: 0.0
  cutmix_alpha: 0.0
regularization:
  label_smoothing: 0.1
snn:
  T: 4
```

Sparsity-specific recipes add a `structured_sparse` or `dynamic_sparsity` section (see recipe files for full options).

### Spiking neuron primitives (`models/neurons.py`)

Standalone LIF/IF neurons with surrogate gradients (ATan, Sigmoid, GateGrad). `reset_net(model)` must be called after each forward pass to reset membrane potentials. `MultiStep*` variants process `(T, B, ...)` tensors across timesteps. Neuron `tau` and `v_threshold` can be made learnable via config (`learnable_params: true`).

### Key tensor convention

Models accept standard `(B, C, H, W)` images. Internally, the temporal dimension is prepended: `(T, B, C, H, W)`. `SeqToANNContainer` flattens `T*B` for spatial ops (Conv/BN), then reshapes back. `SeqToANNContainerT` processes each timestep independently. DVS datasets directly provide `(T, C, H, W)` tensors per sample.

### Training flow

`tengine/train.py` orchestrates: parse args → load recipe (if provided) → load YAML config (if transformer) → setup DDP → build dataloaders → merge dataset params into config → build model → train loop with LR scheduler + optional warmup → checkpoint best model to `output/<run_name>/`.

Key training features: AMP (automatic mixed precision), MixUp/CutMix augmentation (`datasets/augmentation.py`), SyncBatchNorm for DDP, and `reset_net(model)` called after every forward pass.

When structured sparsity is enabled, the training loop adds SR-STE regularization loss (`loss += sr_ste_regularizer(model, lambda_sr)`) or calls `NMCeilingSparseScheduler.step()` for dynamic mask updates.

### Sparsification pipeline (`sparse/`)

**N:M pruning primitives** (`sparse/pruning.py`): Canonical module for all N:M structured pruning. `prune_n_m(weight, n, m)` is the core function; `prune_2_4` is a convenience alias for the 2:4 case. Also provides `verify_n_m`, `prune_model_linear`, and `prune_n_m_firing_aware` (composite score incorporating upstream firing rates). All pruning imports across the codebase should come from this module.

**SR-STE training** (`sparse/st_train.py`): Two modes for N:M weight sparsity during training:
1. *Static SR-STE*: `ProgressiveSparsityScheduler` anneals `lambda_sr` from 0 to target. Post-training, `apply_hard_n_m_projection()` hard-projects weights in-place.
2. *Dynamic N:M-Ceiling*: Mask-based topology exploration (inspired by SRigL). Alternates prune-by-magnitude and grow-by-gradient steps with cosine-annealed exploration rate.

**Channel permutation** (`sparse/permutation.py`): `compute_permutation_for_n_m()` computes optimal channel ordering so low-firing channels align with pruned positions. `PermutedLinear`/`PermutedConv2d` absorb the permutation into weights. Conv2d layers use NHWC permutation for N:M projection to match NVIDIA Sparse Tensor Core input-channel layout.

**Dense-sparse factorization** (`sparse/comp_2_4.py`): Decomposes W = W_24 + W_residual. The 2:4 component runs through Sparse Tensor Cores; the residual leverages SNN activation sparsity via gather-accumulate.

**Neuron-level firing-rate-aware pruning** (`sparse/fr_prune.py`): Finer-grained alternative to channel-level pruning. `NeuronFiringRateProfiler` hooks Conv `(C,H,W)` and Transformer `(N,C)` layers to capture per-position spike rates. `prune_n_m_neuron_aware` uses a composite score mixing weight magnitude and spatial activation statistics (lam, alpha, scoring=`multiplicative`|`additive`). `sweep_fr_prune.py` profiles once then sweeps hyperparameter combinations, outputting ranked results to CSV.

### Post-training acceleration (`iengine/structured_sparse/`)

`semi_structured_path.py` converts SR-STE trained models to `torch.sparse.SparseSemiStructuredTensor` for NVIDIA Sparse Tensor Core inference. Pipeline: hard-project weights → convert to fp16 → wrap in SparseSemiStructuredTensor → benchmark latency.

### Weight-sparse inference benchmarking (`dmEngine/`)

Backend registry pattern with pluggable backends: `torch_csr` (CSR sparse), `semi_structured` (NVIDIA 2:4 with CUTLASS), `triton_wsparse`, `sputnik_wsparse`. `benchmark.py` sweeps across model × backend × sparsity grid, measuring speedup vs dense baseline. Uses `dmModels/` for standard ANN ResNet models with random unstructured sparsification.

### Default surrogate gradients per architecture
- **Spikformer/QKFormer/MaxFormer:** ATan (alpha=2.0) via MultiStepLIFNeuron
- **MS-ResNet:** GateGrad (lens=0.5) via MultiStepLIFNeuron(tau=4, v_threshold=0.5)
- **SEW-ResNet:** Default ATan via MultiStepIFNeuron(detach_reset=True)
