# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is a Spiking Neural Network (SNN) training framework implementing multiple SNN architectures with standalone spiking neuron primitives (no spikingjelly dependency). It supports single-GPU and multi-GPU (DDP) training on CIFAR-10, CIFAR-100, ImageNet, and CIFAR-10-DVS.

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
- **`datasets/`** — Dataset loaders returning `(train_loader, val_loader)` tuples. Each dataset has a config in `tengine/utils.py:_DATASET_CONFIG` (includes `num_classes`, `img_size`, `in_channels`).
- **`tengine/`** — Training engine: training loop (`train.py`), evaluation (`test.py`), distributed utils (`dist.py`), logging (`logger.py`), and shared utilities (`utils.py`).
- **`configs/`** — YAML config files for transformer model variants, organized by architecture family. Each arch folder also has a `recipes/` subfolder with per-dataset training recipes.
- **`vis/`** — Visualization and analysis tools. `density_hooks.py` registers forward hooks on Conv/Linear/SSA layers to track spike firing rates as computation density proxies.
- **`iengine/`** — Research documentation on SNN sparsity exploitation and inference engine plans.

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

### Spiking neuron primitives (`models/neurons.py`)

Standalone LIF/IF neurons with surrogate gradients (ATan, Sigmoid, GateGrad). `reset_net(model)` must be called after each forward pass to reset membrane potentials. `MultiStep*` variants process `(T, B, ...)` tensors across timesteps.

### Key tensor convention

Models accept standard `(B, C, H, W)` images. Internally, the temporal dimension is prepended: `(T, B, C, H, W)`. `SeqToANNContainer` flattens `T*B` for spatial ops (Conv/BN), then reshapes back. `SeqToANNContainerT` processes each timestep independently.

### Training flow

`tengine/train.py` orchestrates: parse args → load recipe (if provided) → load YAML config (if transformer) → setup DDP → build dataloaders → merge dataset params into config → build model → train loop with LR scheduler + optional warmup → checkpoint best model to `output/<run_name>/`.

Key training features: AMP (automatic mixed precision), MixUp/CutMix augmentation (`datasets/augmentation.py`), SyncBatchNorm for DDP, and `reset_net(model)` called after every forward pass.

### Default surrogate gradients per architecture
- **Spikformer/QKFormer/MaxFormer:** ATan (alpha=2.0) via MultiStepLIFNeuron
- **MS-ResNet:** GateGrad (lens=0.5) via MultiStepLIFNeuron(tau=4, v_threshold=0.5)
- **SEW-ResNet:** Default ATan via MultiStepIFNeuron(detach_reset=True)
