"""Post-training inference accelerator for SR-STE trained models.

Takes a model trained with SR-STE 2:4 structured sparsity regularization,
applies hard 2:4 projection, converts to SparseSemiStructuredTensor, and
benchmarks latency vs dense baseline.

Usage:
    uv run python -m iengine.structured_sparse.semi_structured_path \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/.../best_sparse.pth \
        --gpu-ids 0
"""

import argparse
import sys
import os
import time

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import reset_net
from sparse.st_train import apply_hard_n_m_projection as apply_hard_2_4_projection
from iengine.semi_structured.pruning import prune_2_4, verify_2_4
from iengine.semi_structured.conversion import (
    _dims_valid_for_semi_structured,
    _should_exclude,
    _make_semi_structured_forward,
    _DENSE_WEIGHT_ATTR,
    _CONVERTED_FLAG,
    _FP16_HOOK_ATTR,
)
from tengine.utils import (
    set_seed,
    build_model_from_config,
    load_model_config,
    build_dataloaders,
    get_dataset_config,
    AverageMeter,
    accuracy,
)


def convert_srste_model(
    model: nn.Module,
    exclude_names: list = None,
    exclude_head: bool = True,
) -> dict:
    """Convert an SR-STE trained model to semi-structured sparse format.

    Assumes weights are already close to 2:4 from SR-STE training.
    Steps for each eligible layer:
        1. Hard-project to exact 2:4 (in case training ended mid-anneal)
        2. Convert to fp16
        3. Convert to SparseSemiStructuredTensor
        4. Monkey-patch forward for fp16 handling

    Args:
        model: SR-STE trained model on CUDA. Modified in-place.
        exclude_names: Module name prefixes to skip.
        exclude_head: If True, exclude modules named 'head'.

    Returns:
        Dict of per-layer conversion info.
    """
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

    # Enable CUTLASS fast path
    SparseSemiStructuredTensor._FORCE_CUTLASS = True

    if exclude_names is None:
        exclude_names = []
    if exclude_head:
        exclude_names = list(exclude_names) + ['head']

    conversion_info = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        info = {
            'shape': tuple(module.weight.shape),
            'dtype_before': str(module.weight.dtype),
            'converted': False,
            'reason': '',
        }

        # Check exclusion
        if _should_exclude(name, exclude_names):
            info['reason'] = 'excluded by name'
            conversion_info[name] = info
            continue

        # Check dimension requirements
        if not _dims_valid_for_semi_structured(module.weight):
            info['reason'] = (
                f'dimensions {tuple(module.weight.shape)} not multiples of 16'
            )
            conversion_info[name] = info
            continue

        with torch.no_grad():
            # Back up original dense weight
            setattr(module, _DENSE_WEIGHT_ATTR, module.weight.data.clone().cpu())

            # Hard-project to 2:4 (in case not already exact)
            w_projected = prune_2_4(module.weight.data)

            # Convert to fp16
            w_fp16 = w_projected.half()

            # Verify 2:4 pattern
            if not verify_2_4(w_fp16):
                info['reason'] = 'failed 2:4 verification after projection'
                conversion_info[name] = info
                continue

            # Convert to SparseSemiStructuredTensor
            module.weight = nn.Parameter(
                to_sparse_semi_structured(w_fp16),
                requires_grad=False,
            )

            # Convert bias to fp16 if present
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.half(),
                    requires_grad=False,
                )

        # Monkey-patch forward for fp32->fp16 input cast and ND->2D flattening
        _original_forward = module.forward
        module.forward = _make_semi_structured_forward(_original_forward)
        setattr(module, _FP16_HOOK_ATTR, _original_forward)

        info['converted'] = True
        info['reason'] = 'success'
        conversion_info[name] = info

    setattr(model, _CONVERTED_FLAG, True)

    return conversion_info


@torch.no_grad()
def _evaluate(model, loader, device, max_samples=None):
    """Evaluate model accuracy."""
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    n_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        reset_net(model)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)

        n_seen += bs
        if max_samples and n_seen >= max_samples:
            break

    return {'acc1': top1.avg, 'acc5': top5.avg}


@torch.no_grad()
def _benchmark_latency(model, loader, device, n_warmup=10, n_measure=50):
    """Benchmark per-sample inference latency."""
    model.eval()
    # Get a single batch for latency measurement
    images, _ = next(iter(loader))
    images = images.to(device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(images)
        reset_net(model)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_measure):
        _ = model(images)
        reset_net(model)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_sample_ms = (elapsed / n_measure / images.size(0)) * 1000
    return per_sample_ms


def benchmark_structured_sparse(
    model,
    test_loader,
    device,
    max_samples=500,
    exclude_head=True,
):
    """Full benchmark: dense baseline vs SR-STE semi-structured sparse.

    Args:
        model: Trained model (already on device, with SR-STE trained weights).
        test_loader: Test data loader.
        device: CUDA device.
        max_samples: Max samples for accuracy evaluation.
        exclude_head: Whether to exclude classification head from conversion.

    Returns:
        Dict with dense and sparse metrics and comparison.
    """
    results = {}

    # ---- Dense baseline ----
    print("--- Dense Baseline ---")
    dense_acc = _evaluate(model, test_loader, device, max_samples)
    dense_latency = _benchmark_latency(model, test_loader, device)
    results['dense'] = {**dense_acc, 'latency_ms': dense_latency}
    print(f"  Acc@1: {dense_acc['acc1']:.2f}%  |  Latency: {dense_latency:.3f} ms/sample")

    # ---- Hard project + convert to semi-structured ----
    print("\n--- Applying SR-STE Hard Projection + Semi-Structured Conversion ---")
    proj_stats = apply_hard_2_4_projection(model)
    n_projected = sum(1 for v in proj_stats.values() if v['projected'])
    print(f"  Hard-projected {n_projected} Linear layers to 2:4")

    conv_info = convert_srste_model(model, exclude_head=exclude_head)
    n_converted = sum(1 for v in conv_info.values() if v['converted'])
    print(f"  Converted {n_converted} layers to SparseSemiStructuredTensor")

    # ---- Sparse evaluation ----
    print("\n--- Sparse (2:4 Semi-Structured) ---")
    sparse_acc = _evaluate(model, test_loader, device, max_samples)
    sparse_latency = _benchmark_latency(model, test_loader, device)
    results['sparse'] = {**sparse_acc, 'latency_ms': sparse_latency}
    print(f"  Acc@1: {sparse_acc['acc1']:.2f}%  |  Latency: {sparse_latency:.3f} ms/sample")

    # ---- Comparison ----
    acc_drop = dense_acc['acc1'] - sparse_acc['acc1']
    speedup = dense_latency / sparse_latency if sparse_latency > 0 else float('inf')
    results['comparison'] = {
        'acc_drop': acc_drop,
        'speedup': speedup,
        'projection_stats': proj_stats,
        'conversion_info': conv_info,
    }

    print(f"\n--- Comparison ---")
    print(f"  Accuracy drop: {acc_drop:+.2f}%")
    print(f"  Speedup:       {speedup:.2f}x")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark SR-STE structured sparse model with semi-structured acceleration'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Model config YAML path')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'imagenet', 'cifar10dvs'])
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root path for dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SR-STE trained checkpoint (.pth)')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-head', action='store_true', default=True)
    parser.add_argument('--no-exclude-head', dest='exclude_head',
                        action='store_false')
    parser.add_argument('--T', type=int, default=None,
                        help='Override number of timesteps')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Build dataloader
    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        num_workers=4, distributed=False,
    )

    # Load model
    config = load_model_config(args.config)
    ds_config = get_dataset_config(args.dataset)
    config.update(ds_config)
    if args.T is not None:
        config['T'] = args.T
    elif 'T' not in config:
        config['T'] = 4

    model = build_model_from_config(config)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run benchmark
    results = benchmark_structured_sparse(
        model, test_loader, device,
        max_samples=args.max_samples,
        exclude_head=args.exclude_head,
    )

    return results


if __name__ == '__main__':
    main()
