"""Post-training 2:4 sparse inference benchmark.

Takes a dense or SR-STE trained SNN model, applies hard 2:4 projection,
converts to SparseSemiStructuredTensor (Linear + Conv2d), and benchmarks
latency vs dense fp16 baseline with torch.compile.

Usage:
    python -m iengine.semi_structured.inference \
        --config configs/spikformer/spikformer_cifar.yaml \
        --dataset cifar100 --data-root /home/twt/datasets/ \
        --checkpoint output/.../best.pth \
        --gpu-ids 0 --compile
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
from sparse.pruning import prune_2_4, verify_2_4
from .conversion import (
    _dims_valid_for_semi_structured,
    _should_exclude,
    _make_semi_structured_forward,
    _reshape_pre_hook,
    _reshape_post_hook,
    _DENSE_WEIGHT_ATTR,
    _CONVERTED_FLAG,
    _FP16_HOOK_ATTR,
    convert_to_semi_structured,
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
    model_is_fp16: bool = False,
) -> dict:
    """Convert an SR-STE trained model to semi-structured sparse format.

    Assumes weights are already close to 2:4 from SR-STE training.
    Steps for each eligible layer:
        1. Hard-project to exact 2:4 (in case training ended mid-anneal)
        2. Convert to fp16
        3. Convert to SparseSemiStructuredTensor
        4. If model is fp32: monkey-patch forward for fp16 handling
           If model is fp16: no monkey-patch needed (no dtype cast overhead)

    Args:
        model: SR-STE trained model on CUDA. Modified in-place.
        exclude_names: Module name prefixes to skip.
        exclude_head: If True, exclude modules named 'head'.
        model_is_fp16: If True, skip the monkey-patched forward wrapper
            that does fp32↔fp16 casts.  Set this when the entire model
            has already been converted to fp16 via model.half().

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

        # SparseSemiStructuredTensor requires 2D input — always need reshape.
        # When model is fp16: use lightweight pre/post hooks (no dtype cast,
        #   minimal Python overhead via PyTorch's native hook dispatch).
        # When model is fp32: monkey-patch forward with dtype cast wrapper.
        if model_is_fp16:
            module.register_forward_pre_hook(_reshape_pre_hook)
            module.register_forward_hook(_reshape_post_hook)
        else:
            _original_forward = module.forward
            module.forward = _make_semi_structured_forward(_original_forward)
            setattr(module, _FP16_HOOK_ATTR, _original_forward)

        info['converted'] = True
        info['reason'] = 'success'
        conversion_info[name] = info

    setattr(model, _CONVERTED_FLAG, True)

    return conversion_info


@torch.no_grad()
def _evaluate(model, loader, device, max_samples=None, input_dtype=None):
    """Evaluate model accuracy."""
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    n_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if input_dtype is not None:
            images = images.to(dtype=input_dtype)
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
def _benchmark_latency(model, loader, device, n_warmup=10, n_measure=50,
                       input_dtype=None):
    """Benchmark per-sample inference latency."""
    model.eval()
    # Get a single batch for latency measurement
    images, _ = next(iter(loader))
    images = images.to(device)
    if input_dtype is not None:
        images = images.to(dtype=input_dtype)

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
    use_compile=True,
    fuse_neurons=False,
):
    """Benchmark dense fp16 vs 2:4 sparse fp16 — fair same-dtype comparison.

    SparseSemiStructuredTensor requires fp16 inputs, so both dense and sparse
    baselines run entirely in fp16.

    When use_compile=True (default), both models are wrapped with
    torch.compile to eliminate Python-level overhead from reshape hooks.
    This gives a fair kernel-level comparison.

    Args:
        model: Trained model (already on device, fp32 weights).
        test_loader: Test data loader.
        device: CUDA device.
        max_samples: Max samples for accuracy evaluation.
        exclude_head: Whether to exclude classification head from conversion.
        use_compile: Whether to apply torch.compile to both models.

    Returns:
        Dict with dense and sparse metrics and comparison.
    """
    import copy
    from iengine.common.neuron_utils import maybe_fuse_neurons
    results = {}

    compile_tag = " + torch.compile" if use_compile else ""
    neuron_tag = " + fused neurons" if fuse_neurons else ""

    # ---- Dense fp16 baseline ----
    print(f"--- Dense Baseline (fp16{compile_tag}{neuron_tag}) ---")
    model_dense_fp16 = copy.deepcopy(model).half()
    maybe_fuse_neurons(model_dense_fp16, fuse=fuse_neurons)
    if use_compile:
        model_dense_fp16 = torch.compile(model_dense_fp16)
    dense_acc = _evaluate(model_dense_fp16, test_loader, device,
                          max_samples, input_dtype=torch.float16)
    dense_latency = _benchmark_latency(model_dense_fp16, test_loader, device,
                                       input_dtype=torch.float16)
    results['dense'] = {**dense_acc, 'latency_ms': dense_latency}
    print(f"  Acc@1: {dense_acc['acc1']:.2f}%  |  Latency: {dense_latency:.3f} ms/sample")
    del model_dense_fp16

    # ---- Build sparse fp16 model ----
    print(f"\n--- Applying 2:4 Pruning + Semi-Structured Conversion (fp16{compile_tag}) ---")
    model_sparse = copy.deepcopy(model)

    # convert_to_semi_structured handles both Linear and Conv2d:
    #   - 2:4 magnitude pruning
    #   - fp16 conversion
    #   - SparseSemiStructuredTensor wrapping (Linear) / SparseConv2d replacement (Conv2d)
    model_sparse = model_sparse.to(device)
    conv_info = convert_to_semi_structured(
        model_sparse, exclude_head=exclude_head, convert_conv2d=True)
    n_lin = sum(1 for v in conv_info.values() if v['converted'] and v['type'] == 'Linear')
    n_conv = sum(1 for v in conv_info.values() if v['converted'] and v['type'] == 'Conv2d')
    print(f"  Converted {n_lin} Linear + {n_conv} Conv2d to SparseSemiStructuredTensor")

    # Convert remaining non-sparse layers to fp16 for fair comparison
    # (SparseConv2d and converted Linear are already fp16;
    #  this catches BN, LIF neurons, unconverted layers, etc.)
    model_sparse = model_sparse.half()
    maybe_fuse_neurons(model_sparse, fuse=fuse_neurons)

    if use_compile:
        model_sparse = torch.compile(model_sparse)

    # ---- Sparse fp16 evaluation ----
    print(f"\n--- Sparse 2:4 (fp16{compile_tag}) ---")
    sparse_acc = _evaluate(model_sparse, test_loader, device,
                           max_samples, input_dtype=torch.float16)
    sparse_latency = _benchmark_latency(model_sparse, test_loader, device,
                                        input_dtype=torch.float16)
    results['sparse'] = {**sparse_acc, 'latency_ms': sparse_latency}
    print(f"  Acc@1: {sparse_acc['acc1']:.2f}%  |  Latency: {sparse_latency:.3f} ms/sample")

    del model_sparse

    # ---- Comparison ----
    acc_drop = dense_acc['acc1'] - sparse_acc['acc1']
    speedup = dense_latency / sparse_latency if sparse_latency > 0 else float('inf')

    results['comparison'] = {
        'acc_drop': acc_drop,
        'speedup': speedup,
        'conversion_info': conv_info,
    }

    print(f"\n{'='*60}")
    print(f"  Summary (fp16{compile_tag}{neuron_tag})")
    print(f"{'='*60}")
    print(f"  Dense fp16:      {dense_latency:.3f} ms/sample  Acc: {dense_acc['acc1']:.2f}%")
    print(f"  Sparse fp16:     {sparse_latency:.3f} ms/sample  Acc: {sparse_acc['acc1']:.2f}%")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"  Accuracy drop: {acc_drop:+.2f}%")
    print(f"{'='*60}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark SR-STE structured sparse model with semi-structured acceleration'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Model config YAML path')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. ms_resnet34)')
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
    parser.add_argument('--compile', action='store_true', default=True,
                        help='Use torch.compile to remove hook/reshape overhead '
                             '(default: True)')
    parser.add_argument('--no-compile', dest='compile', action='store_false',
                        help='Disable torch.compile (eager mode)')
    parser.add_argument('--fuse-neurons', action='store_true', default=False,
                        help='Replace LIF/IF neurons with fused Triton kernels')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Build dataloader
    ds_config = get_dataset_config(args.dataset)
    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        num_workers=4, distributed=False,
    )

    # Load model
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_config)
        if args.T is not None:
            config['T'] = args.T
        elif 'T' not in config:
            config['T'] = 4
        model = build_model_from_config(config)
    elif args.model:
        from tengine.utils import build_model
        model = build_model(args.model, num_classes=ds_config['num_classes'],
                            in_channels=ds_config['in_channels'], T=args.T or 4)
    else:
        raise ValueError("Must provide --config or --model")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run benchmark
    print(f"torch.compile: {'enabled' if args.compile else 'disabled'}")
    results = benchmark_structured_sparse(
        model, test_loader, device,
        max_samples=args.max_samples,
        exclude_head=args.exclude_head,
        use_compile=args.compile,
        fuse_neurons=args.fuse_neurons,
    )

    return results


if __name__ == '__main__':
    main()
