"""End-to-end TensorRT benchmark for SNN models.

Two pipelines:
  1. Dense FP16:  PyTorch → ONNX → TensorRT (dense)  → evaluate
  2. Sparse FP16: PyTorch → SBC prune → ONNX → TensorRT (sparse) → evaluate

Measures: accuracy (top-1, top-5), latency (ms/batch), throughput (img/s).
Compares dense vs sparse side-by-side when both engines are built.

Usage:
    # Dense baseline
    python -m iengine.tensorrt.benchmark \
        --model sew_resnet_cifar32 --dataset cifar100 --data-root /data/twt/datasets \
        --checkpoint output/.../best.pth --T 4 --mode dense

    # Sparse 2:4 (checkpoint already pruned by SBC)
    python -m iengine.tensorrt.benchmark \
        --model sew_resnet_cifar32 --dataset cifar100 --data-root /data/twt/datasets \
        --checkpoint obc_pt/...sbc_2_4_global.pth --T 4 --mode sparse

    # Both (compare side-by-side)
    python -m iengine.tensorrt.benchmark \
        --model sew_resnet_cifar32 --dataset cifar100 --data-root /data/twt/datasets \
        --dense-checkpoint output/.../best.pth \
        --sparse-checkpoint obc_pt/...sbc_2_4_global.pth \
        --T 4 --mode compare
"""

import argparse
import os
import time

import torch

from models.neurons import reset_net
from tengine.utils import (
    set_seed, build_model_from_config, load_model_config,
    build_model, build_dataloaders, get_dataset_config,
    AverageMeter, accuracy,
)
from iengine.backends.tensorrt.export import export_onnx
from iengine.backends.tensorrt.builder import build_engine
from iengine.backends.tensorrt.runtime import TRTRunner


# ---------------------------------------------------------------------------
# PyTorch baseline evaluation (for reference comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pytorch(model, loader, device, max_batches=0):
    """Evaluate PyTorch model accuracy."""
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for i, (images, targets) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)
        output = model(images)
        reset_net(model)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    return {'acc1': top1.avg, 'acc5': top5.avg}


def evaluate_trt(runner, loader, device_id, max_batches=0):
    """Evaluate TensorRT engine accuracy."""
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    for i, (images, targets) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        images = images.cuda(device_id).float()
        targets = targets.cuda(device_id)
        output = runner.infer(images)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    return {'acc1': top1.avg, 'acc5': top5.avg}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_model(args, ds_cfg, img_size, device):
    """Build and load a model from args."""
    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T
        config['img_size'] = img_size
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T)
    else:
        raise ValueError("Must provide --config or --model")
    return model.to(device).eval()


def _load_checkpoint(model, ckpt_path):
    """Load checkpoint into model."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict)
    return model


def _get_engine_dir(args):
    """Get engine output directory."""
    engine_dir = args.engine_dir
    os.makedirs(engine_dir, exist_ok=True)
    return engine_dir


def _model_tag(args):
    """Get model name tag for file naming."""
    if args.model:
        return args.model
    elif args.config:
        return os.path.splitext(os.path.basename(args.config))[0]
    return "model"


def run_pipeline(
    model, tag, sparse, args, ds_cfg, img_size, val_loader, device_id,
):
    """Run the full export → build → evaluate pipeline for one model.

    Returns dict with accuracy and latency results.
    """
    device = torch.device(f'cuda:{device_id}')
    engine_dir = _get_engine_dir(args)
    mode_str = "sparse" if sparse else "dense"

    onnx_path = os.path.join(engine_dir, f"{tag}_{mode_str}.onnx")
    engine_path = os.path.join(engine_dir, f"{tag}_{mode_str}.engine")

    input_shape = (args.batch_size, ds_cfg['in_channels'], img_size, img_size)

    # Step 1: ONNX export
    print(f"\n--- [{mode_str.upper()}] Step 1: ONNX export ---")
    export_onnx(
        model, onnx_path,
        input_shape=input_shape,
        opset=args.opset,
    )

    # Step 2: Build TensorRT engine
    print(f"\n--- [{mode_str.upper()}] Step 2: Build TensorRT engine ---")
    build_engine(
        onnx_path, engine_path,
        sparse=sparse,
        fp16=True,
        min_batch=1,
        opt_batch=args.batch_size,
        max_batch=args.batch_size,
        workspace_gb=args.workspace_gb,
    )

    # Step 3: Evaluate
    print(f"\n--- [{mode_str.upper()}] Step 3: Evaluate ---")
    with TRTRunner(engine_path, device=device_id) as runner:
        # Accuracy
        acc = evaluate_trt(runner, val_loader, device_id)

        # Latency
        latency = runner.benchmark_latency(
            input_shape=input_shape,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )

    result = {
        'mode': mode_str,
        'acc1': acc['acc1'],
        'acc5': acc['acc5'],
        **latency,
        'engine_path': engine_path,
    }

    print(f"\n  [{mode_str.upper()}] Results:")
    print(f"    Acc@1: {acc['acc1']:.2f}%  Acc@5: {acc['acc5']:.2f}%")
    print(f"    Latency: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms/batch")
    print(f"    Throughput: {latency['throughput_img_s']:.0f} img/s")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='TensorRT benchmark for SNN models (dense vs 2:4 sparse)')
    # Model
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--T', type=int, default=4)
    # Data
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=None)
    # Checkpoints
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for single-mode run (dense or sparse)')
    parser.add_argument('--dense-checkpoint', type=str, default=None,
                        help='Dense checkpoint (for compare mode)')
    parser.add_argument('--sparse-checkpoint', type=str, default=None,
                        help='SBC-pruned checkpoint (for compare mode)')
    # Mode
    parser.add_argument('--mode', type=str, default='dense',
                        choices=['dense', 'sparse', 'compare'],
                        help='Pipeline mode')
    # Engine
    parser.add_argument('--engine-dir', type=str, default='trt_engines')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--workspace-gb', type=float, default=4.0)
    # Benchmark
    parser.add_argument('--n-warmup', type=int, default=20)
    parser.add_argument('--n-measure', type=int, default=100)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    # PyTorch reference
    parser.add_argument('--pytorch-ref', action='store_true',
                        help='Also evaluate PyTorch FP16 as reference')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)

    ds_cfg = get_dataset_config(args.dataset)
    img_size = args.img_size or ds_cfg['img_size']

    _, val_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=img_size, num_workers=4,
    )

    tag = _model_tag(args)
    results = {}

    if args.mode == 'dense':
        ckpt = args.checkpoint or args.dense_checkpoint
        if not ckpt:
            raise ValueError("--checkpoint or --dense-checkpoint required")

        model = _build_model(args, ds_cfg, img_size, device)
        _load_checkpoint(model, ckpt)

        if args.pytorch_ref:
            print("\n--- PyTorch FP16 reference ---")
            model_fp16 = model.half()
            pt_acc = evaluate_pytorch(model_fp16, val_loader, device)
            print(f"  PyTorch Acc@1: {pt_acc['acc1']:.2f}%")
            results['pytorch'] = pt_acc

        results['dense'] = run_pipeline(
            model, tag, sparse=False, args=args,
            ds_cfg=ds_cfg, img_size=img_size,
            val_loader=val_loader, device_id=device_id,
        )

    elif args.mode == 'sparse':
        ckpt = args.checkpoint or args.sparse_checkpoint
        if not ckpt:
            raise ValueError("--checkpoint or --sparse-checkpoint required")

        model = _build_model(args, ds_cfg, img_size, device)
        _load_checkpoint(model, ckpt)

        results['sparse'] = run_pipeline(
            model, tag, sparse=True, args=args,
            ds_cfg=ds_cfg, img_size=img_size,
            val_loader=val_loader, device_id=device_id,
        )

    elif args.mode == 'compare':
        if not args.dense_checkpoint or not args.sparse_checkpoint:
            raise ValueError("compare mode requires --dense-checkpoint and --sparse-checkpoint")

        # Dense
        model_dense = _build_model(args, ds_cfg, img_size, device)
        _load_checkpoint(model_dense, args.dense_checkpoint)

        if args.pytorch_ref:
            print("\n--- PyTorch FP16 reference (dense) ---")
            pt_acc = evaluate_pytorch(model_dense.half(), val_loader, device)
            print(f"  PyTorch Acc@1: {pt_acc['acc1']:.2f}%")
            results['pytorch_dense'] = pt_acc

        results['dense'] = run_pipeline(
            model_dense, tag, sparse=False, args=args,
            ds_cfg=ds_cfg, img_size=img_size,
            val_loader=val_loader, device_id=device_id,
        )
        del model_dense
        torch.cuda.empty_cache()

        # Sparse
        model_sparse = _build_model(args, ds_cfg, img_size, device)
        _load_checkpoint(model_sparse, args.sparse_checkpoint)

        results['sparse'] = run_pipeline(
            model_sparse, tag, sparse=True, args=args,
            ds_cfg=ds_cfg, img_size=img_size,
            val_loader=val_loader, device_id=device_id,
        )

        # Summary
        d = results['dense']
        s = results['sparse']
        speedup = d['mean_ms'] / s['mean_ms'] if s['mean_ms'] > 0 else 0
        acc_diff = s['acc1'] - d['acc1']

        print(f"\n{'='*60}")
        print(f"  COMPARISON: {tag} on {args.dataset}")
        print(f"{'='*60}")
        print(f"  {'':20} {'Dense':>12} {'Sparse 2:4':>12} {'Delta':>12}")
        print(f"  {'Acc@1':20} {d['acc1']:>11.2f}% {s['acc1']:>11.2f}% {acc_diff:>+11.2f}%")
        print(f"  {'Latency (ms)':20} {d['mean_ms']:>12.3f} {s['mean_ms']:>12.3f} {speedup:>11.2f}x")
        print(f"  {'Throughput (img/s)':20} {d['throughput_img_s']:>12.0f} {s['throughput_img_s']:>12.0f}")
        print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
