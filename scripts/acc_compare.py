#!/usr/bin/env python3
"""Accuracy comparison: PyTorch eager (FP32) vs sengine vs TensorRT (FP16).

Evaluates sew_resnet101 (or any SEW-ResNet) on a *fixed, evenly-spaced* subset
of the ImageNet validation set so that all three backends see identical images.

Usage:
    # Eager reference only (validates checkpoint loads + data pipeline)
    python scripts/acc_compare.py --backend eager \
        --checkpoint checkpoints/sewresnet/imagenet/sew101_checkpoint_319.pth \
        --num-images 1600

    # All three backends
    python scripts/acc_compare.py --backend all \
        --checkpoint checkpoints/sewresnet/imagenet/sew101_checkpoint_319.pth \
        --sengine GPUtil/engines/sew_resnet101_B16_trained.sengine \
        --trt GPUtil/trt_engines/sew_resnet101_B16_trained.engine \
        --num-images 4800
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_val_subset_loader(data_root, batch, num_images, num_workers=8):
    """Build a DataLoader over an evenly-spaced subset of ImageNet val.

    Evenly-spaced indices span all 1000 classes (val is sorted by class), giving
    a representative estimate. num_images is rounded down to a multiple of batch.
    """
    val_t = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),   # 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_dir = os.path.join(data_root, 'val')
    full = datasets.ImageFolder(val_dir, transform=val_t)
    total = len(full)

    if num_images <= 0 or num_images >= total:
        n = (total // batch) * batch
        indices = list(range(n))
    else:
        n = (num_images // batch) * batch
        step = total / n
        indices = [int(i * step) for i in range(n)]

    subset = Subset(full, indices)
    loader = DataLoader(subset, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    print(f"  ImageNet val: {total} total -> evaluating {len(indices)} images "
          f"({len(loader)} batches of {batch}, evenly spaced)")
    return loader


def accuracy_counts(logits, targets, topk=(1, 5)):
    """Return number of correct predictions for each k (not percentage)."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)   # (B, maxk)
    pred = pred.t()                               # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).float().sum().item()) for k in topk]


# ─────────────────────────── Eager (PyTorch FP32) ───────────────────────────

def eval_eager(checkpoint, loader, device='cuda:0'):
    from tengine.utils import build_model
    from models.neurons import reset_net

    model = build_model('sew_resnet101', T=4, num_classes=1000, in_channels=3)
    model = model.to(device).eval()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    # num_batches_tracked are harmless BN buffers; report real mismatches loudly
    real_missing = [k for k in missing if 'num_batches_tracked' not in k]
    print(f"  load_state_dict: {len(state_dict)} ckpt tensors | "
          f"missing={len(missing)} (real={len(real_missing)}) | unexpected={len(unexpected)}")
    if real_missing:
        print(f"    !! MISSING (sample): {real_missing[:6]}")
    if unexpected:
        print(f"    !! UNEXPECTED (sample): {unexpected[:6]}")
    if len(real_missing) > 5 or len(unexpected) > 5:
        print("    !! WARNING: large key mismatch — weights may NOT be loaded correctly!")

    n = 0
    c1 = c5 = 0
    t0 = time.time()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            out = model(images)             # (B, 1000), already temporal-mean
            reset_net(model)
            a1, a5 = accuracy_counts(out, targets)
            c1 += a1; c5 += a5; n += targets.size(0)
    dt = time.time() - t0
    return {'top1': 100.0 * c1 / n, 'top5': 100.0 * c5 / n, 'n': n,
            'img_s': n / dt}


# ─────────────────────────────── sengine ───────────────────────────────────

def eval_sengine(sengine_path, loader, batch):
    import sengine
    eng = sengine.load(sengine_path)
    print(f"  sengine loaded: T={eng.T}, batch_size={eng.batch_size}")
    assert eng.batch_size == batch, \
        f"engine batch_size={eng.batch_size} != loader batch={batch}"

    n = 0
    c1 = c5 = 0
    t0 = time.time()
    for images, targets in loader:
        x = images.numpy().astype(np.float32)        # (B,3,224,224) NCHW
        out = eng.infer(x)                            # expect (B,1000)
        out_t = torch.from_numpy(np.asarray(out))
        if out_t.ndim == 2 and out_t.shape[0] == batch * eng.T:
            out_t = out_t.view(eng.T, batch, -1).mean(0)   # safety: external mean
        a1, a5 = accuracy_counts(out_t, targets)
        c1 += a1; c5 += a5; n += targets.size(0)
    dt = time.time() - t0
    eng.destroy()
    return {'top1': 100.0 * c1 / n, 'top5': 100.0 * c5 / n, 'n': n,
            'img_s': n / dt}


# ─────────────────── sengine B=1 (validated Python runtime) ─────────────────

def eval_sengine_b1(plugin_onnx, loader, T=4, max_images=0, precision='fp16'):
    """Evaluate sengine via the validated batch_size=1 Python runtime.

    The prebuilt B16 .sengine engines are latency-only and produce wrong output
    via the C++ paths; the correctness-validated path is EngineBuilder at B=1 with
    capture_graph=False (same as sengine/scripts/verify_correctness.py). FP16
    per-image compute is batch-independent, so B=1 accuracy == B16 accuracy.
    """
    from sengine.build.engine_builder import EngineBuilder
    print(f"  building sengine (B=1, precision={precision}, autotune=False, capture_graph=False) from {plugin_onnx} ...")
    t0 = time.time()
    builder = EngineBuilder(plugin_onnx, T=T, batch_size=1)
    engine = builder.build(autotune=False, capture_graph=False, precision=precision)
    print(f"  built in {time.time()-t0:.0f}s; running per-image inference (this is slow)...")

    n = 0
    c1 = c5 = 0
    t0 = time.time()
    for images, targets in loader:
        for i in range(images.shape[0]):
            x = images[i:i+1].cuda()                 # (1,3,224,224) NCHW
            out = engine(x).float().cpu()            # (1,1000)
            a1, a5 = accuracy_counts(out, targets[i:i+1])
            c1 += a1; c5 += a5; n += 1
            if max_images and n >= max_images:
                break
        if n % 160 == 0 or (max_images and n >= max_images):
            print(f"    {n} imgs | running top-1={100.0*c1/n:.2f}% ({n/(time.time()-t0):.1f} img/s)")
        if max_images and n >= max_images:
            break
    dt = time.time() - t0
    return {'top1': 100.0 * c1 / n, 'top5': 100.0 * c5 / n, 'n': n, 'img_s': n / dt}


# ─────────────────────────────── TensorRT ──────────────────────────────────

def eval_trt(trt_path, loader, batch, device_id=0):
    from iengine.backends.tensorrt.runtime import TRTRunner
    n = 0
    c1 = c5 = 0
    t0 = time.time()
    with TRTRunner(trt_path, device=device_id) as runner:
        for images, targets in loader:
            x = images.to(f'cuda:{device_id}').contiguous()  # (B,3,224,224)
            out = runner.infer(x)                            # (B,1000) torch cuda
            out = out.detach().float().cpu()
            if out.ndim == 2 and out.shape[0] == batch and out.shape[0] != targets.size(0):
                pass
            a1, a5 = accuracy_counts(out, targets)
            c1 += a1; c5 += a5; n += targets.size(0)
    dt = time.time() - t0
    return {'top1': 100.0 * c1 / n, 'top5': 100.0 * c5 / n, 'n': n,
            'img_s': n / dt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['eager', 'sengine', 'sengine_b1', 'trt', 'all'], default='eager')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='dense SEW-ResNet checkpoint (.pth)')
    p.add_argument('--sengine', type=str, default=None)
    p.add_argument('--plugin-onnx', type=str,
                   default='sengine/exports_trained/sew_resnet101_imagenet_plugin.onnx')
    p.add_argument('--trt', type=str, default=None)
    p.add_argument('--data-root', type=str, default='/data/twt/datasets/imagenet')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--num-images', type=int, default=1600)
    p.add_argument('--sengine-max-images', type=int, default=0,
                   help='Cap images for the slow B=1 sengine runtime (0 = same as --num-images)')
    p.add_argument('--sengine-precision', choices=['fp16', 'fp32'], default='fp16')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    print(f"=== ImageNet val subset (data={args.data_root}) ===")
    loader = build_val_subset_loader(args.data_root, args.batch, args.num_images, args.workers)

    results = {}
    if args.backend in ('eager', 'all'):
        print("\n[1] PyTorch eager (FP32)")
        results['eager-fp32'] = eval_eager(args.checkpoint, loader)
    if args.backend == 'sengine':
        assert args.sengine, "--sengine path required"
        print("\n[2] sengine (FP16, prebuilt B16 — latency-only engine)")
        results['sengine-fp16'] = eval_sengine(args.sengine, loader, args.batch)
    if args.backend in ('sengine_b1', 'all'):
        print("\n[2] sengine (FP16, validated B=1 runtime)")
        results[f'sengine-{args.sengine_precision}'] = eval_sengine_b1(
            args.plugin_onnx, loader, T=4, max_images=args.sengine_max_images,
            precision=args.sengine_precision)
    if args.backend in ('trt', 'all'):
        assert args.trt, "--trt path required"
        print("\n[3] TensorRT (FP16)")
        results['trt-fp16'] = eval_trt(args.trt, loader, args.batch)

    print("\n" + "=" * 64)
    print(f"{'backend':<16}{'top-1':>10}{'top-5':>10}{'img/s':>10}{'N':>8}")
    print("-" * 64)
    for k, r in results.items():
        print(f"{k:<16}{r['top1']:>9.3f}%{r['top5']:>9.3f}%{r['img_s']:>10.1f}{r['n']:>8}")
    if 'eager-fp32' in results and len(results) > 1:
        base = results['eager-fp32']['top1']
        print("-" * 64)
        for k, r in results.items():
            if k == 'eager-fp32':
                continue
            print(f"  {k} vs eager-fp32:  Δtop-1 = {r['top1'] - base:+.3f}%")
    print("=" * 64)


if __name__ == '__main__':
    main()
