#!/usr/bin/env python3
"""End-to-end correctness check of an engine (sengine / sengine_edge / sengine_cpu)
against the PyTorch model on real dataset samples.

For each of N test samples the script compares the engine logits with the
PyTorch FP32 and FP16 (model.half()) outputs: top-1 agreement, cosine
similarity, max abs diff.  Optionally (--eval-acc) it also runs the whole test
split through the engine and reports engine vs PyTorch accuracy.

Usage:
    python scripts/verify_workloads.py --engine sengine --model snn_vgg16 \
        --dataset ut_har --T 4 --checkpoint output/snn_vgg16_ut_har_bs64_lr0.001/best.pth \
        --num-samples 100 --eval-acc --gpu-ids 1
    python scripts/verify_workloads.py --engine sengine_cpu --model snn_vgg9 \
        --dataset urbansound8k --T 4 --checkpoint ... --threads 16
    python scripts/verify_workloads.py --engine sengine_edge --model snn_vgg16 --dataset ut_har --T 4

Pass criteria (per engine): top-1 agreement with the PyTorch FP16 reference
>= --min-agree (default 0.99) and mean cosine >= --min-cos (default 0.99).
"""
import argparse
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

for _p in ['/usr/local/cuda-12.8', '/usr/local/cuda']:
    if os.path.isdir(_p):
        os.environ.setdefault('CUDA_HOME', _p)
        os.environ['PATH'] = os.path.join(_p, 'bin') + ':' + os.environ.get('PATH', '')
        break
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn.functional as F


ENGINES = ('sengine', 'sengine_edge', 'sengine_cpu')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', required=True, choices=ENGINES)
    p.add_argument('--model', type=str, default=None, help='factory model name (e.g. snn_vgg16)')
    p.add_argument('--config', type=str, default=None, help='transformer YAML config')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--data-root', type=str, default='/data/twt/datasets')
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--img-size', type=str, default=None, help='H or H,W (default: dataset config)')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--eval-batch', type=int, default=16, help='PyTorch reference batch size')
    p.add_argument('--random-input', action='store_true', help='use N(0,1) inputs instead of dataset samples')
    p.add_argument('--eval-acc', action='store_true', help='also compare full test-split accuracy')
    p.add_argument('--fusion', type=str, default='slicer', choices=['none', 'slicer'])
    p.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'])
    p.add_argument('--autotune', action='store_true')
    p.add_argument('--threads', type=int, default=0, help='sengine_cpu thread count (0 = all)')
    p.add_argument('--engine-batch', type=int, default=1, help='engine batch size (samples per infer call)')
    p.add_argument('--export-dir', type=str, default=None)
    p.add_argument('--gpu-ids', type=str, default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-agree', type=float, default=0.99)
    p.add_argument('--min-cos', type=float, default=0.99)
    p.add_argument('--fp16-tolerance', action='store_true',
                   help='pass criterion relative to PyTorch FP16-vs-FP32 disagreement: '
                        'agree >= (ref16-vs-ref32 agree - 0.01) and |acc - acc_fp32| <= 1%%')
    return p.parse_args()


def parse_img_size(s, default):
    if s is None:
        return default
    if ',' in s:
        h, w = s.split(',')
        return (int(h), int(w))
    return int(s)


def build_reference_model(args, ds_cfg, img_size, device):
    from tengine.utils import build_model, build_model_from_config, load_model_config
    num_classes, in_channels = ds_cfg['num_classes'], ds_cfg['in_channels']
    if args.config:
        cfg = load_model_config(args.config)
        cfg.update({'num_classes': num_classes, 'T': args.T, 'in_channels': in_channels,
                    'img_size': img_size[0] if isinstance(img_size, tuple) else img_size})
        model = build_model_from_config(cfg)
    else:
        model = build_model(args.model, T=args.T, num_classes=num_classes, in_channels=in_channels)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print("  No checkpoint: random weights")
    return model.to(device).eval()


def collect_samples(args, ds_cfg, img_size):
    """Return (list of (1,C,H,W) float32 numpy, labels) for the comparison set."""
    C = ds_cfg['in_channels']
    H, W = img_size if isinstance(img_size, tuple) else (img_size, img_size)
    if args.random_input:
        rng = np.random.RandomState(args.seed)
        xs = [rng.randn(1, C, H, W).astype(np.float32) for _ in range(args.num_samples)]
        return xs, [None] * len(xs), None
    from tengine.utils import build_dataloaders
    _, test_loader = build_dataloaders(args.dataset, args.data_root, args.eval_batch, img_size=img_size,
                                       num_workers=2, T=args.T)
    xs, ys = [], []
    for xb, yb in test_loader:
        for i in range(xb.shape[0]):
            if xb.dim() == 5:  # (B, T, C, H, W) datasets -> use frame stack as-is
                raise SystemExit("5D (per-timestep) inputs are not supported by this checker")
            xs.append(xb[i:i + 1].numpy().astype(np.float32))
            ys.append(int(yb[i]))
            if len(xs) >= args.num_samples:
                break
        if len(xs) >= args.num_samples:
            break
    return xs, ys, test_loader


@torch.no_grad()
def torch_forward(model, x_np, device, half=False):
    from models.neurons import reset_net
    reset_net(model)
    x = torch.from_numpy(x_np).to(device)
    if half:
        out = model(x.half())
    else:
        out = model(x)
    reset_net(model)
    return out.float().cpu().numpy()


def export_onnx(args, ds_cfg, img_size, state_dict):
    """Export plugin ONNX with the reference weights. Returns path."""
    pkg = 'sengine_edge' if args.engine == 'sengine_edge' else 'sengine'
    export_dir = args.export_dir or os.path.join(pkg, 'exports')
    os.makedirs(export_dir, exist_ok=True)
    tmp_ckpt = os.path.join(tempfile.gettempdir(), f'verify_{os.getpid()}.pth')
    torch.save({'model': state_dict}, tmp_ckpt)
    name = args.model or os.path.splitext(os.path.basename(args.config))[0]
    plugin = os.path.join(export_dir, f"{name}_{args.dataset}_plugin.onnx")
    if os.path.exists(plugin):
        os.remove(plugin)  # always re-export with the reference weights
    mod = __import__(f'{pkg}.scripts.export_onnx', fromlist=['export_model'])
    path = mod.export_model(model_name=name, output_dir=export_dir, T=args.T,
                            dataset=args.dataset, img_size=img_size,
                            config=args.config, checkpoint=tmp_ckpt)
    os.remove(tmp_ckpt)
    if not path:
        raise SystemExit("ONNX export failed")
    return path


def build_engine(args, onnx_path):
    if args.engine == 'sengine_cpu':
        import sengine_cpu
        return sengine_cpu.build(onnx_path, T=args.T, batch_size=args.engine_batch, n_threads=args.threads)
    pkg = __import__(args.engine)
    return pkg.build(onnx_path, T=args.T, batch_size=args.engine_batch, fusion=args.fusion,
                     autotune=args.autotune, precision=args.precision)


def metrics(a, b):
    a = torch.from_numpy(np.asarray(a, dtype=np.float32)).flatten()
    b = torch.from_numpy(np.asarray(b, dtype=np.float32)).flatten()
    cos = F.cosine_similarity(a[None], b[None]).item() if a.norm() > 0 and b.norm() > 0 else 0.0
    return {'cos': cos, 'max_abs': (a - b).abs().max().item(),
            'top1': int(a.argmax().item() == b.argmax().item())}


def main():
    args = parse_args()
    if not (args.model or args.config):
        raise SystemExit("--model or --config required")
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_ids)
    torch.manual_seed(args.seed)
    from tengine.utils import get_dataset_config
    ds_cfg = get_dataset_config(args.dataset)
    img_size = parse_img_size(args.img_size, ds_cfg['img_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = args.model or os.path.basename(args.config)
    print(f"=== verify {args.engine}: {name} / {args.dataset} T={args.T} img={img_size} "
          f"C={ds_cfg['in_channels']} classes={ds_cfg['num_classes']} ===")

    # 1. Reference model + all PyTorch outputs FIRST (TDL export mutates model classes)
    model = build_reference_model(args, ds_cfg, img_size, device)
    xs, ys, test_loader = collect_samples(args, ds_cfg, img_size)
    print(f"  Comparison set: {len(xs)} samples")
    ref32 = [torch_forward(model, x, device, half=False) for x in xs]
    model_half = model.half()
    ref16 = [torch_forward(model_half, x, device, half=True) for x in xs]
    model = model.float()

    acc_ref32 = acc_ref16 = None
    if args.eval_acc and test_loader is not None:
        c32 = c16 = n = 0
        for xb, yb in test_loader:
            o32 = torch_forward(model, xb.numpy().astype(np.float32), device)
            o16 = torch_forward(model.half(), xb.numpy().astype(np.float32), device, half=True)
            model = model.float()
            c32 += int((o32.argmax(1) == yb.numpy()).sum()); c16 += int((o16.argmax(1) == yb.numpy()).sum())
            n += len(yb)
        acc_ref32, acc_ref16 = c32 / n, c16 / n
        print(f"  PyTorch test acc: fp32={acc_ref32:.4f} fp16={acc_ref16:.4f} (n={n})")
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 2. Export + build engine
    t0 = time.time()
    onnx_path = export_onnx(args, ds_cfg, img_size, state_dict)
    print(f"  ONNX: {onnx_path}")
    engine = build_engine(args, onnx_path)
    print(f"  Engine built in {time.time() - t0:.1f}s")

    # 3. Per-sample comparison
    rows = []
    EB = args.engine_batch
    outs = []
    for i in range(0, len(xs), EB):
        chunk = xs[i:i + EB]
        if len(chunk) < EB:  # pad the last chunk to the engine batch
            chunk = chunk + [chunk[-1]] * (EB - len(chunk))
        xb_np = np.concatenate(chunk, axis=0)
        ob = np.asarray(engine.infer(xb_np), dtype=np.float32).reshape(EB, -1)
        outs.extend([ob[j:j + 1] for j in range(min(EB, len(xs) - i))])
    for i, x in enumerate(xs):
        out = outs[i]
        m32, m16 = metrics(out, ref32[i]), metrics(out, ref16[i])
        rows.append((m32, m16))
        if not m16['top1']:
            o = np.sort(out.reshape(-1))[::-1]; r = np.sort(ref16[i].reshape(-1))[::-1]
            print(f"  MISMATCH sample {i}: engine top1={out.argmax()} (margin {o[0]-o[1]:.3f}) "
                  f"ref16 top1={ref16[i].argmax()} (margin {r[0]-r[1]:.3f}) label={ys[i]}")
        if i < 3:
            print(f"  sample {i}: engine top1={out.argmax()} ref32 top1={ref32[i].argmax()} "
                  f"cos32={m32['cos']:.4f} cos16={m16['cos']:.4f} maxabs16={m16['max_abs']:.4f}")
    agree32 = np.mean([r[0]['top1'] for r in rows]); agree16 = np.mean([r[1]['top1'] for r in rows])
    cos32 = np.mean([r[0]['cos'] for r in rows]); cos16 = np.mean([r[1]['cos'] for r in rows])
    mincos16 = np.min([r[1]['cos'] for r in rows]); maxabs16 = np.max([r[1]['max_abs'] for r in rows])

    # 4. Optional full test accuracy through the engine
    acc_eng = None
    if args.eval_acc and test_loader is not None:
        correct = n = 0
        t0 = time.time()
        for xb, yb in test_loader:
            xb_np = xb.numpy().astype(np.float32)
            for j in range(0, xb.shape[0], EB):
                chunk = xb_np[j:j + EB]
                valid = chunk.shape[0]
                if valid < EB:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], EB - valid, axis=0)], axis=0)
                ob = np.asarray(engine.infer(chunk), dtype=np.float32).reshape(EB, -1)
                correct += int((ob[:valid].argmax(1) == yb[j:j + valid].numpy()).sum()); n += valid
        acc_eng = correct / n
        print(f"  Engine test acc: {acc_eng:.4f} (n={n}, {time.time() - t0:.1f}s)")

    # 5. Summary
    print("\n  ---------------- summary ----------------")
    print(f"  engine={args.engine} model={name} dataset={args.dataset} T={args.T} precision={args.precision}")
    agree_1632 = np.mean([int(a.argmax() == b.argmax()) for a, b in zip(ref16, ref32)])
    print(f"  PyTorch FP16 vs FP32 (reference tolerance): top1 agree={agree_1632:.4f}")
    print(f"  vs PyTorch FP32: top1 agree={agree32:.4f} mean cos={cos32:.4f}")
    print(f"  vs PyTorch FP16: top1 agree={agree16:.4f} mean cos={cos16:.4f} min cos={mincos16:.4f} max|d|={maxabs16:.4f}")
    if acc_eng is not None:
        print(f"  test acc: engine={acc_eng:.4f} torch_fp32={acc_ref32:.4f} torch_fp16={acc_ref16:.4f}")
    if args.fp16_tolerance:
        need_agree = min(args.min_agree, agree_1632 - 0.01)
        ok = agree16 >= need_agree and cos16 >= args.min_cos
        if acc_eng is not None:
            ok = ok and abs(acc_eng - acc_ref32) <= 0.01 + 1e-9
        crit = f"agree>={need_agree:.2f} (fp16 tolerance), cos>={args.min_cos}" + (", |acc-acc32|<=1%" if acc_eng is not None else "")
    else:
        ok = agree16 >= args.min_agree and cos16 >= args.min_cos
        if acc_eng is not None:
            ok = ok and abs(acc_eng - acc_ref16) <= 0.005 + 1e-9
        crit = f"agree>={args.min_agree}, cos>={args.min_cos}" + (", |acc-acc16|<=0.5%" if acc_eng is not None else "")
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} ({crit})")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
