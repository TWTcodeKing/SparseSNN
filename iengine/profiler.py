"""Per-module-type latency profiler for SNN inference.

Profiles detailed latency breakdown by operation type:
  - Conv2d:  all nn.Conv2d layers
  - Linear:  all nn.Linear layers
  - LIF/IF:  spiking neuron layers (LIFNeuron, IFNeuron, MultiStep*)
  - SSA:     Spiking Self-Attention (Spikformer only)
  - BN:      BatchNorm layers
  - Other:   everything else

Identifies the inference bottleneck for SNN models.

Usage:
    python -m iengine.profiler \
        --config configs/spikformer/spikformer_cifar.yaml \
        --checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets

    python -m iengine.profiler \
        --model ms_resnet34 \
        --checkpoint output/.../best.pth \
        --dataset cifar100 --data-root /home/twt/datasets --T 6
"""

import argparse
import sys
import os
import time
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from models.neurons import LIFNeuron, IFNeuron, MultiStepLIFNeuron, MultiStepIFNeuron


# Module type classification
_NEURON_TYPES = (LIFNeuron, IFNeuron, MultiStepLIFNeuron, MultiStepIFNeuron)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)


def _classify_module(module):
    """Classify a module into one of the profiled categories."""
    if module.__class__.__name__ == 'SSA':
        return 'SSA'
    if isinstance(module, nn.Conv2d):
        return 'Conv2d'
    if isinstance(module, nn.Linear):
        return 'Linear'
    if isinstance(module, _NEURON_TYPES):
        return 'LIF/IF'
    if isinstance(module, _BN_TYPES):
        return 'BN'
    return None  # skip — don't profile containers/wrappers


class ModuleProfiler:
    """CUDA event-based per-module latency profiler.

    Registers paired pre/post hooks on all target modules. Uses deferred
    CUDA events (no synchronization during forward) to measure kernel-level
    latency per module.
    """

    def __init__(self):
        self._hooks = []
        self._events = []      # (name, category, t_start, t_end)
        self._per_layer = OrderedDict()   # {name: {category, total_ms, calls}}
        self._per_type = defaultdict(lambda: {'total_ms': 0.0, 'calls': 0, 'layers': set()})

    def register(self, model, skip_containers=True):
        """Register hooks on all classifiable modules.

        Args:
            model: The model to profile.
            skip_containers: If True, skip parent modules that contain
                profiled children (avoid double-counting). SSA is an
                exception — profiled as a unit even though it contains
                Linear/BN/LIF children.
        """
        # First pass: find which modules to profile
        profiled_modules = {}
        for name, module in model.named_modules():
            cat = _classify_module(module)
            if cat is not None:
                profiled_modules[name] = (module, cat)

        # For SSA modules: skip their children to avoid double-counting
        ssa_prefixes = [n + '.' for n, (_, c) in profiled_modules.items() if c == 'SSA']

        for name, (module, cat) in profiled_modules.items():
            # Skip children of SSA (they're profiled as part of SSA)
            if cat != 'SSA' and any(name.startswith(p) for p in ssa_prefixes):
                continue

            event_holder = {}

            def make_pre(n, c, eh):
                def pre_hook(mod, inp):
                    t0 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    eh['t0'] = t0
                    eh['name'] = n
                    eh['cat'] = c
                return pre_hook

            def make_post(n, c, eh):
                def post_hook(mod, inp, out):
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record()
                    self._events.append((eh['name'], eh['cat'], eh['t0'], t1))
                return post_hook

            h1 = module.register_forward_pre_hook(make_pre(name, cat, event_holder))
            h2 = module.register_forward_hook(make_post(name, cat, event_holder))
            self._hooks.extend([h1, h2])

        return len(self._hooks) // 2

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def resolve(self):
        """Resolve all deferred CUDA events. Call after all forward passes."""
        if not self._events:
            return
        torch.cuda.synchronize()
        for name, cat, t0, t1 in self._events:
            ms = t0.elapsed_time(t1)
            if name not in self._per_layer:
                self._per_layer[name] = {'category': cat, 'total_ms': 0.0, 'calls': 0}
            self._per_layer[name]['total_ms'] += ms
            self._per_layer[name]['calls'] += 1
            self._per_type[cat]['total_ms'] += ms
            self._per_type[cat]['calls'] += 1
            self._per_type[cat]['layers'].add(name)
        self._events.clear()

    def reset(self):
        self._events.clear()
        self._per_layer.clear()
        self._per_type.clear()

    def print_summary(self, num_samples=1, total_forward_ms=None):
        """Print profiling results.

        Args:
            num_samples: Total samples processed (for per-sample stats).
            total_forward_ms: Total wall-clock forward time (if known).
        """
        self.resolve()
        if not self._per_type:
            print("  No profiling data.")
            return

        # ── Per-type summary ──
        print(f"\n{'='*70}")
        print(f"  Per-Type Latency Breakdown ({num_samples} samples)")
        print(f"{'='*70}")

        total_profiled = sum(d['total_ms'] for d in self._per_type.values())
        sorted_types = sorted(self._per_type.items(),
                              key=lambda x: x[1]['total_ms'], reverse=True)

        print(f"\n  {'Type':<12} {'Total (ms)':>12} {'Per-sample':>12} "
              f"{'Fraction':>10} {'Layers':>8} {'Calls':>8}")
        print("  " + "-" * 66)

        for cat, d in sorted_types:
            per_sample = d['total_ms'] / num_samples
            frac = d['total_ms'] / max(total_profiled, 1e-6) * 100
            n_layers = len(d['layers'])
            print(f"  {cat:<12} {d['total_ms']:>12.2f} {per_sample:>12.4f} "
                  f"{frac:>9.1f}% {n_layers:>8} {d['calls']:>8}")

        per_sample_total = total_profiled / num_samples
        print(f"  {'TOTAL':<12} {total_profiled:>12.2f} {per_sample_total:>12.4f} "
              f"{'100.0':>9}%")

        if total_forward_ms is not None:
            overhead = total_forward_ms - total_profiled
            print(f"\n  Wall-clock forward: {total_forward_ms:.2f} ms  "
                  f"(profiled: {total_profiled:.2f} ms, "
                  f"overhead/untracked: {overhead:.2f} ms)")

        # ── Per-layer detail ──
        print(f"\n  {'Layer':<45} {'Type':<8} {'Avg/call (ms)':>14} {'Calls':>6}")
        print("  " + "-" * 76)

        for name, d in self._per_layer.items():
            avg = d['total_ms'] / max(d['calls'], 1)
            print(f"  {name:<45} {d['category']:<8} {avg:>14.4f} {d['calls']:>6}")

        print(f"{'='*70}")

    def get_type_breakdown(self, num_samples=1) -> dict:
        """Return per-type breakdown as a dict."""
        self.resolve()
        result = {}
        for cat, d in self._per_type.items():
            result[cat] = {
                'total_ms': d['total_ms'],
                'per_sample_ms': d['total_ms'] / num_samples,
                'layers': len(d['layers']),
                'calls': d['calls'],
            }
        return result


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_inference(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> dict:
    """Profile per-module-type inference latency.

    Args:
        model:      SNN model (on CUDA, eval mode).
        dataloader: Data loader (uses single batch for latency).
        device:     CUDA device.
        n_warmup:   Warmup iterations.
        n_measure:  Timed iterations.

    Returns:
        Dict with type breakdown and per-layer details.
    """
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)
    batch_size = images.shape[0]

    # Warmup (no profiling)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(images)
            reset_net(model)

    # Profiled run
    profiler = ModuleProfiler()
    n_hooked = profiler.register(model)
    print(f"  Registered {n_hooked} module hooks")

    with torch.no_grad():
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for _ in range(n_measure):
            model(images)
            reset_net(model)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    total_forward_ms = (t_end - t_start) * 1000
    num_samples = n_measure * batch_size

    profiler.print_summary(
        num_samples=num_samples,
        total_forward_ms=total_forward_ms,
    )

    result = profiler.get_type_breakdown(num_samples)
    result['_total_forward_ms'] = total_forward_ms
    result['_num_samples'] = num_samples
    result['_per_sample_ms'] = total_forward_ms / num_samples

    profiler.remove()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='SNN inference latency profiler — per-module-type breakdown')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='ResNet model name (e.g. ms_resnet34)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--T', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fuse-neurons', action='store_true', default=False,
                        help='Replace LIF/IF neurons with fused Triton kernels')
    return parser.parse_args()


def main():
    args = parse_args()

    from tengine.utils import (
        set_seed, build_model, build_model_from_config,
        load_model_config, get_dataset_config, build_dataloaders,
    )
    set_seed(args.seed)

    gpu_id = int(args.gpu_ids.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    ds_cfg = get_dataset_config(args.dataset)

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    if args.config:
        config = load_model_config(args.config)
        config.update(ds_cfg)
        config['T'] = args.T if args.T else config.get('T', 4)
        model = build_model_from_config(config)
    elif args.model:
        model = build_model(args.model, num_classes=ds_cfg['num_classes'],
                            in_channels=ds_cfg['in_channels'], T=args.T or 4)
    else:
        raise ValueError("Must provide --config or --model")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(device).eval()

    if args.fuse_neurons:
        from iengine.common.neuron_utils import maybe_fuse_neurons
        maybe_fuse_neurons(model, fuse=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    _, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size,
        img_size=ds_cfg['img_size'], num_workers=0)

    print(f"\nProfiling with batch_size={args.batch_size}, "
          f"warmup={args.warmup}, iterations={args.iterations}")

    profile_inference(
        model, test_loader, device,
        n_warmup=args.warmup, n_measure=args.iterations,
    )


if __name__ == '__main__':
    main()
