"""End-to-end Dense-Sparse Factorized Inference Engine for SNN models.

Decomposes every eligible Linear weight into a 2:4 sparse component (accelerated
by NVIDIA Sparse Tensor Cores) and a dense residual (processed via
gather-accumulate exploiting SNN activation sparsity).

Usage:
    python -m iengine.factored_inference \
        --config configs/spikformer/spikformer_cifar.yaml \
        --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth \
        --dataset cifar100 --data-root /data --gpu-ids 0
"""

import os
import sys
import time
import argparse
import copy
from typing import Optional

import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import reset_net
from sparse.comp_2_4 import (
    convert_model_factorized,
    verify_factorization,
    FactorizedLinear,
    _ORIGINAL_LINEAR_ATTR,
)


class DenseLinearProfiler:
    """Profile per-layer latency of dense nn.Linear layers using deferred CUDA events.

    Registers forward hooks on eligible Linear layers (same ones FactorizedLinear
    would convert) to measure their kernel-level latency for fair comparison.
    """

    def __init__(self):
        self._hooks = []
        self._events = []  # (layer_name, t_start, t_end, shape)
        self._profile_data = {}  # {name: {total_ms, calls, shape}}

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def register_paired(self, model: nn.Module, exclude_names: list = None):
        """Register pre-hook + post-hook pairs on eligible Linear layers."""
        if exclude_names is None:
            exclude_names = []
        if 'head' not in exclude_names:
            exclude_names = list(exclude_names) + ['head']

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            skip = False
            for excl in exclude_names:
                if name == excl or name.startswith(excl + '.'):
                    skip = True
                    break
            out_f, in_f = module.weight.shape
            if out_f % 16 != 0 or in_f % 16 != 0:
                skip = True
            if skip:
                continue

            shape = (out_f, in_f)
            # Mutable container shared between pre and post hooks
            event_holder = {}

            def make_pre_hook(n, s, eh):
                def pre_hook(module, input):
                    t0 = torch.cuda.Event(enable_timing=True)
                    t0.record()
                    eh['t0'] = t0
                    eh['name'] = n
                    eh['shape'] = s
                return pre_hook

            def make_post_hook(n, s, eh):
                def post_hook(module, input, output):
                    t1 = torch.cuda.Event(enable_timing=True)
                    t1.record()
                    self._events.append((eh['name'], eh['t0'], t1, eh['shape']))
                return post_hook

            h1 = module.register_forward_pre_hook(make_pre_hook(name, shape, event_holder))
            h2 = module.register_forward_hook(make_post_hook(name, shape, event_holder))
            self._hooks.extend([h1, h2])

    def resolve(self):
        """Resolve all deferred events. Call once after all forward passes."""
        if not self._events:
            return
        torch.cuda.synchronize()
        for (name, t0, t1, shape) in self._events:
            ms = t0.elapsed_time(t1)
            if name not in self._profile_data:
                self._profile_data[name] = {'total_ms': 0.0, 'calls': 0, 'shape': shape}
            self._profile_data[name]['total_ms'] += ms
            self._profile_data[name]['calls'] += 1
        self._events = []

    def reset(self):
        self._events = []
        self._profile_data = {}

    def print_summary(self, num_samples: int = 0, total_forward_ms: float = 0.0):
        """Print per-layer dense Linear profiling breakdown."""
        self.resolve()
        if not self._profile_data:
            print("  No dense profiling data.")
            return

        n_calls = max(d['calls'] for d in self._profile_data.values())

        print(f"\n{'Layer':<45} {'Shape':<14} {'Avg/call (ms)':<14} {'Calls':<6}")
        print('-' * 85)

        total_linear_ms = 0.0
        for name, d in sorted(self._profile_data.items()):
            shape_str = f"{d['shape'][0]}x{d['shape'][1]}"
            avg = d['total_ms'] / max(d['calls'], 1)
            total_linear_ms += d['total_ms']
            print(f"  {name:<43} {shape_str:<14} {avg:<14.4f} {d['calls']:<6}")

        divisor = max(num_samples, 1) if num_samples else max(n_calls, 1)
        unit = 'samples' if num_samples else 'calls'
        per_sample_linear = total_linear_ms / divisor

        print(f"\n  === Dense Linear Per-Sample Breakdown ({num_samples or n_calls} {unit}) ===")
        print(f"  Dense Linear total:          {per_sample_linear:.4f} ms/sample")

        if total_forward_ms > 0:
            per_sample_total = total_forward_ms / max(num_samples, 1)
            non_linear_ms = total_forward_ms - total_linear_ms
            per_sample_nonlinear = non_linear_ms / max(num_samples, 1)
            print(f"  Non-linear (Conv/BN/LIF/..): {per_sample_nonlinear:.4f} ms/sample")
            print(f"  Total forward:               {per_sample_total:.4f} ms/sample")
            print(f"  Dense Linear fraction:       {total_linear_ms / max(total_forward_ms, 1e-6) * 100:.1f}%")

    def get_per_layer_ms(self) -> dict:
        """Return {layer_name: avg_ms_per_call}."""
        self.resolve()
        return {
            name: d['total_ms'] / max(d['calls'], 1)
            for name, d in self._profile_data.items()
        }


class FactoredInferenceEngine:
    """Dense-Sparse factorized inference engine.

    Factorizes all eligible Linear layers into:
        - W_24: 2:4 semi-structured sparse (Sparse Tensor Core path)
        - W_res: dense residual (gather-accumulate path)

    The two paths are summed to produce exact (within fp16 precision) results
    identical to the original dense model.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._conversion_info = {}
        self._original_state = None
        self._prepared = False

    def prepare(self, model: nn.Module) -> nn.Module:
        """Factorize all eligible Linear layers in the model.

        Args:
            model: Model to factorize (modified in-place). Should be on CUDA
                for SparseSemiStructuredTensor support.

        Returns:
            The model with FactorizedLinear layers replacing nn.Linear.
        """
        exclude_names = list(self.config.get('exclude_names', []))

        self._conversion_info = convert_model_factorized(
            model, exclude_names=exclude_names
        )

        converted = sum(1 for v in self._conversion_info.values() if v['converted'])
        total = len(self._conversion_info)
        print(f"  [FactoredInference] Converted {converted}/{total} Linear layers")

        self._prepared = True
        return model

    def cleanup(self, model: nn.Module) -> nn.Module:
        """Restore original Linear layers from FactorizedLinear modules.

        Note: This requires that the original weights can be reconstructed
        from W_24 + W_res. Since factorization is exact, this is lossless.

        Args:
            model: Model previously prepared with prepare().

        Returns:
            The model with standard nn.Linear layers restored.
        """
        replacements = []

        for name, module in model.named_modules():
            if isinstance(module, FactorizedLinear):
                replacements.append((name, module))

        for name, fmod in replacements:
            # Reconstruct the original weight from W_24 + W_res
            W_24 = fmod.W_24.data
            # If W_24 is a SparseSemiStructuredTensor, convert to dense first
            if hasattr(W_24, 'to_dense'):
                W_24 = W_24.to_dense()
            W_original = (W_24.float() + fmod.W_res.data.float())

            linear = nn.Linear(
                fmod.in_features, fmod.out_features,
                bias=(fmod.bias is not None),
                device=W_original.device,
                dtype=torch.float32,
            )
            linear.weight = nn.Parameter(W_original, requires_grad=True)
            if fmod.bias is not None:
                linear.bias = nn.Parameter(fmod.bias.data.float(), requires_grad=True)

            # Navigate to parent and replace
            parts = name.rsplit('.', 1)
            if len(parts) == 1:
                parent = model
                child_name = parts[0]
            else:
                parent = dict(model.named_modules())[parts[0]]
                child_name = parts[1]
            setattr(parent, child_name, linear)

        self._prepared = False
        self._conversion_info = {}
        return model

    def benchmark(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        """Benchmark factorized inference vs dense baseline.

        Runs the model on data from the dataloader, measuring latency and
        accuracy for both the original (dense) and factorized models.

        Args:
            model: The original (unfactorized) model.
            dataloader: Validation dataloader.
            device: CUDA device.
            warmup: Number of warmup iterations.
            iterations: Number of timed iterations.

        Returns:
            Dict with benchmark results:
                - dense_ms: average per-sample latency (dense), ms
                - factored_ms: average per-sample latency (factorized), ms
                - speedup: dense_ms / factored_ms
                - dense_acc: dense top-1 accuracy (%)
                - factored_acc: factorized top-1 accuracy (%)
                - acc_delta: factored_acc - dense_acc
                - layers_converted: number of factorized layers
                - conversion_info: per-layer conversion details
        """
        model.eval()

        exclude_names = list(self.config.get('exclude_names', []))

        # ── Dense baseline (accurate timing) ──────────────────────────
        print("  Running dense baseline...")
        dense_acc, dense_ms = self._run_eval(
            model, dataloader, device, warmup, iterations
        )
        print(f"    Dense: {dense_ms:.3f} ms/sample, Acc: {dense_acc:.2f}%")

        # ── Dense baseline (profiled per-layer) ──────────────────────
        print("  Running dense baseline (profiled for per-layer breakdown)...")
        dense_profiler = DenseLinearProfiler()
        dense_profiler.register_paired(model, exclude_names=exclude_names)
        _, dense_prof_ms = self._run_eval(
            model, dataloader, device, warmup=0, iterations=iterations
        )
        dense_profiler.remove()
        print(f"    Dense profiled: {dense_prof_ms:.3f} ms/sample")

        batch_size = next(iter(dataloader))[0].shape[0]
        num_samples = iterations * batch_size
        dense_total_forward_ms = dense_prof_ms * num_samples

        print("\n  --- Dense Per-Layer Linear Profiling ---")
        dense_profiler.print_summary(
            num_samples=num_samples,
            total_forward_ms=dense_total_forward_ms,
        )
        dense_per_layer = dense_profiler.get_per_layer_ms()

        # ── Factorized ─────────────────────────────────────────────────
        print("\n  Preparing factorized model...")
        model_fact = copy.deepcopy(model)
        self.prepare(model_fact)

        # Print conversion diagnostics (why did/didn't semi-structured work?)
        print("\n  --- Semi-Structured Conversion Diagnostics ---")
        FactorizedLinear.print_conversion_diagnostics(model_fact)

        # Warmup run (compiles kernels, warms caches)
        print("\n  Running factorized inference (warmup)...")
        self._run_eval(model_fact, dataloader, device, warmup=warmup, iterations=5)

        # ── Accurate latency measurement (NO profiling overhead) ──
        print("  Running factorized inference (accurate timing, no profiling)...")
        FactorizedLinear.PROFILE = False
        fact_acc, fact_ms = self._run_eval(
            model_fact, dataloader, device, warmup=0, iterations=iterations
        )
        print(f"    Factored: {fact_ms:.3f} ms/sample, Acc: {fact_acc:.2f}%")

        # ── Separate profiled run for per-layer breakdown ──
        print("  Running factorized inference (profiled for breakdown)...")
        FactorizedLinear.PROFILE = True
        FactorizedLinear.reset_profile()
        _, prof_ms = self._run_eval(
            model_fact, dataloader, device, warmup=0, iterations=iterations
        )
        FactorizedLinear.PROFILE = False
        print(f"    Profiled run: {prof_ms:.3f} ms/sample (includes profiling overhead)")
        print(f"    Profiling overhead: {prof_ms - fact_ms:.3f} ms/sample")

        # Compute total samples and total forward time for breakdown
        batch_size = next(iter(dataloader))[0].shape[0]
        num_samples = iterations * batch_size
        # Use the PROFILED run's forward time for breakdown attribution
        total_forward_ms = prof_ms * num_samples

        # Print per-layer profiling breakdown
        print("\n  --- Factored Per-Layer Profiling (Main vs Residual vs Overhead) ---")
        FactorizedLinear.print_profile_summary(
            total_forward_ms=total_forward_ms,
            num_samples=num_samples,
        )

        # ── Side-by-side per-layer comparison ─────────────────────────
        FactorizedLinear.resolve_events()
        fact_per_layer = {}
        for lid, d in FactorizedLinear._profile_data.items():
            fact_per_layer[str(lid)] = {
                'total': d['total_layer_ms'] / max(d['calls'], 1),
                'main': d['main_ms'] / max(d['calls'], 1),
                'res': d['residual_ms'] / max(d['calls'], 1),
                'oh': d['overhead_ms'] / max(d['calls'], 1),
            }

        print(f"\n  --- Per-Layer Comparison: Dense vs Factored (ms/call) ---")
        print(f"  {'Layer':<40} {'Dense':<10} {'Factored':<10} {'Main':<10} {'Res':<10} {'OH':<10} {'Speedup':<8}")
        print('  ' + '-' * 98)

        total_dense_linear = 0.0
        total_fact_linear = 0.0
        for name in sorted(set(list(dense_per_layer.keys()) + list(fact_per_layer.keys()))):
            d_ms = dense_per_layer.get(name, 0.0)
            f_data = fact_per_layer.get(name, {})
            f_ms = f_data.get('total', 0.0)
            f_main = f_data.get('main', 0.0)
            f_res = f_data.get('res', 0.0)
            f_oh = f_data.get('oh', 0.0)
            sp = d_ms / max(f_ms, 1e-6) if f_ms > 0 else 0.0
            total_dense_linear += d_ms
            total_fact_linear += f_ms
            print(f"  {name:<40} {d_ms:<10.4f} {f_ms:<10.4f} {f_main:<10.4f} {f_res:<10.4f} {f_oh:<10.4f} {sp:<8.2f}x")

        overall_sp = total_dense_linear / max(total_fact_linear, 1e-6)
        print(f"  {'TOTAL':<40} {total_dense_linear:<10.4f} {total_fact_linear:<10.4f} {'':30} {overall_sp:<8.2f}x")

        speedup = dense_ms / max(fact_ms, 1e-6)

        # ── FP16 full-model comparison ────────────────────────────────
        # Eliminates dtype cast overhead in FactorizedLinear
        print(f"\n{'='*70}")
        print("  FP16 Full-Model Comparison (no dtype cast overhead)")
        print(f"{'='*70}")

        # Dense fp16
        model_fp16 = copy.deepcopy(model).half()
        print("  Running dense fp16 baseline (warmup)...")
        self._run_eval(model_fp16, dataloader, device, warmup=warmup, iterations=5,
                        input_dtype=torch.float16)
        print("  Running dense fp16 baseline (accurate)...")
        dense_fp16_acc, dense_fp16_ms = self._run_eval(
            model_fp16, dataloader, device, warmup=0, iterations=iterations,
            input_dtype=torch.float16,
        )
        print(f"    Dense fp16: {dense_fp16_ms:.3f} ms/sample, Acc: {dense_fp16_acc:.2f}%")
        del model_fp16

        # Factored fp16 (model.half() then factorize — no casts needed)
        model_fact_fp16 = copy.deepcopy(model).half()
        self.prepare(model_fact_fp16)
        print("  Running factored fp16 (warmup)...")
        self._run_eval(model_fact_fp16, dataloader, device, warmup=warmup, iterations=5,
                        input_dtype=torch.float16)
        print("  Running factored fp16 (accurate)...")
        fact_fp16_acc, fact_fp16_ms = self._run_eval(
            model_fact_fp16, dataloader, device, warmup=0, iterations=iterations,
            input_dtype=torch.float16,
        )
        print(f"    Factored fp16: {fact_fp16_ms:.3f} ms/sample, Acc: {fact_fp16_acc:.2f}%")

        fp16_speedup = dense_fp16_ms / max(fact_fp16_ms, 1e-6)
        print(f"\n  FP16 Speedup: {fp16_speedup:.2f}x")
        print(f"  FP16 Acc delta: {fact_fp16_acc - dense_fp16_acc:+.2f}%")

        self.cleanup(model_fact_fp16)
        del model_fact_fp16

        # Cleanup fp32 factored model
        self.cleanup(model_fact)
        del model_fact

        layers_converted = sum(
            1 for v in self._conversion_info.values() if v.get('converted', False)
        )

        # Also count semi-structured successes from FactorizedLinear instances
        semi_structured_count = sum(
            1 for d in FactorizedLinear._profile_data.values()
            if d.get('has_semi_structured', False)
        )

        results = {
            'dense_ms': dense_ms,
            'factored_ms': fact_ms,
            'speedup': speedup,
            'dense_acc': dense_acc,
            'factored_acc': fact_acc,
            'acc_delta': fact_acc - dense_acc,
            'dense_fp16_ms': dense_fp16_ms,
            'fact_fp16_ms': fact_fp16_ms,
            'fp16_speedup': fp16_speedup,
            'dense_fp16_acc': dense_fp16_acc,
            'fact_fp16_acc': fact_fp16_acc,
            'layers_converted': layers_converted,
            'layers_semi_structured': semi_structured_count,
            'conversion_info': self._conversion_info,
        }

        print(f"\n  FP32 Speedup: {speedup:.2f}x  |  Acc delta: {results['acc_delta']:+.2f}%")
        print(f"  FP16 Speedup: {fp16_speedup:.2f}x  |  Acc delta: {fact_fp16_acc - dense_fp16_acc:+.2f}%")
        print(f"  Layers with semi-structured TC: {semi_structured_count}")
        return results

    def _run_eval(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        warmup: int,
        iterations: int,
        input_dtype: Optional[torch.dtype] = None,
    ) -> tuple[float, float]:
        """Run evaluation and return (accuracy%, per_sample_ms).

        Args:
            model: Model to evaluate.
            dataloader: Validation dataloader.
            device: CUDA device.
            warmup: Warmup iterations (not timed).
            iterations: Timed iterations.
            input_dtype: Override dtype for input images. If None, auto-detect
                from model parameters.

        Returns:
            (top1_accuracy_percent, per_sample_ms)
        """
        model.eval()
        if input_dtype is None:
            # Auto-detect: scan all parameters for consistent dtype
            dtypes = {p.dtype for p in model.parameters()}
            if torch.float16 in dtypes and torch.float32 not in dtypes:
                input_dtype = torch.float16
            else:
                input_dtype = torch.float32
        model_dtype = input_dtype
        correct = 0
        total = 0
        total_time_ms = 0.0

        data_iter = iter(dataloader)
        batch_count = 0

        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                try:
                    images, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    images, targets = next(data_iter)

                images = images.to(device=device, dtype=model_dtype, non_blocking=True)
                _ = model(images)
                reset_net(model)

            # Timed iterations
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            data_iter = iter(dataloader)
            for _ in range(iterations):
                try:
                    images, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    images, targets = next(data_iter)

                images = images.to(device=device, dtype=model_dtype, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

                t0 = time.perf_counter()
                outputs = model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()

                total_time_ms += (t1 - t0) * 1000.0

                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                batch_count += 1

                reset_net(model)

        accuracy = 100.0 * correct / max(total, 1)
        per_sample_ms = total_time_ms / max(total, 1)

        return accuracy, per_sample_ms


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dense-Sparse Factorized Inference Engine'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for transformer models')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name for ResNet models (e.g. sew_resnet34)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of timesteps')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Timed iterations')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--exclude-names', nargs='*', default=None,
                        help='Module names to exclude from factorization')
    return parser.parse_args()


def main():
    args = parse_args()

    from tengine.utils import (
        set_seed, build_model, build_model_from_config,
        load_model_config, build_dataloaders, get_dataset_config,
    )

    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device('cuda')
    set_seed(42)

    # Load dataset config
    ds_cfg = get_dataset_config(args.dataset)
    num_classes = ds_cfg['num_classes']
    img_size = ds_cfg['img_size']
    in_channels = ds_cfg['in_channels']

    # Build model
    if args.config is not None:
        model_cfg = load_model_config(args.config)
        model_cfg.update({
            'num_classes': num_classes,
            'T': args.T,
            'img_size': img_size,
            'in_channels': in_channels,
        })
        model = build_model_from_config(model_cfg)
    elif args.model is not None:
        model_kwargs = {'num_classes': num_classes}
        if 'sew_' in args.model:
            model_kwargs['T'] = args.T
            model_kwargs['connect_f'] = 'ADD'
        else:
            model_kwargs['time_window'] = args.T
        model = build_model(args.model, **model_kwargs)
    else:
        raise ValueError("Must provide either --config or --model")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Build dataloader
    # num_workers=0 to avoid Triton JIT + DataLoader fork conflict
    # (LLVM ERROR: pthread_join failed when Triton compiles inside forked workers)
    _, val_loader = build_dataloaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=0,
    )

    # Run benchmark
    engine_config = {}
    if args.exclude_names:
        engine_config['exclude_names'] = args.exclude_names

    engine = FactoredInferenceEngine(config=engine_config)

    print(f"\n{'='*70}")
    print(f"  Dense-Sparse Factorized Inference Benchmark")
    print(f"  Dataset: {args.dataset}  |  Warmup: {args.warmup}  |  Iters: {args.iterations}")
    print(f"{'='*70}\n")

    results = engine.benchmark(
        model, val_loader, device,
        warmup=args.warmup, iterations=args.iterations,
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Results Summary")
    print(f"{'='*70}")
    print(f"  --- FP32 (with dtype cast overhead) ---")
    print(f"  Dense latency:     {results['dense_ms']:.3f} ms/sample")
    print(f"  Factored latency:  {results['factored_ms']:.3f} ms/sample")
    print(f"  Speedup:           {results['speedup']:.2f}x")
    print(f"  Dense accuracy:    {results['dense_acc']:.2f}%")
    print(f"  Factored accuracy: {results['factored_acc']:.2f}%")
    print(f"  Accuracy delta:    {results['acc_delta']:+.2f}%")
    print(f"")
    print(f"  --- FP16 (no dtype cast overhead) ---")
    print(f"  Dense fp16:        {results['dense_fp16_ms']:.3f} ms/sample")
    print(f"  Factored fp16:     {results['fact_fp16_ms']:.3f} ms/sample")
    print(f"  FP16 Speedup:      {results['fp16_speedup']:.2f}x")
    print(f"  Dense fp16 acc:    {results['dense_fp16_acc']:.2f}%")
    print(f"  Factored fp16 acc: {results['fact_fp16_acc']:.2f}%")
    print(f"  FP16 Acc delta:    {results['fact_fp16_acc'] - results['dense_fp16_acc']:+.2f}%")
    print(f"")
    print(f"  Layers converted:  {results['layers_converted']}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
