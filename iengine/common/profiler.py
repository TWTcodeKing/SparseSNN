"""Per-layer activation sparsity profiler for sparse accelerator backends.

Registers forward pre-hooks on Conv2d, Linear, and custom modules (e.g. SSA)
to measure input activation density during actual inference. Collects per-layer
statistics and exports results to markdown files.
"""

import os
import time
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Optional

from .density import measure_density


class SparsityProfiler:
    """Tracks per-layer activation density during inference.

    Attaches forward pre-hooks to target module types. On each forward call,
    records the input tensor density. Results are exported to markdown via
    export_report().

    Usage:
        profiler = SparsityProfiler()
        profiler.attach(model)
        # ... run inference ...
        profiler.export_report('sparsity_profile.md')
        profiler.detach()
    """

    def __init__(self, track_module_types: Optional[list] = None,
                 track_class_names: Optional[list] = None):
        """
        Args:
            track_module_types: List of nn.Module types to hook (e.g. [nn.Conv2d, nn.Linear]).
                Defaults to [nn.Conv2d, nn.Linear].
            track_class_names: List of class name strings to hook (e.g. ['SSA']).
                Defaults to ['SSA'].
        """
        self._module_types = tuple(track_module_types or [nn.Conv2d, nn.Linear])
        self._class_names = set(track_class_names or ['SSA'])
        self._hooks = []
        self._records = defaultdict(lambda: {
            'densities': [],
            'shapes': [],
            'type': '',
        })

    def attach(self, model: nn.Module) -> int:
        """Register density-tracking hooks on all target modules.

        Returns the number of hooks registered.
        """
        self.detach()
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, self._module_types) or \
               module.__class__.__name__ in self._class_names:
                hook = module.register_forward_pre_hook(self._make_hook(name, module))
                self._hooks.append(hook)
                count += 1
        return count

    def detach(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        """Clear all recorded density data."""
        self._records.clear()

    def _make_hook(self, layer_name: str, module: nn.Module):
        records = self._records
        mod_type = module.__class__.__name__

        def hook_fn(mod, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if not isinstance(x, torch.Tensor):
                return
            density = measure_density(x)
            numel = x.numel()
            nz = int((x != 0).sum().item())
            rec = records[layer_name]
            rec['densities'].append(density)
            rec['shapes'].append(tuple(x.shape))
            rec['type'] = mod_type
            rec['numel'] = numel
            rec['nonzero'] = nz
        return hook_fn

    def get_report(self) -> dict:
        """Return per-layer density statistics.

        Returns:
            {
                'per_layer': {
                    layer_name: {
                        'type': str,
                        'mean_density': float,
                        'min_density': float,
                        'max_density': float,
                        'num_calls': int,
                        'last_shape': tuple,
                        'last_numel': int,
                        'last_nonzero': int,
                        'sparsity': float,
                    }
                },
                'summary': {
                    'overall_mean_density': float,
                    'overall_sparsity': float,
                    'total_layers': int,
                    'sparse_layers': int,
                    'very_sparse_layers': int,
                    'by_type': {type_name: {'count': int, 'mean_density': float}}
                }
            }
        """
        per_layer = {}
        all_densities = []
        type_densities = defaultdict(list)

        for name, rec in sorted(self._records.items()):
            densities = rec['densities']
            if not densities:
                continue
            mean_d = sum(densities) / len(densities)
            min_d = min(densities)
            max_d = max(densities)

            per_layer[name] = {
                'type': rec['type'],
                'mean_density': mean_d,
                'min_density': min_d,
                'max_density': max_d,
                'num_calls': len(densities),
                'last_shape': rec['shapes'][-1] if rec['shapes'] else (),
                'last_numel': rec.get('numel', 0),
                'last_nonzero': rec.get('nonzero', 0),
                'sparsity': 1.0 - mean_d,
            }

            all_densities.append(mean_d)
            type_densities[rec['type']].append(mean_d)

        overall_mean = sum(all_densities) / max(len(all_densities), 1)
        sparse_count = sum(1 for d in all_densities if d < 0.5)
        very_sparse_count = sum(1 for d in all_densities if d < 0.15)

        by_type = {}
        for tname, dlist in type_densities.items():
            by_type[tname] = {
                'count': len(dlist),
                'mean_density': sum(dlist) / len(dlist),
            }

        summary = {
            'overall_mean_density': overall_mean,
            'overall_sparsity': 1.0 - overall_mean,
            'total_layers': len(all_densities),
            'sparse_layers': sparse_count,
            'very_sparse_layers': very_sparse_count,
            'by_type': by_type,
        }

        return {'per_layer': per_layer, 'summary': summary}

    def export_report(self, filepath: str, model_name: str = '',
                      extra_info: str = ''):
        """Export density report to a markdown file.

        Args:
            filepath: Output .md file path.
            model_name: Model name for the report header.
            extra_info: Additional context to include in the header.
        """
        report = self.get_report()
        per_layer = report['per_layer']
        summary = report['summary']

        lines = []
        lines.append(f'# Activation Sparsity Profile{" — " + model_name if model_name else ""}')
        lines.append('')
        if extra_info:
            lines.append(extra_info)
            lines.append('')

        # Summary
        lines.append('## Summary')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Overall mean density | {summary["overall_mean_density"]:.4f} |')
        lines.append(f'| Overall sparsity | {summary["overall_sparsity"]:.4f} |')
        lines.append(f'| Total profiled layers | {summary["total_layers"]} |')
        lines.append(f'| Sparse layers (density < 50%) | {summary["sparse_layers"]} |')
        lines.append(f'| Very sparse layers (density < 15%) | {summary["very_sparse_layers"]} |')
        lines.append('')

        # By type
        if summary['by_type']:
            lines.append('### Density by module type')
            lines.append('')
            lines.append('| Type | Count | Mean Density |')
            lines.append('|------|-------|-------------|')
            for tname, tinfo in sorted(summary['by_type'].items()):
                lines.append(f'| {tname} | {tinfo["count"]} | {tinfo["mean_density"]:.4f} |')
            lines.append('')

        # Per-layer table
        if per_layer:
            lines.append('## Per-Layer Density')
            lines.append('')
            lines.append('| Layer | Type | Mean Density | Sparsity | Min | Max | Calls | Shape |')
            lines.append('|-------|------|-------------|----------|-----|-----|-------|-------|')
            for name, info in sorted(per_layer.items(), key=lambda x: x[1]['mean_density']):
                shape_str = 'x'.join(str(s) for s in info['last_shape'])
                lines.append(
                    f'| {name} | {info["type"]} | '
                    f'{info["mean_density"]:.4f} | {info["sparsity"]:.4f} | '
                    f'{info["min_density"]:.4f} | {info["max_density"]:.4f} | '
                    f'{info["num_calls"]} | {shape_str} |'
                )
            lines.append('')

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def export_benchmark_report(all_results: list, filepath: str,
                            dataset: str = '', max_samples: int = 0,
                            density_profiles: Optional[dict] = None):
    """Export benchmark results (latency + accuracy + sparsity) to markdown.

    Args:
        all_results: List of dicts from SparseBenchmark.compare(), each with
            keys: model, backend, dense_ms, sparse_ms, speedup,
            dense_acc, sparse_acc, acc_delta.
            Optionally 'backend_stats' with get_stats() output.
        filepath: Output .md file path.
        dataset: Dataset name for the header.
        max_samples: Number of samples benchmarked.
        density_profiles: Optional dict mapping model_name -> SparsityProfiler report
            (from profiler.get_report()). If provided, density summary is appended
            per model.
    """
    lines = []
    lines.append('# Sparse Acceleration Benchmark Results')
    lines.append('')
    if dataset or max_samples:
        lines.append(f'Dataset: **{dataset}**  |  Samples: **{max_samples}**')
        lines.append('')

    # Main comparison table
    lines.append('## Latency & Accuracy Comparison')
    lines.append('')
    lines.append('| Model | Backend | Dense (ms) | Sparse (ms) | Speedup | Dense Acc | Sparse Acc | Acc Delta |')
    lines.append('|-------|---------|-----------|------------|---------|-----------|------------|-----------|')
    for r in all_results:
        lines.append(
            f'| {r["model"]} | {r["backend"]} | '
            f'{r["dense_ms"]:.3f} | {r["sparse_ms"]:.3f} | '
            f'{r["speedup"]:.2f}x | '
            f'{r["dense_acc"]:.2f}% | {r["sparse_acc"]:.2f}% | '
            f'{r["acc_delta"]:+.2f}% |'
        )
    lines.append('')

    # Backend-specific stats
    has_backend_stats = any('backend_stats' in r for r in all_results)
    if has_backend_stats:
        lines.append('## Backend Execution Stats')
        lines.append('')
        for r in all_results:
            bs = r.get('backend_stats', {})
            if not bs:
                continue
            lines.append(f'### {r["model"]} — {r["backend"]}')
            lines.append('')
            lines.append(f'| Metric | Value |')
            lines.append(f'|--------|-------|')
            for key, val in bs.items():
                if key == 'per_layer':
                    continue
                if isinstance(val, float):
                    lines.append(f'| {key} | {val:.4f} |')
                else:
                    lines.append(f'| {key} | {val} |')

            # Per-layer stats if available
            per_layer = bs.get('per_layer', {})
            if per_layer:
                lines.append('')
                lines.append(f'<details><summary>Per-layer stats ({len(per_layer)} layers)</summary>')
                lines.append('')
                first = next(iter(per_layer.values()))
                cols = [k for k in first.keys() if k != 'shape']
                lines.append('| Layer | ' + ' | '.join(cols) + ' |')
                lines.append('|-------| ' + ' | '.join(['---'] * len(cols)) + ' |')
                for lname, linfo in sorted(per_layer.items()):
                    vals = []
                    for c in cols:
                        v = linfo.get(c, '')
                        if isinstance(v, float):
                            vals.append(f'{v:.4f}')
                        else:
                            vals.append(str(v))
                    lines.append(f'| {lname} | ' + ' | '.join(vals) + ' |')
                lines.append('')
                lines.append('</details>')
            lines.append('')

    # Density profiles per model
    if density_profiles:
        lines.append('## Activation Sparsity Profiles')
        lines.append('')
        for model_name, profile in density_profiles.items():
            summary = profile['summary']
            per_layer = profile['per_layer']

            lines.append(f'### {model_name}')
            lines.append('')
            lines.append(f'Overall mean density: **{summary["overall_mean_density"]:.4f}** '
                         f'(sparsity: **{summary["overall_sparsity"]:.4f}**)')
            lines.append('')
            lines.append(f'- Sparse layers (< 50%): {summary["sparse_layers"]} / {summary["total_layers"]}')
            lines.append(f'- Very sparse layers (< 15%): {summary["very_sparse_layers"]} / {summary["total_layers"]}')
            lines.append('')

            if summary['by_type']:
                lines.append('| Type | Count | Mean Density |')
                lines.append('|------|-------|-------------|')
                for tname, tinfo in sorted(summary['by_type'].items()):
                    lines.append(f'| {tname} | {tinfo["count"]} | {tinfo["mean_density"]:.4f} |')
                lines.append('')

            if per_layer:
                lines.append(f'<details><summary>Per-layer density ({len(per_layer)} layers)</summary>')
                lines.append('')
                lines.append('| Layer | Type | Density | Sparsity | Min | Max | Calls | Shape |')
                lines.append('|-------|------|---------|----------|-----|-----|-------|-------|')
                for name, info in sorted(per_layer.items(), key=lambda x: x[1]['mean_density']):
                    shape_str = 'x'.join(str(s) for s in info['last_shape'])
                    lines.append(
                        f'| {name} | {info["type"]} | '
                        f'{info["mean_density"]:.4f} | {info["sparsity"]:.4f} | '
                        f'{info["min_density"]:.4f} | {info["max_density"]:.4f} | '
                        f'{info["num_calls"]} | {shape_str} |'
                    )
                lines.append('')
                lines.append('</details>')
                lines.append('')

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
