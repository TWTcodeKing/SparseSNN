from collections import defaultdict


def export_results_md(results, filepath, title='Weight-Sparse Inference Benchmark'):
    """Export benchmark results to markdown.

    Args:
        results: List of dicts, each with keys:
            model, backend, sparsity, dense_mean_ms, sparse_mean_ms, speedup
            Optionally: config (dict with input_shape, batch_size, device,
            num_warmup, num_iters).
        filepath: Output .md file path.
        title: Report title.
    """
    lines = [f'# {title}', '']

    # Configuration section (use first result's config if available)
    config = results[0].get('config', {}) if results else {}
    lines.append('## Configuration')
    input_shape = config.get('input_shape', '3x224x224')
    batch_size = config.get('batch_size', 'N/A')
    device = config.get('device', 'CUDA')
    num_warmup = config.get('num_warmup', 10)
    num_iters = config.get('num_iters', 100)
    lines.append(f'- Input: ImageNet ({input_shape}), batch_size={batch_size}')
    lines.append(f'- Device: {device}')
    lines.append(f'- Warmup: {num_warmup}, Iterations: {num_iters}')
    lines.append('')

    # Results table
    lines.append('## Results')
    lines.append('')
    lines.append('| Model | Backend | Sparsity | Dense (ms) | Sparse (ms) | Speedup |')
    lines.append('|-------|---------|----------|-----------|-------------|---------|')
    for r in results:
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['sparsity']:.2f} "
            f"| {r['dense_mean_ms']:.2f} | {r['sparse_mean_ms']:.2f} "
            f"| {r['speedup']:.2f}x |"
        )
    lines.append('')

    # Per-backend summary
    lines.append('## Per-Backend Summary')
    lines.append('')
    backend_speedups = defaultdict(list)
    for r in results:
        backend_speedups[r['backend']].append(r['speedup'])
    lines.append('| Backend | Avg Speedup | Num Runs |')
    lines.append('|---------|-------------|----------|')
    for backend, speedups in sorted(backend_speedups.items()):
        avg = sum(speedups) / len(speedups)
        lines.append(f'| {backend} | {avg:.2f}x | {len(speedups)} |')
    lines.append('')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def print_result_row(model, backend, sparsity, dense_ms, sparse_ms, speedup):
    """Print a single result row to console for live progress."""
    print(f"  {model:>12s} | {backend:>16s} | {sparsity:.2f} | "
          f"{dense_ms:8.2f} ms | {sparse_ms:8.2f} ms | {speedup:.2f}x")
