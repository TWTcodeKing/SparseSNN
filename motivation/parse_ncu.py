"""Parse ncu CSV output and produce analysis tables for the motivation experiment.

Usage:
    python motivation/parse_ncu.py  # analyzes all batch sizes in motivation/output/
    python motivation/parse_ncu.py motivation/output/ncu_b1.csv  # single file
"""

import csv
import sys
import os
from collections import defaultdict


def parse_ncu(path):
    """Parse ncu --set basic --csv output into per-kernel records."""
    with open(path) as f:
        lines = [l for l in f if not l.startswith('==') and not l.startswith('Done:')]
    reader = csv.DictReader(lines)

    kernels = defaultdict(dict)
    for row in reader:
        kid = row.get('ID', '')
        kname = row.get('Kernel Name', '')
        metric = row.get('Metric Name', '')
        value_str = row.get('Metric Value', '0')
        unit = row.get('Metric Unit', '')

        if not kid or not kname:
            continue

        kernels[kid]['name'] = kname
        try:
            val = float(value_str.replace(',', ''))
        except ValueError:
            val = 0.0

        if metric == 'Compute (SM) Throughput':
            kernels[kid]['sm_pct'] = val
        elif metric == 'DRAM Throughput':
            kernels[kid]['dram_pct'] = val
        elif metric == 'Memory Throughput':
            kernels[kid]['mem_pct'] = val
        elif metric == 'Duration':
            kernels[kid]['duration_us'] = val / 1000.0  # ns -> us
        elif metric == 'Achieved Occupancy':
            kernels[kid]['occupancy'] = val

    return dict(kernels)


def classify(name):
    """Classify kernel into GEMM, LIF, Reformat, or skip."""
    if 'xmma_gemm' in name or 'cutlass' in name or 'cublas' in name:
        return 'GEMM'
    if 'genericReformat' in name or 'copyVectorized' in name:
        return 'Reformat'
    if 'distribution_elementwise' in name or 'normal_kernel' in name:
        return 'SKIP'
    if '__myl_' in name:
        return 'LIF'
    return 'Other'


def analyze_single(path, batch):
    """Print detailed per-kernel analysis for one batch size."""
    kernels = parse_ncu(path)
    trt = {k: v for k, v in kernels.items() if classify(v['name']) != 'SKIP'}

    print(f"\n{'='*120}")
    print(f"  ncu Per-Kernel Analysis: batch={batch}")
    print(f"{'='*120}")
    print(f"{'#':>3} | {'Type':>8} | {'Dur(us)':>8} | {'SM%':>6} | "
          f"{'DRAM%':>6} | {'Mem%':>6} | {'Occ%':>6} | {'Bound':>8} | Kernel")
    print("-" * 120)

    cats = defaultdict(lambda: {"count": 0, "total_us": 0.0,
                                "sm_sum": 0.0, "dram_sum": 0.0,
                                "mem_sum": 0.0, "occ_sum": 0.0})
    total_us = 0.0

    for i, (kid, k) in enumerate(sorted(trt.items(), key=lambda x: int(x[0]))):
        dur = k.get('duration_us', 0)
        sm = k.get('sm_pct', 0)
        dram = k.get('dram_pct', 0)
        mem = k.get('mem_pct', 0)
        occ = k.get('occupancy', 0)
        cat = classify(k['name'])
        bound = 'Compute' if sm > mem else 'Memory'

        total_us += dur
        c = cats[cat]
        c['count'] += 1
        c['total_us'] += dur
        c['sm_sum'] += sm
        c['dram_sum'] += dram
        c['mem_sum'] += mem
        c['occ_sum'] += occ

        short = k['name'][:60]
        print(f"{i:>3} | {cat:>8} | {dur:>8.3f} | {sm:>6.1f} | "
              f"{dram:>6.1f} | {mem:>6.1f} | {occ:>5.1f}% | {bound:>8} | {short}")

    # Aggregate
    print(f"\n  AGGREGATE: total={total_us:.2f}us, {len(trt)} kernels")
    print(f"  {'Category':>10} | {'Count':>6} | {'Total(us)':>10} | {'%Time':>6} | "
          f"{'Avg SM%':>8} | {'Avg DRAM%':>10} | {'Avg Mem%':>9}")
    print(f"  {'-'*80}")

    for cat in ['GEMM', 'LIF', 'Reformat']:
        c = cats.get(cat)
        if not c or c['count'] == 0:
            continue
        n = c['count']
        pct = c['total_us'] / total_us * 100 if total_us else 0
        print(f"  {cat:>10} | {n:>6} | {c['total_us']:>10.2f} | {pct:>5.1f}% | "
              f"{c['sm_sum']/n:>7.1f}% | {c['dram_sum']/n:>9.1f}% | {c['mem_sum']/n:>8.1f}%")

    return cats, total_us


def analyze_all(out_dir="motivation/output"):
    """Cross-batch comparison analysis."""
    results = []
    for b in [1, 4, 8, 16]:
        path = os.path.join(out_dir, f"ncu_b{b}.csv")
        if not os.path.exists(path):
            continue
        kernels = parse_ncu(path)
        trt = {k: v for k, v in kernels.items() if classify(v['name']) != 'SKIP'}

        cats = defaultdict(lambda: {"count": 0, "total_us": 0.0,
                                    "sm_sum": 0.0, "dram_sum": 0.0, "mem_sum": 0.0})
        total_us = 0.0
        for kid, k in trt.items():
            dur = k.get('duration_us', 0)
            cat = classify(k['name'])
            total_us += dur
            c = cats[cat]
            c['count'] += 1
            c['total_us'] += dur
            c['sm_sum'] += k.get('sm_pct', 0)
            c['dram_sum'] += k.get('dram_pct', 0)
            c['mem_sum'] += k.get('mem_pct', 0)

        n = len(trt)
        results.append({
            'batch': b,
            'total_us': total_us,
            'n_kernels': n,
            'cats': dict(cats),
            'all_sm': sum(k.get('sm_pct', 0) for k in trt.values()) / max(n, 1),
            'all_dram': sum(k.get('dram_pct', 0) for k in trt.values()) / max(n, 1),
            'all_mem': sum(k.get('mem_pct', 0) for k in trt.values()) / max(n, 1),
        })

    if not results:
        print("No ncu files found")
        return

    # Table: cross-batch summary
    print(f"\n{'='*100}")
    print("Cross-Batch Summary")
    print(f"{'='*100}")
    print(f"{'Batch':>6} | {'T*B':>5} | {'Total(us)':>10} | {'GEMM(us)':>9} | "
          f"{'LIF(us)':>8} | {'GEMM%':>6} | {'LIF%':>6} | "
          f"{'Avg SM%':>8} | {'Avg DRAM%':>10}")
    print("-" * 100)

    for r in results:
        b = r['batch']
        total = r['total_us']
        gemm = r['cats'].get('GEMM', {}).get('total_us', 0)
        lif = r['cats'].get('LIF', {}).get('total_us', 0)
        print(f"{b:>6} | {b*4:>5} | {total:>10.2f} | {gemm:>9.2f} | {lif:>8.2f} | "
              f"{gemm/total*100 if total else 0:>5.1f}% | "
              f"{lif/total*100 if total else 0:>5.1f}% | "
              f"{r['all_sm']:>7.1f}% | {r['all_dram']:>9.1f}%")


def main():
    if len(sys.argv) > 1:
        # Single file mode
        path = sys.argv[1]
        batch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        analyze_single(path, batch)
    else:
        # Analyze all batch sizes
        out_dir = "motivation/output"
        for b in [1, 16]:
            path = os.path.join(out_dir, f"ncu_b{b}.csv")
            if os.path.exists(path):
                analyze_single(path, b)
        analyze_all(out_dir)


if __name__ == "__main__":
    main()
