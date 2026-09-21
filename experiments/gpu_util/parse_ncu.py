#!/usr/bin/env python3
"""Phase 3: Parse ncu reports and generate comparison tables.

Usage:
    python experiments/gpu_util/parse_ncu.py [--top 10]
"""

import argparse
import csv
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.gpu_util.config import (
    MODELS, BATCH_SIZES, NCU_PATH, NCU_REPORTS_DIR, RESULTS_DIR,
    METRIC_LABELS, ncu_report_path,
)

EXCLUDE_PATTERNS = ['Memset', 'Memcpy', 'memset', 'memcpy']

SUMMARY_METRICS = [
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_fma.sum",
    "smsp__inst_executed_pipe_alu.sum",
]

DURATION_COL = "gpu__time_duration.sum"
KERNEL_NAME_COL = "Kernel Name"

# Unit normalization: ncu may report in different units depending on magnitude
UNIT_TO_MS = {"nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "s": 1e3}
UNIT_TO_GBS = {"byte/second": 1e-9, "Kbyte/s": 1e-6, "Mbyte/s": 1e-3, "Gbyte/s": 1.0, "Tbyte/s": 1e3}


def extract_csv(report_path):
    """Run ncu --import --csv and parse. Returns (rows, units_map)."""
    ncu_rep = report_path + ".ncu-rep"
    if not os.path.exists(ncu_rep):
        print(f"  WARNING: {ncu_rep} not found")
        return [], {}

    cmd = [NCU_PATH, "--import", ncu_rep, "--csv", "--page", "raw"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: ncu --csv timed out")
        return [], {}

    if result.returncode != 0:
        print(f"  WARNING: ncu --csv failed: {result.stderr[:200]}")
        return [], {}

    lines = result.stdout.strip().split('\n')
    if len(lines) < 3:
        return [], {}

    # Row 0 = header, Row 1 = units, Row 2+ = data
    header_row = list(csv.reader(io.StringIO(lines[0])))[0]
    units_row = list(csv.reader(io.StringIO(lines[1])))[0]
    units_map = {}
    for i in range(min(len(header_row), len(units_row))):
        units_map[header_row[i]] = units_row[i]

    data_lines = [lines[0]] + lines[2:]
    reader = csv.DictReader(io.StringIO('\n'.join(data_lines)))
    return list(reader), units_map


def parse_float(val_str):
    if not val_str or val_str in ('n/a', 'N/A', '', '--'):
        return 0.0
    val_str = val_str.strip().replace(',', '')
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def get_duration_ms(row, units_map):
    """Get kernel duration in milliseconds."""
    raw = parse_float(row.get(DURATION_COL, '0'))
    unit = units_map.get(DURATION_COL, 'ms')
    return raw * UNIT_TO_MS.get(unit, 1.0)


def get_bandwidth_gbs(row, units_map):
    """Get DRAM bandwidth in GB/s."""
    col = "dram__bytes.sum.per_second"
    raw = parse_float(row.get(col, '0'))
    unit = units_map.get(col, 'Gbyte/s')
    return raw * UNIT_TO_GBS.get(unit, 1.0)


def filter_kernels(rows):
    filtered = []
    for row in rows:
        name = row.get(KERNEL_NAME_COL, "")
        if any(pat in name for pat in EXCLUDE_PATTERNS):
            continue
        filtered.append(row)
    return filtered


def shorten_kernel_name(name):
    if 'void ' in name:
        name = name.replace('void ', '')
    for sep in ['<', '(']:
        if sep in name:
            name = name[:name.index(sep)]
    if '::' in name:
        name = name.split('::')[-1]
    if name.startswith('at::'):
        name = name[4:]
    if len(name) > 45:
        name = name[:42] + '...'
    return name


def compute_weighted_averages(rows, units_map):
    """Duration-weighted average. Returns (weighted, total_ms, count)."""
    if not rows:
        return {}, 0.0, 0

    total_ms = 0.0
    metric_sums = {m: 0.0 for m in SUMMARY_METRICS}
    valid = 0

    for row in rows:
        dur = get_duration_ms(row, units_map)
        if dur <= 0:
            continue
        total_ms += dur
        valid += 1
        for m in SUMMARY_METRICS:
            val = parse_float(row.get(m, '0'))
            # Normalize bandwidth to GB/s
            if m == "dram__bytes.sum.per_second":
                val = get_bandwidth_gbs(row, units_map)
            metric_sums[m] += dur * val

    weighted = {}
    for m in SUMMARY_METRICS:
        weighted[m] = metric_sums[m] / total_ms if total_ms > 0 else 0.0
    return weighted, total_ms, valid


def get_kernel_hotspots(rows, units_map, top_n=10):
    if not rows:
        return []
    entries = []
    for row in rows:
        dur = get_duration_ms(row, units_map)
        name = shorten_kernel_name(row.get(KERNEL_NAME_COL, 'unknown'))
        metrics = {}
        for m in SUMMARY_METRICS:
            if m == "dram__bytes.sum.per_second":
                metrics[m] = get_bandwidth_gbs(row, units_map)
            else:
                metrics[m] = parse_float(row.get(m, '0'))
        entries.append((dur, name, metrics))
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries[:top_n]


def format_value(metric, value):
    if 'pct' in metric or 'occupancy' in metric:
        return f"{value:.1f}%"
    elif 'bytes.sum.per_second' in metric:
        return f"{value:.1f} GB/s"  # already normalized to GB/s
    elif 'inst_executed' in metric:
        if value > 1e6:
            return f"{value / 1e6:.1f}M"
        elif value > 1e3:
            return f"{value / 1e3:.1f}K"
        return f"{value:.0f}"
    else:
        return f"{value:.2f}"


def generate_comparison(model_key, batch):
    print(f"\n{'='*70}")
    print(f"  {model_key} | B={batch}")
    print(f"{'='*70}")

    results = {}
    for backend in ['sengine', 'trt']:
        report = ncu_report_path(model_key, batch, backend)
        rows, units_map = extract_csv(report)
        if not rows:
            print(f"  {backend}: no data")
            results[backend] = None
            continue
        filtered = filter_kernels(rows)
        weighted, total_ms, n_kernels = compute_weighted_averages(filtered, units_map)
        hotspots = get_kernel_hotspots(filtered, units_map)
        results[backend] = {
            'weighted': weighted,
            'total_ms': total_ms,
            'kernel_count': n_kernels,
            'hotspots': hotspots,
        }
        print(f"  {backend}: {n_kernels} kernels, total GPU time = {total_ms:.3f} ms")

    se = results.get('sengine')
    tr = results.get('trt')
    if se and tr:
        print(f"\n  {'Metric':<35} {'sengine':>12} {'TensorRT':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12}")
        for m in SUMMARY_METRICS:
            label = METRIC_LABELS.get(m, m.split('.')[-2])
            sv = se['weighted'].get(m, 0)
            tv = tr['weighted'].get(m, 0)
            print(f"  {label:<35} {format_value(m, sv):>12} {format_value(m, tv):>12}")
        print(f"  {'Kernel count':<35} {se['kernel_count']:>12} {tr['kernel_count']:>12}")
        print(f"  {'Total GPU time':<35} "
              f"{se['total_ms']:>10.3f}ms "
              f"{tr['total_ms']:>10.3f}ms")

    for backend in ['sengine', 'trt']:
        r = results.get(backend)
        if not r or not r['hotspots']:
            continue
        sm_k = "sm__warps_active.avg.pct_of_peak_sustained_active"
        tc_k = "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
        dr_k = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        bw_k = "dram__bytes.sum.per_second"
        print(f"\n  Top-10 kernels ({backend}):")
        print(f"  {'#':<3} {'Duration':>10} {'SM%':>7} {'TC%':>7} {'DRAM%':>7} {'BW GB/s':>8}  Kernel")
        print(f"  {'-'*3} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*8}  {'-'*40}")
        for i, (dur, name, metrics) in enumerate(r['hotspots'], 1):
            sm = metrics.get(sm_k, 0)
            tc = metrics.get(tc_k, 0)
            dram = metrics.get(dr_k, 0)
            bw = metrics.get(bw_k, 0)
            if dur >= 1.0:
                dur_str = f"{dur:.2f}ms"
            elif dur >= 0.001:
                dur_str = f"{dur*1000:.1f}us"
            else:
                dur_str = f"{dur*1e6:.0f}ns"
            print(f"  {i:<3} {dur_str:>10} {sm:>6.1f}% {tc:>6.1f}% {dram:>6.1f}% {bw:>7.1f}  {name}")

    return results


def save_csv_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "gpu_utilization_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Model', 'Batch', 'Backend', 'Kernel Count', 'Total GPU Time (ms)']
        header += [METRIC_LABELS.get(m, m) for m in SUMMARY_METRICS]
        writer.writerow(header)
        for (model_key, batch), results in all_results.items():
            for backend in ['sengine', 'trt']:
                r = results.get(backend)
                if not r:
                    continue
                row = [model_key, batch, backend, r['kernel_count'],
                       f"{r['total_ms']:.3f}"]
                for m in SUMMARY_METRICS:
                    row.append(f"{r['weighted'].get(m, 0):.4f}")
                writer.writerow(row)
    print(f"\nSaved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()

    all_results = {}
    # Discover all available reports (don't restrict to BATCH_SIZES)
    for model_key in MODELS:
        for batch in BATCH_SIZES:
            has_any = any(
                os.path.exists(ncu_report_path(model_key, batch, be) + ".ncu-rep")
                for be in ['sengine', 'trt']
            )
            if not has_any:
                continue
            results = generate_comparison(model_key, batch)
            all_results[(model_key, batch)] = results
    # Also scan for reports with batch sizes not in BATCH_SIZES
    if os.path.isdir(NCU_REPORTS_DIR):
        import re
        for fname in os.listdir(NCU_REPORTS_DIR):
            m = re.match(r'(.+)_B(\d+)_(sengine|trt)\.ncu-rep$', fname)
            if not m:
                continue
            mk, bs = m.group(1), int(m.group(2))
            if mk in MODELS and (mk, bs) not in all_results:
                results = generate_comparison(mk, bs)
                all_results[(mk, bs)] = results

    if not all_results:
        print("No ncu reports found.")
        return

    save_csv_results(all_results)

    # Final summary
    print(f"\n{'='*90}")
    print(f"  GPU Utilization Summary: sengine vs TensorRT")
    print(f"{'='*90}")
    print(f"  {'Model':<25} {'B':>3} {'Backend':<9} "
          f"{'SM%':>6} {'TC%':>6} {'DRAM%':>6} {'FMA%':>6} {'BW GB/s':>8} {'Kernels':>8} {'Time':>10}")
    print(f"  {'-'*25} {'-'*3} {'-'*9} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")

    sm_k = "sm__warps_active.avg.pct_of_peak_sustained_active"
    tc_k = "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    dr_k = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    fm_k = "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
    bw_k = "dram__bytes.sum.per_second"

    for (model_key, batch), results in all_results.items():
        for backend in ['sengine', 'trt']:
            r = results.get(backend)
            if not r:
                continue
            w = r['weighted']
            print(f"  {model_key:<25} {batch:>3} {backend:<9} "
                  f"{w.get(sm_k, 0):>5.1f}% {w.get(tc_k, 0):>5.1f}% "
                  f"{w.get(dr_k, 0):>5.1f}% {w.get(fm_k, 0):>5.1f}% "
                  f"{w.get(bw_k, 0):>7.1f} "
                  f"{r['kernel_count']:>8} "
                  f"{r['total_ms']:>8.3f}ms")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
