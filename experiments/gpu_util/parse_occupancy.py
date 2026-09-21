#!/usr/bin/env python3
"""Parse ncu reports for occupancy analysis: achieved vs theoretical.

For each kernel, shows:
  - Achieved occupancy (sm__warps_active.avg.pct_of_peak_sustained_active)
  - Theoretical occupancy (min of register/smem/warp/block limits)
  - Efficiency = achieved / theoretical (how close to the HW-imposed ceiling)

Usage:
    python experiments/gpu_util/parse_occupancy.py
    python experiments/gpu_util/parse_occupancy.py --top 15
    python experiments/gpu_util/parse_occupancy.py --model maxformer_10_512 --batch 16
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.gpu_util.config import (
    MODELS, BATCH_SIZES, NCU_PATH, NCU_REPORTS_DIR, RESULTS_DIR,
    ncu_report_path,
)

EXCLUDE_PATTERNS = ['Memset', 'Memcpy', 'memset', 'memcpy']

# Metrics
ACHIEVED_OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
LIMIT_REG    = "launch__occupancy_limit_registers"
LIMIT_SMEM   = "launch__occupancy_limit_shared_mem"
LIMIT_WARPS  = "launch__occupancy_limit_warps"
LIMIT_BLOCKS = "launch__occupancy_limit_blocks"
DURATION_COL = "gpu__time_duration.sum"
KERNEL_NAME  = "Kernel Name"
BLOCK_SIZE   = "launch__block_size"
GRID_SIZE    = "launch__grid_size"
MAX_WARPS_SM = "device__attribute_max_warps_per_multiprocessor"

UNIT_TO_MS = {"nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0, "s": 1e3}


def extract_csv(report_path):
    """Run ncu --import --csv and parse."""
    ncu_rep = report_path + ".ncu-rep"
    if not os.path.exists(ncu_rep):
        return [], {}

    cmd = [NCU_PATH, "--import", ncu_rep, "--csv", "--page", "raw"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return [], {}

    if result.returncode != 0:
        return [], {}

    lines = result.stdout.strip().split('\n')
    if len(lines) < 3:
        return [], {}

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
    raw = parse_float(row.get(DURATION_COL, '0'))
    unit = units_map.get(DURATION_COL, 'ms')
    return raw * UNIT_TO_MS.get(unit, 1.0)


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
    if len(name) > 50:
        name = name[:47] + '...'
    return name


def _theoretical_occ_pct(row):
    """Compute theoretical occupancy % from per-resource block limits.

    launch__occupancy_limit_* are in units of 'blocks per SM'.
    theoretical_pct = min(limits) * warps_per_block / max_warps_per_sm * 100
    """
    lim_reg  = parse_float(row.get(LIMIT_REG, '0'))
    lim_smem = parse_float(row.get(LIMIT_SMEM, '0'))
    lim_warp = parse_float(row.get(LIMIT_WARPS, '0'))
    lim_blk  = parse_float(row.get(LIMIT_BLOCKS, '0'))
    block_sz = parse_float(row.get(BLOCK_SIZE, '0'))
    max_warps = parse_float(row.get(MAX_WARPS_SM, '48'))  # 48 for sm_89

    if block_sz <= 0 or max_warps <= 0:
        return 0, 'n/a', lim_reg, lim_smem

    warps_per_block = block_sz / 32.0

    # Convert each limit from blocks → occupancy %
    def blk_to_pct(n_blocks):
        return n_blocks * warps_per_block / max_warps * 100 if n_blocks > 0 else 0

    limits = {
        'reg':  blk_to_pct(lim_reg),
        'smem': blk_to_pct(lim_smem),
        'warp': blk_to_pct(lim_warp),
        'blk':  blk_to_pct(lim_blk),
    }
    bottleneck = min(limits, key=limits.get)
    theoretical = limits[bottleneck]
    # Cap at 100%
    theoretical = min(theoretical, 100.0)

    return theoretical, bottleneck, limits['reg'], limits['smem']


def analyze_occupancy(rows, units_map, top_n=10):
    """Analyze occupancy for all kernels. Returns sorted list of entries."""
    entries = []
    for row in rows:
        name = row.get(KERNEL_NAME, "")
        if any(pat in name for pat in EXCLUDE_PATTERNS):
            continue

        dur = get_duration_ms(row, units_map)
        achieved = parse_float(row.get(ACHIEVED_OCC, '0'))
        block_sz = parse_float(row.get(BLOCK_SIZE, '0'))
        grid_sz  = parse_float(row.get(GRID_SIZE, '0'))

        theoretical, bottleneck, lim_reg_pct, lim_smem_pct = _theoretical_occ_pct(row)
        efficiency = achieved / theoretical * 100 if theoretical > 0 else 0

        entries.append({
            'name': shorten_kernel_name(name),
            'dur_ms': dur,
            'achieved': achieved,
            'theoretical': theoretical,
            'efficiency': efficiency,
            'bottleneck': bottleneck,
            'lim_reg': lim_reg_pct,
            'lim_smem': lim_smem_pct,
            'block_size': int(block_sz),
            'grid_size': int(grid_sz),
        })

    entries.sort(key=lambda x: x['dur_ms'], reverse=True)
    return entries[:top_n]


def weighted_summary(rows, units_map):
    """Duration-weighted achieved and theoretical occupancy."""
    total_dur = 0
    sum_achieved = 0
    sum_theoretical = 0
    n = 0

    for row in rows:
        name = row.get(KERNEL_NAME, "")
        if any(pat in name for pat in EXCLUDE_PATTERNS):
            continue
        dur = get_duration_ms(row, units_map)
        if dur <= 0:
            continue
        achieved = parse_float(row.get(ACHIEVED_OCC, '0'))
        theoretical, _, _, _ = _theoretical_occ_pct(row)

        total_dur += dur
        sum_achieved += dur * achieved
        sum_theoretical += dur * theoretical
        n += 1

    if total_dur > 0:
        avg_achieved = sum_achieved / total_dur
        avg_theoretical = sum_theoretical / total_dur
        return {
            'achieved': avg_achieved,
            'theoretical': avg_theoretical,
            'efficiency': avg_achieved / avg_theoretical * 100
                          if avg_theoretical > 0 else 0,
            'total_ms': total_dur,
            'n_kernels': n,
        }
    return None


def print_report(model_key, batch, top_n):
    print(f"\n{'='*100}")
    print(f"  {model_key} | B={batch} | Occupancy Analysis (achieved vs theoretical)")
    print(f"{'='*100}")

    for backend in ['sengine', 'trt']:
        report = ncu_report_path(model_key, batch, backend)
        rows, units_map = extract_csv(report)
        if not rows:
            print(f"\n  [{backend}] no data")
            continue

        summary = weighted_summary(rows, units_map)
        if not summary:
            print(f"\n  [{backend}] no valid kernels")
            continue

        print(f"\n  [{backend}] {summary['n_kernels']} kernels, "
              f"total = {summary['total_ms']:.3f}ms")
        print(f"  Weighted avg:  achieved = {summary['achieved']:.1f}%  "
              f"theoretical = {summary['theoretical']:.1f}%  "
              f"efficiency = {summary['efficiency']:.1f}%")

        entries = analyze_occupancy(rows, units_map, top_n)
        if not entries:
            continue

        print(f"\n  {'#':<3} {'Duration':>9} {'Achv%':>6} {'Theo%':>6} {'Eff%':>6} "
              f"{'Limit':>5} {'Reg%':>5} {'Smem%':>6} {'Blk':>5} {'Grid':>7}  Kernel")
        print(f"  {'-'*3} {'-'*9} {'-'*6} {'-'*6} {'-'*6} "
              f"{'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*7}  {'-'*50}")

        for i, e in enumerate(entries, 1):
            if e['dur_ms'] >= 1.0:
                dur_str = f"{e['dur_ms']:.2f}ms"
            elif e['dur_ms'] >= 0.001:
                dur_str = f"{e['dur_ms']*1000:.1f}us"
            else:
                dur_str = f"{e['dur_ms']*1e6:.0f}ns"

            print(f"  {i:<3} {dur_str:>9} {e['achieved']:>5.1f} {e['theoretical']:>5.1f} "
                  f"{e['efficiency']:>5.1f} "
                  f"{e['bottleneck']:>5} {e['lim_reg']:>4.0f} {e['lim_smem']:>5.0f} "
                  f"{e['block_size']:>5} {e['grid_size']:>7}  {e['name']}")

    # Side-by-side comparison
    se_summary = weighted_summary(
        *extract_csv(ncu_report_path(model_key, batch, 'sengine')))
    tr_summary = weighted_summary(
        *extract_csv(ncu_report_path(model_key, batch, 'trt')))

    if se_summary and tr_summary:
        print(f"\n  {'Comparison':<20} {'sengine':>12} {'TensorRT':>12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        print(f"  {'Achieved occ':<20} {se_summary['achieved']:>10.1f}% {tr_summary['achieved']:>10.1f}%")
        print(f"  {'Theoretical occ':<20} {se_summary['theoretical']:>10.1f}% {tr_summary['theoretical']:>10.1f}%")
        print(f"  {'Efficiency':<20} {se_summary['efficiency']:>10.1f}% {tr_summary['efficiency']:>10.1f}%")
        print(f"  {'Total GPU time':<20} {se_summary['total_ms']:>10.3f}ms {tr_summary['total_ms']:>10.3f}ms")


def main():
    parser = argparse.ArgumentParser(description="Occupancy analysis from ncu reports")
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--model', type=str, default=None, choices=list(MODELS.keys()))
    parser.add_argument('--batch', type=int, default=None)
    args = parser.parse_args()

    # Discover all available reports
    pairs = []
    if args.model and args.batch:
        pairs.append((args.model, args.batch))
    else:
        for model_key in MODELS:
            for batch in BATCH_SIZES:
                has_any = any(
                    os.path.exists(ncu_report_path(model_key, batch, be) + ".ncu-rep")
                    for be in ['sengine', 'trt']
                )
                if has_any:
                    pairs.append((model_key, batch))
        # Also scan for non-standard batch sizes
        if os.path.isdir(NCU_REPORTS_DIR):
            for fname in os.listdir(NCU_REPORTS_DIR):
                m = re.match(r'(.+)_B(\d+)_(sengine|trt)\.ncu-rep$', fname)
                if m:
                    mk, bs = m.group(1), int(m.group(2))
                    if mk in MODELS and (mk, bs) not in pairs:
                        pairs.append((mk, bs))

    if not pairs:
        print("No ncu reports found.")
        return

    for model_key, batch in pairs:
        print_report(model_key, batch, args.top)

    # Final cross-model summary
    print(f"\n{'='*100}")
    print(f"  Occupancy Summary: sengine vs TensorRT")
    print(f"{'='*100}")
    print(f"  {'Model':<25} {'B':>3} {'Backend':<9} "
          f"{'Achieved':>9} {'Theoret':>9} {'Effic':>7} {'Kernels':>8} {'Time':>10}")
    print(f"  {'-'*25} {'-'*3} {'-'*9} "
          f"{'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*10}")

    for model_key, batch in pairs:
        for backend in ['sengine', 'trt']:
            report = ncu_report_path(model_key, batch, backend)
            rows, units_map = extract_csv(report)
            if not rows:
                continue
            s = weighted_summary(rows, units_map)
            if not s:
                continue
            print(f"  {model_key:<25} {batch:>3} {backend:<9} "
                  f"{s['achieved']:>7.1f}% {s['theoretical']:>7.1f}% "
                  f"{s['efficiency']:>5.1f}% "
                  f"{s['n_kernels']:>8} {s['total_ms']:>8.3f}ms")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
