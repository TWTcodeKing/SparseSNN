#!/usr/bin/env python3
"""Collect the logs of scripts/bench_new_workloads.sh into one latency table.

Usage: python scripts/parse_bench_new_workloads.py [output/bench_new_workloads]
"""
import glob
import os
import re
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'output/bench_new_workloads'
rows = {}  # (model, tool) -> {B: ms}
for log in sorted(glob.glob(os.path.join(out_dir, '*.log'))):
    name = os.path.basename(log)[:-4]
    if name == 'driver':
        continue
    tool, model = name.rsplit('_snn_', 1)
    model = 'snn_' + model
    txt = open(log, errors='ignore').read()
    res = {}
    if tool.startswith('sengine'):
        # "  [slicer, B=4] 1.234 ms | 3241 img/s | ..." (per-B lines) or summary table
        for m in re.finditer(r'\[fusion=slicer, B=(\d+)\]\s+([\d.]+) ms', txt):
            res[int(m.group(1))] = float(m.group(2))
    elif tool.startswith('trt'):
        for m in re.finditer(r'B=(\d+)\s+([\d.]+)ms\s+([\d.]+)ms', txt):
            res[int(m.group(1))] = float(m.group(2))
    elif tool.startswith('inductor'):
        for m in re.finditer(r'^\s+(\d+)\s+([\d.]+)\s*ms?\s+([\d.]+)', txt, re.M):
            res[int(m.group(1))] = float(m.group(2))
        if not res:
            for m in re.finditer(r'B=(\d+)[^\n]*?([\d.]+) ms', txt):
                res[int(m.group(1))] = float(m.group(2))
    rows[(model, tool)] = res

batches = [4, 8, 16, 32]
for model in ('snn_vgg16', 'snn_vgg9'):
    print(f"\n{model}  (latency ms per batch, T=4)")
    print(f"  {'tool':<16}" + ''.join(f"{'B=' + str(b):>10}" for b in batches))
    for (m, tool), res in sorted(rows.items()):
        if m != model:
            continue
        print(f"  {tool:<16}" + ''.join(f"{res.get(b, float('nan')):>10.3f}" for b in batches))
    se = rows.get((model, 'sengine_fp16'), {})
    for base in ('trt_fp16', 'trt_fp32', 'inductor_fp16', 'inductor_fp32'):
        r = rows.get((model, base), {})
        if se and r:
            print(f"  speedup vs {base:<13}" + ''.join(
                f"{(r[b] / se[b]) if (b in r and b in se and se[b] > 0) else float('nan'):>10.2f}x" for b in batches))
