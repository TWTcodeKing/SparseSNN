#!/usr/bin/env python3
"""Export the UT-HAR / UrbanSound8K experiment logs (RTX 4090) to CSV.

Writes into output/bench_new_workloads/:
  latency_4090.csv      sengine / TensorRT / Inductor latency per batch size
  correctness_4090.csv  engine-vs-PyTorch agreement and accuracy (verify_workloads.py logs)
  kernel_breakdown_4090.csv  TensorRT per-category GPU time (nsys) and sengine per-layer kernel time
Usage: python scripts/export_new_workload_results.py [--verify-logs DIR]
"""
import argparse
import csv
import glob
import os
import re
import subprocess

OUT = 'output/bench_new_workloads'
MODEL_INFO = {
    'snn_vgg16': ('ut_har', '1x250x90', 7),
    'snn_vgg9': ('urbansound8k', '1x64x176', 10),
}


def latency_rows():
    rows = []
    # clean interleaved run (GPU 6, idle): sengine fp16, trt fp16, inductor fp16
    log = os.path.join(OUT, 'remeasure_gpu6.log')
    for line in open(log):
        if not line.startswith('RESULT'):
            continue
        parts = line.split()
        tool, model = parts[1], parts[2]
        model = 'snn_vgg16' if model.startswith('snn_vgg16') else 'snn_vgg9'
        b = next(p for p in parts if p.startswith('B='))[2:]
        i = parts.index(next(p for p in parts if p.startswith('B=')))
        lat = float(parts[i + 1].replace('ms', ''))
        std = ''
        thr = ''
        if tool == 'sengine_fp16':
            thr = parts[-2]
        else:
            std = parts[i + 2].replace('ms', '')
            thr = parts[i + 3].replace('/s', '')
        base, prec = tool.rsplit('_', 1)
        mode = {'sengine': 'slicer+validated', 'inductor': 'reduce-overhead', 'trt': ''}.get(base, '')
        rows.append((model, base, prec, mode, int(b), lat, std, thr, 6, 'remeasure_gpu6.log'))
    # extra baselines: ONNX Runtime CUDA EP and Inductor max-autotune (GPU 6)
    for log in sorted(glob.glob(os.path.join(OUT, 'extra_gpu*.log'))):
        gpu = int(re.search(r'extra_gpu(\d+)', log).group(1))
        for line in open(log):
            if not line.startswith('RESULT'):
                continue
            parts = line.split()
            tool, model = parts[1], parts[2]
            base, prec = tool.rsplit('_', 1)
            b = int(next(p for p in parts if p.startswith('B='))[2:])
            if 'FAILED' in line:
                rows.append((model, base, prec, 'max-autotune' if 'maxautotune' in base else 'cuda_ep graph_opt=all',
                             b, float('nan'), '', '', gpu, os.path.basename(log)))
                continue
            i = parts.index(next(p for p in parts if p.startswith('B=')))
            lat = float(parts[i + 1].replace('ms', '')); std = parts[i + 2].replace('ms', ''); thr = parts[i + 3].replace('/s', '')
            base = 'inductor' if base.startswith('inductor') else base
            mode = 'max-autotune' if 'maxautotune' in tool else 'cuda_ep graph_opt=all'
            rows.append((model, base, prec, mode, b, lat, std, thr, gpu, os.path.basename(log)))
    # fp32 baselines (GPU 1)
    for tool, pat in (('trt', 'trt_fp32real_{m}.log'), ('inductor', 'inductor_fp32_{m}.log')):
        for model in MODEL_INFO:
            path = os.path.join(OUT, pat.format(m=model))
            if not os.path.exists(path):
                continue
            for m in re.finditer(r'^\s+B=(\d+)\s+([\d.]+)ms\s+([\d.]+)ms\s+(\d+)/s', open(path).read(), re.M):
                rows.append((model, tool, 'fp32', 'reduce-overhead' if tool == 'inductor' else '',
                             int(m.group(1)), float(m.group(2)), m.group(3), m.group(4), 1, os.path.basename(path)))
    rows.sort(key=lambda r: (r[0], r[4], r[1], r[3], r[2]))
    with open(os.path.join(OUT, 'latency_4090.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'dataset', 'input_shape', 'T', 'batch', 'tool', 'precision', 'mode',
                    'latency_ms', 'std_ms', 'throughput_img_per_s', 'gpu_index', 'source_log'])
        for model, tool, prec, mode, b, lat, std, thr, gpu, src in rows:
            ds, shape, _ = MODEL_INFO[model]
            w.writerow([model, ds, shape, 4, b, tool, prec, mode, 'FAILED' if lat != lat else f'{lat:.3f}', std, thr, gpu, src])
    return len(rows)


def correctness_rows(verify_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(verify_dir, 'final_*.log'))):
        txt = open(path, errors='ignore').read()
        m = re.search(r'engine=(\w+) model=(\w+) dataset=(\w+) T=(\d+) precision=(\w+)', txt)
        if not m:
            continue
        engine, model, ds, T, prec = m.groups()
        if engine == 'sengine_cpu':
            prec = 'fp32'
        def grab(pat, default=''):
            g = re.search(pat, txt)
            return g.group(1) if g else default
        rows.append([engine, model, ds, T, prec,
                     grab(r'vs PyTorch FP32: top1 agree=([\d.]+)'), grab(r'vs PyTorch FP32: top1 agree=[\d.]+ mean cos=([\d.]+)'),
                     grab(r'vs PyTorch FP16: top1 agree=([\d.]+)'), grab(r'vs PyTorch FP16: top1 agree=[\d.]+ mean cos=([\d.]+)'),
                     grab(r'min cos=([\d.-]+)'), grab(r'max\|d\|=([\d.]+)'),
                     grab(r'PyTorch FP16 vs FP32 \(reference tolerance\): top1 agree=([\d.]+)'),
                     grab(r'test acc: engine=([\d.]+)'), grab(r'torch_fp32=([\d.]+)'), grab(r'torch_fp16=([\d.]+)'),
                     grab(r'Engine test acc: [\d.]+ \(n=(\d+)'), grab(r'VERDICT: (\w+)')])
    with open(os.path.join(OUT, 'correctness_4090.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['engine', 'model', 'dataset', 'T', 'engine_precision', 'top1_agree_vs_fp32', 'mean_cos_vs_fp32',
                    'top1_agree_vs_fp16', 'mean_cos_vs_fp16', 'min_cos_vs_fp16', 'max_abs_diff_vs_fp16',
                    'torch_fp16_vs_fp32_agree', 'engine_test_acc', 'torch_fp32_test_acc', 'torch_fp16_test_acc',
                    'test_samples', 'strict_verdict'])
        w.writerows(rows)
    return len(rows)


def breakdown_rows():
    rows = []
    # TensorRT nsys traces: category split per iteration (20 launches per trace)
    for rep in sorted(glob.glob('profiles/trt_snn_*.nsys-rep')):
        m = re.match(r'trt_(snn_vgg\d+)_b(\d+)', os.path.basename(rep))
        if not m:
            continue
        model, b = m.group(1), int(m.group(2))
        out = subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv', rep],
                             capture_output=True, text=True).stdout
        cat = {'conv_gemm': 0, 'neuron_myelin': 0, 'layout_transpose': 0, 'pool': 0, 'other': 0}
        for line in out.splitlines():
            parts = line.split(',')
            if len(parts) < 9 or not parts[1].isdigit():
                continue
            ns, name = int(parts[1]), ','.join(parts[8:])
            if re.search(r'xmma_fprop|gemm', name):
                cat['conv_gemm'] += ns
            elif '__myl' in name:
                cat['neuron_myelin'] += ns
            elif re.search(r'Tonchw|Tonhwc|permutation', name):
                cat['layout_transpose'] += ns
            elif 'pool' in name:
                cat['pool'] += ns
            else:
                cat['other'] += ns
        for k, v in cat.items():
            rows.append(['tensorrt_fp16', model, b, 'category', k, '', f'{v / 20 / 1e3:.1f}', ''])
    # sengine standalone per-layer kernel profiles at B=4 (analyze_kernel_latency logs)
    for model in MODEL_INFO:
        path = os.path.join(OUT, f'kernels_{model}_b4.log')
        if not os.path.exists(path):
            continue
        for line in open(path, errors='ignore'):
            # the analyzer truncates the shape column, so take it as free text
            m = re.match(r'\s+(\d+)\s+Conv\+BN\s+(\S+)\s+(.+?)\s+([\d.]+)\s+([\d.]+)\s*$', line)
            if m:
                nid, kv, shape, gflops, us = m.groups()
                tflops = float(gflops) / (float(us) * 1e-6) / 1e3 if float(us) > 0 else 0
                rows.append(['sengine_fp16', model, 4, 'layer', f'node{nid}:{kv}', shape.strip(), us, f'{tflops:.1f}'])
    with open(os.path.join(OUT, 'kernel_breakdown_4090.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tool', 'model', 'batch', 'kind', 'name', 'shape', 'time_us_per_iter', 'tflops'])
        w.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-logs', default=OUT, help='directory holding final_*.log from verify_workloads.py')
    a = ap.parse_args()
    print('latency rows:', latency_rows())
    print('correctness rows:', correctness_rows(a.verify_logs))
    print('breakdown rows:', breakdown_rows())
