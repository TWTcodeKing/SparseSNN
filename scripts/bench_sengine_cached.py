#!/usr/bin/env python3
"""Build a sengine from a plugin ONNX with cached kernels + validated fusion, then benchmark.

Prints one machine-readable line consumed by scripts/export_new_workload_results.py:
    RESULT sengine_<precision> <tag> B=<B> <ms>ms fused=<n> <img/s> img/s

Usage:
    python scripts/bench_sengine_cached.py sengine/exports/snn_vgg16_ut_har_plugin.onnx 4 [warmup] [iters]
The fusion recommendation .cache/fusion_rec_<tag>_T<T>_B<B>.json (from the
bench_sengine_latency.py pre-pass) is used when present; otherwise the slicer
decides without validation. T defaults to 4 (env SENGINE_T overrides).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    onnx_path = sys.argv[1]
    B = int(sys.argv[2])
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    T = int(os.environ.get('SENGINE_T', 4))
    precision = os.environ.get('SENGINE_PRECISION', 'fp16')

    tag = os.path.basename(onnx_path)
    for suffix in ('_plugin.onnx', '.onnx'):
        if tag.endswith(suffix):
            tag = tag[:-len(suffix)]
            break
    rec_path = os.path.join('.cache', f'fusion_rec_{tag}_T{T}_B{B}.json')
    if not os.path.exists(rec_path):
        print(f"  no fusion recommendation at {rec_path}; slicer runs unvalidated")
        rec_path = None

    import sengine
    from sengine.ir import KernelVariant
    e = sengine.build(onnx_path, T=T, batch_size=B, fusion='slicer', autotune=False,
                      fusion_rec=rec_path, precision=precision)
    n_fused = sum(1 for n in e.ir.nodes.values()
                  if n.assigned_kernel in (KernelVariant.TileLangFusedConvBNIF,
                                           KernelVariant.TileLangFusedConv1x1BNIF,
                                           KernelVariant.TileLangLinearBNLIF,
                                           KernelVariant.TileLangFusedGroupedConvBNLIF))
    ms = e.benchmark(warmup=warmup, iters=iters)
    fps = 1000.0 / ms * B if ms > 0 else 0.0
    print(f"RESULT sengine_{precision} {tag} B={B} {ms:.3f}ms fused={n_fused} {fps:.0f} img/s")


if __name__ == '__main__':
    main()
