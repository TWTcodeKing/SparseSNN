#!/usr/bin/env python3
"""
bench_snn_vgg9_cpu.py — Cross-platform SNN-VGG-9 CPU inference benchmark.

Benchmarks sengine_cpu (MLAS), ONNX Runtime, OpenVINO, and ncnn
on SNN-VGG-9 (T=4, B=1, 112×128 NTU-Fi-HumanID input, 14 classes).

Portable across: AMD EPYC (AVX2/FMA3), Intel Xeon (AVX-512), ARM Cortex-A7 (NEON).

Usage:
    python scripts/bench_snn_vgg9_cpu.py [--threads 1,4,8] [--warmup 100] [--iters 500]
    python scripts/bench_snn_vgg9_cpu.py --frameworks sengine,ort   # subset only
    python scripts/bench_snn_vgg9_cpu.py --export-only               # just export models
"""

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
T = 4
B = 1
IMG_C, IMG_H, IMG_W = 3, 112, 128
NUM_CLASSES = 14

VGG9_LAYERS = [
    # (C_in, C_out, H, W, has_pool) — shapes for 112×128 input
    (3,   64,  112, 128, True),   # → pool → 56×64
    (64,  128,  56,  64, True),   # → pool → 28×32
    (128, 256,  28,  32, False),
    (256, 256,  28,  32, True),   # → pool → 14×16
    (256, 512,  14,  16, False),
    (512, 512,  14,  16, True),   # → pool → 7×8
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _REPO_ROOT / ".cache" / "snn_vgg9_cpu_bench"   # gitignored


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def bench(fn, warmup=100, iters=500):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def detect_platform():
    arch = platform.machine()
    if arch in ("x86_64", "AMD64"):
        isa = "x86_64"
        try:
            with open("/proc/cpuinfo") as f:
                info = f.read()
            if "avx512" in info.lower():
                isa_detail = "AVX-512"
            elif "avx2" in info.lower():
                isa_detail = "AVX2/FMA3"
            else:
                isa_detail = "SSE4"
        except FileNotFoundError:
            isa_detail = "x86_64"
    elif arch in ("aarch64", "armv7l"):
        isa = "arm"
        isa_detail = "NEON" if arch == "aarch64" else "ARMv7/NEON"
    else:
        isa = arch
        isa_detail = arch

    cpu_name = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass

    ncores = os.cpu_count() or 1
    return isa, isa_detail, cpu_name, ncores


def get_numa_prefix():
    """Return numactl prefix if available, else empty list."""
    if shutil.which("numactl"):
        return ["numactl", "--cpunodebind=0", "--membind=0"]
    return []


# ──────────────────────────────────────────────────────────────────────
# Model definition (shared across all frameworks)
# ──────────────────────────────────────────────────────────────────────
def build_snn_vgg9_pytorch(export_mode="standard"):
    """Build SNN-VGG-9 in PyTorch. Two export modes:
    - 'standard': uses (T*B, C, H, W) input with SeqToANN-style processing.
      Suitable for ONNX export (ORT, OpenVINO).
    - 'unrolled': T-loop unrolled in graph, input (1, C, H, W).
      Suitable for ncnn (no batch dim).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LIFNeuron_Standard(nn.Module):
        """LIF for standard ONNX export (T*B merged in batch dim)."""
        def __init__(self, decay=0.5, thr=1.0):
            super().__init__()
            self.decay = decay
            self.thr = thr

        def forward(self, x):
            # x: (T*B, C, H, W)
            TB = x.shape[0]
            frame_t = T
            frame_b = TB // frame_t
            # reshape to (T, B, C, H, W)
            x5 = x.view(frame_t, frame_b, *x.shape[1:])
            spikes = []
            mem = torch.zeros_like(x5[0])
            for t in range(frame_t):
                mem = self.decay * mem + (1.0 - self.decay) * x5[t]
                spike = torch.clamp(
                    F.relu(mem - self.thr + 1e-6) * 1e6, max=1.0)
                mem = (1.0 - spike) * mem
                spikes.append(spike)
            return torch.cat(spikes, dim=0)  # (T*B, C, H, W)

    class LIFNeuron_Unrolled(nn.Module):
        """LIF for ncnn (T-loop unrolled, ncnn-compatible ops)."""
        def __init__(self, decay=0.5, thr=1.0):
            super().__init__()
            self.decay = decay
            self.thr = thr

        def forward(self, *inputs):
            spikes = []
            mem = torch.zeros_like(inputs[0])
            for x in inputs:
                mem = self.decay * mem + (1.0 - self.decay) * x
                spike = torch.clamp(mem - self.thr + 1.0, 0.0, 1.0).floor()
                mem = (1.0 - spike) * mem
                spikes.append(spike)
            return tuple(spikes)

    class SNNVGG9_Standard(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for c_in, c_out, _, _, has_pool in VGG9_LAYERS:
                layers.append(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(c_out))
                layers.append(LIFNeuron_Standard())
                if has_pool:
                    layers.append(nn.MaxPool2d(2))
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(512, NUM_CLASSES)

        def forward(self, x):
            import torch.nn.functional as F
            out = self.features(x)  # (T*B, 512, 2, 2)
            # temporal mean
            out5 = out.view(T, B, *out.shape[1:])
            mean_out = out5.mean(dim=0)  # (B, 512, 2, 2)
            gap = F.adaptive_avg_pool2d(mean_out, 1).flatten(1)
            return self.classifier(gap)

    class SNNVGG9_Unrolled(nn.Module):
        def __init__(self):
            super().__init__()
            for i, (c_in, c_out, _, _, _) in enumerate(VGG9_LAYERS):
                setattr(self, f"conv{i}", nn.Conv2d(c_in, c_out, 3, padding=1, bias=False))
                setattr(self, f"bn{i}", nn.BatchNorm2d(c_out))
                setattr(self, f"lif{i}", LIFNeuron_Unrolled())
            self.pool = nn.MaxPool2d(2)
            self.classifier = nn.Linear(512, NUM_CLASSES)

        def _block(self, idx, frames):
            import torch.nn.functional as F
            conv = getattr(self, f"conv{idx}")
            bn = getattr(self, f"bn{idx}")
            lif = getattr(self, f"lif{idx}")
            outs = tuple(bn(conv(f)) for f in frames)
            spikes = lif(*outs)
            if VGG9_LAYERS[idx][4]:  # has_pool
                spikes = tuple(self.pool(s) for s in spikes)
            return spikes

        def forward(self, x):
            import torch.nn.functional as F
            frames = tuple(x for _ in range(T))
            for i in range(6):
                frames = self._block(i, frames)
            mean_out = (frames[0] + frames[1] + frames[2] + frames[3]) * 0.25
            gap = F.adaptive_avg_pool2d(mean_out, 1).flatten(1)
            return self.classifier(gap)

    if export_mode == "standard":
        return SNNVGG9_Standard().eval()
    else:
        return SNNVGG9_Unrolled().eval()


# ──────────────────────────────────────────────────────────────────────
# Model export
# ──────────────────────────────────────────────────────────────────────
def export_models():
    """Export ONNX (for ORT/OpenVINO) and ncnn models."""
    import torch

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = CACHE_DIR / "snn_vgg9.onnx"
    ncnn_param = CACHE_DIR / "snn_vgg9.ncnn.param"
    ncnn_bin = CACHE_DIR / "snn_vgg9.ncnn.bin"

    # ── ONNX (standard mode: input = (T*B, C, H, W)) ──
    if not onnx_path.exists():
        print("[export] Building ONNX model (standard mode)...")
        model = build_snn_vgg9_pytorch("standard")
        dummy = torch.randn(T * B, IMG_C, IMG_H, IMG_W)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=17,
            dynamic_axes=None)
        print(f"  -> {onnx_path}")
    else:
        print(f"[export] ONNX exists: {onnx_path}")

    # ── ncnn (unrolled mode: input = (1, C, H, W)) ──
    if not ncnn_param.exists():
        print("[export] Building ncnn model (T-unrolled mode)...")
        model = build_snn_vgg9_pytorch("unrolled")
        dummy = torch.randn(1, IMG_C, IMG_H, IMG_W)
        ts = torch.jit.trace(model, dummy)
        ts_path = CACHE_DIR / "snn_vgg9_unrolled.pt"
        ts.save(str(ts_path))

        if shutil.which("pnnx"):
            r = subprocess.run(
                ["pnnx", str(ts_path), f"inputshape=[1,{IMG_C},{IMG_H},{IMG_W}]"],
                capture_output=True, cwd=str(CACHE_DIR))
            # pnnx outputs to same dir as input
            pnnx_param = CACHE_DIR / "snn_vgg9_unrolled.ncnn.param"
            pnnx_bin = CACHE_DIR / "snn_vgg9_unrolled.ncnn.bin"
            if pnnx_param.exists():
                pnnx_param.rename(ncnn_param)
                pnnx_bin.rename(ncnn_bin)
                print(f"  -> {ncnn_param}")
            else:
                print("  [WARN] pnnx export failed, ncnn benchmark will be skipped")
        else:
            print("  [WARN] pnnx not found, ncnn benchmark will be skipped")
            print("         Install via: pip install pnnx ncnn")
    else:
        print(f"[export] ncnn exists: {ncnn_param}")

    return onnx_path, ncnn_param, ncnn_bin


# ──────────────────────────────────────────────────────────────────────
# Framework benchmarks
# ──────────────────────────────────────────────────────────────────────
def bench_sengine(n_threads, warmup, iters):
    """Benchmark sengine_cpu MLAS kernel (TB-merged Conv + fused BN+LIF)."""
    so_path = _REPO_ROOT / "sengine_cpu" / "csrc" / "libmlas_conv_bn_lif.so"
    if not so_path.exists():
        return None, ("libmlas_conv_bn_lif.so not found; build it with the g++ command in "
                      "sengine_cpu/BUILD.md (MLAS kernel appendix)")

    lib = ctypes.CDLL(str(so_path))
    lib.mlas_conv2d_bn_lif.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 10 +
        [ctypes.c_float] * 3 + [ctypes.c_int])
    lib.mlas_conv2d_bn_lif.restype = None

    fp = lambda a: a.ctypes.data
    ci = ctypes.c_int
    cf = ctypes.c_float

    np.random.seed(42)
    total_ms = 0.0
    N = T * B

    for c_in, c_out, h, w, has_pool in VGG9_LAYERS:
        oh, ow = h, w
        spt = B * oh * ow
        weight = np.ascontiguousarray(np.random.randn(c_out, c_in, 3, 3).astype("f") * 0.02)
        scale = np.ones(c_out, "f")
        bias = np.zeros(c_out, "f")
        inp = np.ascontiguousarray(np.random.randn(N, c_in, h, w).astype("f") * 0.1)
        mem = np.zeros(spt * c_out, "f")
        spk = np.zeros(c_out * N * oh * ow, "f")

        def run(_inp=inp, _w=weight, _sc=scale, _bi=bias, _mem=mem, _spk=spk,
                _cin=c_in, _cout=c_out, _h=h, _w2=w):
            _mem[:] = 0
            lib.mlas_conv2d_bn_lif(
                fp(_inp), fp(_w), fp(_sc), fp(_bi), fp(_mem), fp(_spk),
                ci(B), ci(_cin), ci(_h), ci(_w2), ci(_cout), ci(T),
                ci(3), ci(3), ci(1), ci(1),
                cf(1.0), cf(0.5), cf(0.5), ci(n_threads))

        total_ms += bench(run, warmup, iters)

    return total_ms, None


def bench_ort(n_threads, warmup, iters, onnx_path):
    """Benchmark ONNX Runtime CPU."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime not installed"

    if not onnx_path.exists():
        return None, "ONNX model not found"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    inp = np.random.randn(T * B, IMG_C, IMG_H, IMG_W).astype("f")
    input_name = sess.get_inputs()[0].name

    def run():
        sess.run(None, {input_name: inp})

    ms = bench(run, warmup, iters)
    return ms, None


def bench_openvino(n_threads, warmup, iters, onnx_path):
    """Benchmark OpenVINO CPU."""
    try:
        from openvino import Core
    except ImportError:
        return None, "openvino not installed"

    if not onnx_path.exists():
        return None, "ONNX model not found"

    core = Core()
    model = core.read_model(str(onnx_path))
    config = {"INFERENCE_NUM_THREADS": str(n_threads)}
    compiled = core.compile_model(model, "CPU", config)
    req = compiled.create_infer_request()

    inp = np.random.randn(T * B, IMG_C, IMG_H, IMG_W).astype("f")

    def run():
        req.infer({"input": inp})

    ms = bench(run, warmup, iters)
    return ms, None


def bench_ncnn(n_threads, warmup, iters, ncnn_param, ncnn_bin):
    """Benchmark ncnn CPU."""
    try:
        import ncnn
    except ImportError:
        return None, "ncnn not installed"

    if not ncnn_param.exists():
        return None, "ncnn model not found"

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = n_threads
    net.load_param(str(ncnn_param))
    net.load_model(str(ncnn_bin))

    inp = np.random.randn(IMG_C, IMG_H, IMG_W).astype("f")
    mat = ncnn.Mat(inp)

    def run():
        ex = net.create_extractor()
        ex.input("in0", mat)
        ex.extract("out0")

    ms = bench(run, warmup, iters)
    return ms, None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SNN-VGG-9 CPU inference benchmark")
    parser.add_argument("--threads", default="1,4",
                        help="Comma-separated thread counts (default: 1,4)")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--frameworks", default="sengine,ort,openvino,ncnn",
                        help="Comma-separated frameworks to benchmark")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export models, don't benchmark")
    parser.add_argument("--numa", type=int, default=None, metavar="NODE",
                        help="Pin to NUMA node via numactl (re-execs process)")
    parser.add_argument("--_numa_pinned", action="store_true",
                        help=argparse.SUPPRESS)  # internal flag
    args = parser.parse_args()

    # Auto re-exec under numactl if --numa given and not already pinned
    if args.numa is not None and not args._numa_pinned:
        numa_bin = shutil.which("numactl")
        if not numa_bin:
            print("[WARN] numactl not found, running without NUMA pinning")
        else:
            cmd = [numa_bin,
                   f"--cpunodebind={args.numa}",
                   f"--membind={args.numa}",
                   sys.executable] + sys.argv + ["--_numa_pinned"]
            os.execvp(cmd[0], cmd)

    thread_counts = [int(t) for t in args.threads.split(",")]
    frameworks = [f.strip() for f in args.frameworks.split(",")]

    isa, isa_detail, cpu_name, ncores = detect_platform()
    numa_status = f"NUMA node {args.numa}" if args._numa_pinned else "not pinned"

    print("=" * 75)
    print(" SNN-VGG-9 CPU Inference Benchmark")
    print("=" * 75)
    print(f"  CPU:       {cpu_name}")
    print(f"  ISA:       {isa_detail}")
    print(f"  Cores:     {ncores}")
    print(f"  Platform:  {platform.system()} {platform.machine()}")
    print(f"  NUMA:      {numa_status}")
    print(f"  Model:     VGG-9 SNN (T={T}, B={B}, {IMG_H}x{IMG_W}, {NUM_CLASSES}cls)")
    print(f"  Threads:   {thread_counts}")
    print(f"  Warmup:    {args.warmup}, Iters: {args.iters}")
    print("=" * 75)
    print()

    # Export models
    onnx_path, ncnn_param, ncnn_bin = export_models()
    print()

    if args.export_only:
        print("Export done. Exiting.")
        return

    # Set OMP threads via env (affects MLAS internally)
    os.environ["OMP_NUM_THREADS"] = str(max(thread_counts))

    # Run benchmarks
    # results[framework][n_threads] = ms
    results = {}

    for fw in frameworks:
        results[fw] = {}
        for nt in thread_counts:
            os.environ["OMP_NUM_THREADS"] = str(nt)

            if fw == "sengine":
                ms, err = bench_sengine(nt, args.warmup, args.iters)
            elif fw == "ort":
                ms, err = bench_ort(nt, args.warmup, args.iters, onnx_path)
            elif fw == "openvino":
                ms, err = bench_openvino(nt, args.warmup, args.iters, onnx_path)
            elif fw == "ncnn":
                ms, err = bench_ncnn(nt, args.warmup, args.iters, ncnn_param, ncnn_bin)
            else:
                ms, err = None, f"unknown framework: {fw}"

            if err:
                print(f"  [{fw}] {nt}T: SKIP ({err})")
                results[fw][nt] = None
            else:
                print(f"  [{fw}] {nt}T: {ms:.2f}ms")
                results[fw][nt] = ms

    # ── Print results table ──
    print()
    print("=" * 75)
    print(" Results: SNN-VGG-9 (T=4, B=1, 112x128)")
    print("=" * 75)

    # Header
    hdr = f"{'Framework':<20}"
    for nt in thread_counts:
        hdr += f"  {'%dT' % nt:>8}"
    print(hdr)
    print("-" * 75)

    # Find baseline (sengine 1T if available, else first available)
    baseline_ms = None
    if "sengine" in results and thread_counts[0] in results["sengine"]:
        baseline_ms = results["sengine"][thread_counts[0]]

    for fw in frameworks:
        row = f"  {fw:<18}"
        for nt in thread_counts:
            ms = results.get(fw, {}).get(nt)
            if ms is not None:
                row += f"  {ms:>7.2f}ms"
            else:
                row += f"  {'--':>8}"
        # Add speedup vs sengine 1T
        if baseline_ms and results.get(fw, {}).get(thread_counts[0]):
            ms1t = results[fw][thread_counts[0]]
            ratio = ms1t / baseline_ms
            row += f"  ({ratio:.1f}x vs se)"
        print(row)

    print("-" * 75)

    # Scaling analysis
    if len(thread_counts) > 1:
        print()
        print("Thread scaling (speedup vs 1T):")
        for fw in frameworks:
            ms_1t = results.get(fw, {}).get(thread_counts[0])
            if ms_1t is None:
                continue
            row = f"  {fw:<18}"
            for nt in thread_counts:
                ms = results.get(fw, {}).get(nt)
                if ms is not None:
                    row += f"  {nt}T={ms_1t/ms:.2f}x"
            print(row)

    print()
    print("=" * 75)
    print(f" Platform: {cpu_name} | ISA: {isa_detail}")
    print("=" * 75)


if __name__ == "__main__":
    main()
