"""Benchmark: temporal pipeline overlap strategies for SNN inference.

Tests whether conv_{t+1} and if_t can run in parallel via CUDA stream overlap,
and compares against our fused interleaved kernel.

Three strategies:
  1. Sequential:  conv_t → if_t → conv_{t+1} → if_{t+1} → ...
                  All on one stream.

  2. Pipelined:   conv_{t+1} ∥ if_t via dual CUDA streams + event sync.

  3. Fused:       TileLang interleaved kernel — per-CTA T-loop, membrane
                  in registers, no DRAM round-trip for intermediates.

Usage:
    python experiments/motivation/bench_temporal_pipeline.py
    python experiments/motivation/bench_temporal_pipeline.py --B 4 --T 8
"""

import torch
import torch.cuda
import argparse
import os
import sys

# Force CUDA 12.8 nvcc
_cuda_home = '/usr/local/cuda-12.8'
if os.path.isdir(_cuda_home):
    os.environ['CUDA_HOME'] = _cuda_home
    os.environ['PATH'] = os.path.join(_cuda_home, 'bin') + ':' + os.environ.get('PATH', '')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ─── IF neuron (single fused CUDA kernel) ───

_if_ext = None


def _build_if_kernel():
    global _if_ext
    if _if_ext is not None:
        return _if_ext

    from torch.utils.cpp_extension import load
    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

__global__ void if_neuron_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    float* __restrict__ membrane,
    int N, int spatial, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int s_idx = idx % spatial;
    float h = membrane[s_idx] + __half2float(input[idx]);
    float spike = (h >= threshold) ? 1.0f : 0.0f;
    membrane[s_idx] = (1.0f - spike) * h;
    output[idx] = __float2half(spike);
}

torch::Tensor if_neuron_cuda(torch::Tensor input, torch::Tensor output,
                             torch::Tensor membrane, float threshold) {
    int N = input.numel();
    int spatial = membrane.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if_neuron_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        membrane.data_ptr<float>(), N, spatial, threshold);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("if_neuron", &if_neuron_cuda, "Fused IF neuron (CUDA)");
}
"""
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache', 'bench_if_ext')
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, 'if_neuron_bench.cu')
    with open(src_path, 'w') as f:
        f.write(cuda_src)

    props = torch.cuda.get_device_properties(0)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{props.major}.{props.minor}')

    _if_ext = load(
        name='if_neuron_bench',
        sources=[src_path],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        build_directory=cache_dir,
        verbose=False,
    )
    return _if_ext


# ─── Strategy 1: Sequential (single stream) ───

def run_sequential(ext, data_per_t, weight, bn_scale, bn_bias,
                   membrane, spikes, conv_buf, T, threshold=1.0):
    membrane.zero_()
    for t in range(T):
        torch.mm(data_per_t[t], weight, out=conv_buf)
        conv_buf.mul_(bn_scale).add_(bn_bias)
        ext.if_neuron(conv_buf, spikes[t], membrane, threshold)


# ─── Strategy 2: Pipelined (dual CUDA streams, pre-allocated) ───

class PipelineRunner:
    def __init__(self, T):
        self.stream_conv = torch.cuda.Stream()
        self.stream_if = torch.cuda.Stream()
        # Pre-allocate events
        self.ev_conv = [torch.cuda.Event() for _ in range(T)]
        self.ev_if = [torch.cuda.Event() for _ in range(T)]
        self.T = T

    def run(self, ext, data_per_t, weight, bn_scale, bn_bias,
            membrane, spikes, conv_bufs, threshold=1.0):
        T = self.T
        sc = self.stream_conv
        si = self.stream_if

        membrane.zero_()
        # Make sure both streams see the zero
        self.ev_conv[0].record()
        sc.wait_event(self.ev_conv[0])
        si.wait_event(self.ev_conv[0])

        # Prologue: conv_0
        with torch.cuda.stream(sc):
            torch.mm(data_per_t[0], weight, out=conv_bufs[0])
            conv_bufs[0].mul_(bn_scale).add_(bn_bias)
        self.ev_conv[0].record(sc)

        for t in range(1, T):
            cur = t % 2
            prev = (t - 1) % 2

            # conv_t on conv stream (independent of IF)
            with torch.cuda.stream(sc):
                torch.mm(data_per_t[t], weight, out=conv_bufs[cur])
                conv_bufs[cur].mul_(bn_scale).add_(bn_bias)
            self.ev_conv[t].record(sc)

            # if_{t-1} on IF stream — depends on conv_{t-1} AND if_{t-2} (membrane)
            si.wait_event(self.ev_conv[t - 1])
            with torch.cuda.stream(si):
                ext.if_neuron(conv_bufs[prev], spikes[t - 1], membrane, threshold)
            self.ev_if[t - 1].record(si)

        # Drain: if_{T-1}
        si.wait_event(self.ev_conv[T - 1])
        with torch.cuda.stream(si):
            ext.if_neuron(conv_bufs[(T - 1) % 2], spikes[T - 1], membrane, threshold)
        self.ev_if[T - 1].record(si)

        # Wait for everything
        torch.cuda.current_stream().wait_event(self.ev_if[T - 1])
        torch.cuda.current_stream().wait_event(self.ev_conv[T - 1])


# ─── Fused TileLang kernel ───

def compile_fused_kernel(B, C_in, H, W, F, T_steps):
    try:
        from sengine.kernels.interleaved_templates import conv1x1_bn_if
    except ImportError as e:
        print(f"  TileLang import failed: {e}")
        return None

    M = B * H * W
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    best_bm, best_bn, best_bk = 32, 64, min(32, C_in)
    for bm in [32, 64]:
        for bn in [64, 32]:
            bk = min(32, C_in)
            grid = ((M + bm - 1) // bm) * ((F + bn - 1) // bn)
            if grid >= max(sm_count // 4, 1):
                best_bm, best_bn, best_bk = bm, bn, bk
                break
        else:
            continue
        break

    try:
        kern = conv1x1_bn_if(
            B=B, C_in=C_in, H=H, W=W, F=F, T_steps=T_steps,
            block_M=best_bm, block_N=best_bn, block_K=best_bk,
            num_stages=2, threads=128)
        return kern
    except Exception as e:
        print(f"  TileLang compile failed: {e}")
        return None


# ─── Profiling ───

def profile(fn, n_warmup=200, n_iter=500):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iter):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iter * 1000)

    times.sort()
    return times[1]


# ─── Main ───

def bench_shape(C_in, C_out, H, W, B, T, threshold=1.0):
    M = B * H * W
    dtype = torch.float16
    device = 'cuda'
    ext = _build_if_kernel()

    weight = torch.randn(C_in, C_out, dtype=dtype, device=device)
    bn_scale = torch.ones(1, C_out, dtype=dtype, device=device)
    bn_bias = torch.zeros(1, C_out, dtype=dtype, device=device)
    data_per_t = [torch.randn(M, C_in, dtype=dtype, device=device) for _ in range(T)]

    # ── Sequential ──
    membrane_seq = torch.zeros(M, C_out, dtype=torch.float32, device=device)
    conv_buf = torch.empty(M, C_out, dtype=dtype, device=device)
    spikes_seq = [torch.empty(M, C_out, dtype=dtype, device=device) for _ in range(T)]

    us_seq = profile(lambda: run_sequential(
        ext, data_per_t, weight, bn_scale, bn_bias,
        membrane_seq, spikes_seq, conv_buf, T, threshold))

    # ── Pipelined ──
    membrane_pip = torch.zeros(M, C_out, dtype=torch.float32, device=device)
    conv_bufs = [torch.empty(M, C_out, dtype=dtype, device=device) for _ in range(2)]
    spikes_pip = [torch.empty(M, C_out, dtype=dtype, device=device) for _ in range(T)]
    runner = PipelineRunner(T)

    us_pip = profile(lambda: runner.run(
        ext, data_per_t, weight, bn_scale, bn_bias,
        membrane_pip, spikes_pip, conv_bufs, threshold))

    # ── Fused (TileLang) ──
    us_fused = None
    kern = compile_fused_kernel(B, C_in, H, W, C_out, T)
    if kern is not None:
        data_nhwc = torch.randn(T * B, H, W, C_in, dtype=dtype, device=device)
        w_2d = torch.randn(C_in, C_out, dtype=dtype, device=device)
        state = torch.zeros(M, C_out, dtype=torch.float32, device=device)
        bn_s = torch.ones(C_out, dtype=torch.float32, device=device)
        bn_b = torch.zeros(C_out, dtype=torch.float32, device=device)

        def _run_fused():
            state.zero_()
            kern(data_nhwc, w_2d, state, bn_s, bn_b)

        us_fused = profile(_run_fused)

    # ── Individual kernel costs (for reference) ──
    us_conv = profile(lambda: (torch.mm(data_per_t[0], weight, out=conv_buf),
                               conv_buf.mul_(bn_scale).add_(bn_bias)))
    us_if = profile(lambda: ext.if_neuron(conv_buf, spikes_seq[0], membrane_seq, threshold))

    return us_seq, us_pip, us_fused, us_conv, us_if


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    sm_count = torch.cuda.get_device_properties(args.gpu).multi_processor_count
    print(f"GPU: {gpu_name} ({sm_count} SMs)")
    print(f"B={args.B}, T={args.T}")
    print()

    print("Building fused IF CUDA kernel...")
    _build_if_kernel()
    print()

    shapes = [
        (64,   64,  32, 32),
        (128, 128,  16, 16),
        (256, 256,   8,  8),
        (512, 512,   4,  4),
        (384, 384,  14, 14),
        (384, 1536, 14, 14),
    ]

    print(f"{'Shape':<28} {'M':>6} {'1×Conv':>8} {'1×IF':>8} "
          f"{'Sequen':>10} {'Pipeln':>10} {'Fused':>10} "
          f"{'Pip/Seq':>8} {'Fus/Seq':>8}")
    print("-" * 115)

    for C_in, C_out, H, W in shapes:
        M = args.B * H * W
        label = f"C={C_in}→{C_out} {H}x{W}"

        try:
            us_seq, us_pip, us_fused, us_conv, us_if = bench_shape(
                C_in, C_out, H, W, args.B, args.T)
        except Exception as e:
            print(f"{label:<28} ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        fused_str = f"{us_fused:.1f}" if us_fused else "N/A"
        fused_ratio = f"{us_fused / us_seq:.3f}" if us_fused else "N/A"

        print(f"{label:<28} {M:>6} {us_conv:>6.1f}us {us_if:>6.1f}us "
              f"{us_seq:>8.1f}us {us_pip:>8.1f}us {fused_str:>8}us "
              f"{us_pip/us_seq:>8.3f} {fused_ratio:>8}")

    print()
    print("Pip/Seq < 1.0 = stream pipelining helps (conv_{t+1} ∥ if_t)")
    print("Fus/Seq < 1.0 = register-level fusion helps (no DRAM round-trip)")
    print()
    print("Theoretical pipeline:  t_conv + (T-1)·max(t_conv, t_if) + t_if")
    print("Theoretical sequential: T · (t_conv + t_if)")


if __name__ == '__main__':
    main()
