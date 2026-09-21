"""Profile interleaved vs decomposed Conv+IF kernels with ncu.

Compares:
  - Decomposed: cuBLAS GEMM (conv+BN) + standalone IF neuron
  - Fused: TileLang interleaved Conv1x1+BN+IF kernel

Key metrics: SM throughput, DRAM throughput, total DRAM bytes, duration.

Usage:
    # Self-profile (no sudo)
    python experiments/motivation/profile_interleaved.py

    # ncu profile (requires sudo)
    sudo /usr/local/cuda-12.8/bin/ncu \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum \
        --kernel-name regex:"main_kernel|if_neuron_kernel|gemm|sgemm|hgemm|cutlass" \
        --launch-skip 20 --launch-count 10 \
        --target-processes all \
        python experiments/motivation/profile_interleaved.py --ncu
"""

import torch
import os
import sys
import argparse

os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def compile_fused(B, C_in, H, W, F, T_steps):
    from sengine.kernels.interleaved_templates import conv1x1_bn_if
    M = B * H * W
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    bm, bn, bk = 32, 64, min(32, C_in)
    for _bm in [32, 64]:
        for _bn in [64, 32]:
            _bk = min(32, C_in)
            g = ((M + _bm - 1) // _bm) * ((F + _bn - 1) // _bn)
            if g >= max(sm // 4, 1):
                bm, bn, bk = _bm, _bn, _bk
                break
        else:
            continue
        break
    return conv1x1_bn_if(B=B, C_in=C_in, H=H, W=W, F=F, T_steps=T_steps,
                         block_M=bm, block_N=bn, block_K=bk,
                         num_stages=2, threads=128)


def build_if_ext():
    from torch.utils.cpp_extension import load
    cuda_src = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
__global__ void if_neuron_kernel(const __half* __restrict__ input,
    __half* __restrict__ output, float* __restrict__ membrane,
    int N, int spatial, float threshold) {
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
    int N = input.numel(); int spatial = membrane.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if_neuron_kernel<<<(N+255)/256, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()),
        membrane.data_ptr<float>(), N, spatial, threshold);
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("if_neuron", &if_neuron_cuda, "Fused IF neuron");
}
"""
    d = os.path.join(os.path.dirname(__file__), '..', '.cache', 'bench_if_ext')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'if_neuron_bench.cu')
    with open(p, 'w') as f:
        f.write(cuda_src)
    props = torch.cuda.get_device_properties(0)
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f'{props.major}.{props.minor}')
    return load(name='if_neuron_bench', sources=[p],
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                build_directory=d, verbose=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncu', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    dtype = torch.float16

    ext = build_if_ext()

    # Shapes: (C_in, C_out, H, W, B, T)
    shapes = [
        (128, 128,  16, 16,  4, 4),   # ResNet mid, grid=64
        (384, 384,  14, 14,  4, 4),   # SpikFormer, grid=150
        (384, 1536, 14, 14,  4, 4),   # SpikFormer FFN, grid=600
        (128, 128,  16, 16, 16, 4),   # Large batch, grid=256
        (384, 384,  14, 14, 16, 4),   # Large batch SpikFormer, grid=600
    ]

    for C_in, C_out, H, W, B, T in shapes:
        M = B * H * W
        TB = T * B
        label = f"C={C_in}→{C_out} {H}x{W} B={B} T={T}"

        print(f"\n{'='*60}")
        print(f"  {label}  (M_per_t={M}, TB={TB})")
        print(f"{'='*60}")

        # Compile fused
        print(f"Compiling fused kernel...", flush=True)
        kern = compile_fused(B, C_in, H, W, C_out, T)

        # Tensors
        data_nhwc = torch.randn(TB, H, W, C_in, dtype=dtype, device='cuda')
        w_2d = torch.randn(C_in, C_out, dtype=dtype, device='cuda')
        state = torch.zeros(M, C_out, dtype=torch.float32, device='cuda')
        bn_s = torch.ones(C_out, dtype=torch.float32, device='cuda')
        bn_b = torch.zeros(C_out, dtype=torch.float32, device='cuda')

        # Decomposed tensors
        data_2d = [torch.randn(M, C_in, dtype=dtype, device='cuda') for _ in range(T)]
        w_mm = torch.randn(C_in, C_out, dtype=dtype, device='cuda')
        bn_s_2d = torch.ones(1, C_out, dtype=dtype, device='cuda')
        bn_b_2d = torch.zeros(1, C_out, dtype=dtype, device='cuda')
        conv_out = torch.empty(M, C_out, dtype=dtype, device='cuda')
        membrane = torch.zeros(M, C_out, dtype=torch.float32, device='cuda')
        spikes = [torch.empty(M, C_out, dtype=dtype, device='cuda') for _ in range(T)]

        # Warmup both paths
        for _ in range(10):
            state.zero_()
            kern(data_nhwc, w_2d, state, bn_s, bn_b)
        for _ in range(10):
            membrane.zero_()
            for t in range(T):
                torch.mm(data_2d[t], w_mm, out=conv_out)
                conv_out.mul_(bn_s_2d).add_(bn_b_2d)
                ext.if_neuron(conv_out, spikes[t], membrane, 1.0)
        torch.cuda.synchronize()

        if args.ncu:
            # ── Section A: decomposed (T conv+BN launches + T IF launches) ──
            print(f"--- DECOMPOSED (T×conv+BN + T×IF) ---", flush=True)
            membrane.zero_()
            for t in range(T):
                torch.mm(data_2d[t], w_mm, out=conv_out)
                conv_out.mul_(bn_s_2d).add_(bn_b_2d)
                ext.if_neuron(conv_out, spikes[t], membrane, 1.0)
            torch.cuda.synchronize()

            # ── Section B: fused interleaved (1 launch) ──
            print(f"--- FUSED INTERLEAVED (1 launch) ---", flush=True)
            state.zero_()
            kern(data_nhwc, w_2d, state, bn_s, bn_b)
            torch.cuda.synchronize()
        else:
            # Self-profile
            n = 500

            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(n):
                state.zero_()
                kern(data_nhwc, w_2d, state, bn_s, bn_b)
            e.record()
            torch.cuda.synchronize()
            us_fused = s.elapsed_time(e) / n * 1000

            s.record()
            for _ in range(n):
                membrane.zero_()
                for t in range(T):
                    torch.mm(data_2d[t], w_mm, out=conv_out)
                    conv_out.mul_(bn_s_2d).add_(bn_b_2d)
                    ext.if_neuron(conv_out, spikes[t], membrane, 1.0)
            e.record()
            torch.cuda.synchronize()
            us_decomp = s.elapsed_time(e) / n * 1000

            print(f"  Fused:      {us_fused:.1f} us")
            print(f"  Decomposed: {us_decomp:.1f} us")
            print(f"  Speedup:    {us_decomp/us_fused:.2f}x")


if __name__ == '__main__':
    main()
