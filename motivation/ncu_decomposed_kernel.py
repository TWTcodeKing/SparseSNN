"""Decomposed Conv+BN+IF for ncu profiling (same shape as fused).
C=384→384, 14x14, B=16, T=4.
"""
import torch, os, sys, importlib.util
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, '/home/twt/SparseSNN')

# Load pre-compiled extension
spec = importlib.util.spec_from_file_location("if_neuron_bench",
    "/home/twt/SparseSNN/.cache/bench_if_ext/if_neuron_bench.so")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

B, C_in, C_out, H, W, T = 16, 384, 384, 14, 14, 4
M = B * H * W  # 3136
dtype = torch.float16

# Allocate
data = [torch.randn(M, C_in, dtype=dtype, device='cuda') for _ in range(T)]
weight = torch.randn(C_in, C_out, dtype=dtype, device='cuda')
bn_scale = torch.ones(1, C_out, dtype=dtype, device='cuda')
bn_bias = torch.zeros(1, C_out, dtype=dtype, device='cuda')
conv_out = torch.empty(M, C_out, dtype=dtype, device='cuda')
membrane = torch.zeros(M, C_out, dtype=torch.float32, device='cuda')
spikes = [torch.empty(M, C_out, dtype=dtype, device='cuda') for _ in range(T)]

# Warmup
for _ in range(20):
    membrane.zero_()
    for t in range(T):
        torch.mm(data[t], weight, out=conv_out)
        conv_out.mul_(bn_scale).add_(bn_bias)
        ext.if_neuron(conv_out, spikes[t], membrane, 1.0)
torch.cuda.synchronize()

# === PROFILED: full T=4 decomposed execution ===
membrane.zero_()
for t in range(T):
    torch.mm(data[t], weight, out=conv_out)
    conv_out.mul_(bn_scale).add_(bn_bias)
    ext.if_neuron(conv_out, spikes[t], membrane, 1.0)
torch.cuda.synchronize()
