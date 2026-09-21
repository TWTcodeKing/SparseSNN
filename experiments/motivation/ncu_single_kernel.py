"""Single fused Conv1x1+BN+IF kernel for ncu profiling.
Profiles C=384→384, 14x14, B=16, T=4 (SpikFormer-like layer).
"""
import torch, os, sys
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sengine.kernels.interleaved_templates import conv1x1_bn_if

B, C_in, C_out, H, W, T = 16, 384, 384, 14, 14, 4
M = B * H * W  # 3136
dtype = torch.float16

# Compile kernel with good tile config
sm_count = torch.cuda.get_device_properties(0).multi_processor_count
bm, bn, bk = 32, 64, 32  # grid = ceil(3136/32)*ceil(384/64) = 98*6 = 588 blocks
kern = conv1x1_bn_if(B=B, C_in=C_in, H=H, W=W, F=C_out, T_steps=T,
                     block_M=bm, block_N=bn, block_K=bk,
                     num_stages=2, threads=128)

# Allocate
data = torch.randn(T*B, H, W, C_in, dtype=dtype, device='cuda')
weight = torch.randn(C_in, C_out, dtype=dtype, device='cuda')
state = torch.zeros(M, C_out, dtype=torch.float32, device='cuda')
bn_scale = torch.ones(C_out, dtype=torch.float32, device='cuda')
bn_bias = torch.zeros(C_out, dtype=torch.float32, device='cuda')

# Warmup (outside ncu capture region)
for _ in range(20):
    state.zero_()
    kern(data, weight, state, bn_scale, bn_bias)
torch.cuda.synchronize()

# === PROFILED REGION ===
state.zero_()
kern(data, weight, state, bn_scale, bn_bias)
torch.cuda.synchronize()
