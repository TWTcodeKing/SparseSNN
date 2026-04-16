"""SNN Compiler v4: CUTLASS Conv+BN+IF fused + CUDA Graph + wavefront streams

Architecture:
  - CUTLASS Conv+BN+IF fused kernels (tensor cores, zero DRAM for Conv→BN→IF)
  - Multi-stream wavefront scheduling (independent layers run in parallel)
  - CUDA Graph capture (ONE cudaGraphLaunch for the entire model)

Usage:
    python -m iengine.backends.cutlass.compile --model sew_resnet34 --T 4 --batch 1 --benchmark
"""

import argparse, math, os, subprocess, sys
import torch, torch.nn as nn, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import reset_net


def analyze_model(model, T, B, img_size=224):
    model.eval()
    shapes = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            def hook(n, m):
                def fn(mod, inp, out):
                    shapes[n] = {
                        'in_shape': tuple(inp[0].shape), 'out_shape': tuple(out.shape),
                        'Ci': m.in_channels, 'Co': m.out_channels,
                        'K': m.kernel_size[0], 'S': m.stride[0],
                        'P': m.padding[0], 'G': m.groups,
                    }
                return fn
            mod.register_forward_hook(hook(name, mod))
    reset_net(model)
    with torch.no_grad():
        model(torch.randn(1, 3, img_size, img_size))

    layers = []
    for name, info in shapes.items():
        _, Ci, Hi, Wi = info['in_shape']
        _, Co, Ho, Wo = info['out_shape']
        layers.append({
            'name': name, 'Ci': info['Ci'], 'Co': info['Co'],
            'K': info['K'], 'S': info['S'], 'P': info['P'], 'G': info['G'],
            'Hi': int(Hi), 'Wi': int(Wi), 'Ho': int(Ho), 'Wo': int(Wo),
        })
    return layers


def extract_weights(model, layers):
    state = model.state_dict()
    weights = {}
    for i, layer in enumerate(layers):
        cname = layer['name']
        w_key = cname + '.weight'
        if w_key not in state:
            continue
        w = state[w_key].cpu()
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()

        bn_prefix = None
        for suffix in [cname.rsplit('.', 1)[0] + '.1',
                       cname.replace('conv1', 'bn1').replace('conv2', 'bn2')]:
            if suffix + '.weight' in state:
                bn_prefix = suffix
                break

        if bn_prefix:
            gamma = state[bn_prefix + '.weight'].cpu().numpy()
            beta = state[bn_prefix + '.bias'].cpu().numpy()
            mean = state[bn_prefix + '.running_mean'].cpu().numpy()
            var = state[bn_prefix + '.running_var'].cpu().numpy()
            scale = gamma / np.sqrt(var + 1e-5)
            bias = beta - gamma * mean / np.sqrt(var + 1e-5)
            w_np = w_nhwc.numpy()
            for co in range(len(scale)):
                w_np[co] *= scale[co]
            weights[f'w_{i}'] = w_np.astype(np.float16)
            weights[f'bias_{i}'] = bias.astype(np.float16)
        else:
            weights[f'w_{i}'] = w_nhwc.numpy().astype(np.float16)
            weights[f'bias_{i}'] = np.zeros(layer['Co'], dtype=np.float16)
    return weights


def generate_snn_rt(layers, T, B, output_dir):
    L = len(layers)
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    def emit(s=''):
        lines.append(s)

    # ---- Headers ----
    emit('// SNN Compiler v4: CUTLASS fused + CUDA Graph + wavefront streams')
    emit(f'// L={L}, T={T}, B={B}')
    emit('#include "cutlass/cutlass.h"')
    emit('#include "cutlass/conv/conv2d_problem_size.h"')
    emit('#include "cutlass/conv/device/implicit_gemm_convolution.h"')
    emit('#include "cutlass/conv/kernel/default_conv2d_fprop.h"')
    emit('#include "cutlass/epilogue/thread/linear_combination_generic.h"')
    emit('#include <cuda_runtime.h>')
    emit('#include <cuda_fp16.h>')
    emit('#include <cstdio>')
    emit('#include <algorithm>')
    emit('using namespace cutlass;')
    emit()

    # ---- IF activation ----
    emit('template<typename T>struct IFA{static const bool kIsHeavy=false;')
    emit('CUTLASS_HOST_DEVICE T operator()(T v)const{return(v>=T(1))?T(1):T(0);}};')
    emit('template<typename T,int N>struct IFA<Array<T,N>>{static const bool kIsHeavy=false;')
    emit('CUTLASS_HOST_DEVICE Array<T,N>operator()(Array<T,N>const&v)const{')
    emit('Array<T,N>r;for(int i=0;i<N;i++)r[i]=(v[i]>=T(1))?T(1):T(0);return r;}};')
    emit()

    # ---- CUTLASS kernel type ----
    emit('using Epi=epilogue::thread::LinearCombinationGeneric<IFA,half_t,8,float,float,')
    emit('    epilogue::thread::ScaleType::NoBetaScaling>;')
    emit('using CK=conv::kernel::DefaultConv2dFprop<')
    emit('    half_t,layout::TensorNHWC,half_t,layout::TensorNHWC,')
    emit('    half_t,layout::TensorNHWC,float,arch::OpClassTensorOp,arch::Sm80,')
    emit('    gemm::GemmShape<128,128,64>,gemm::GemmShape<64,64,64>,')
    emit('    gemm::GemmShape<16,8,16>,Epi,')
    emit('    gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,')
    emit('    arch::OpMultiplyAdd,conv::IteratorAlgorithm::kOptimized,')
    emit('    conv::StrideSupport::kStrided,8,8>::Kernel;')
    emit('using DC=conv::device::ImplicitGemmConvolution<CK>;')
    emit()

    # ---- Layer config ----
    emit(f'#define L {L}')
    emit(f'#define T_STEPS {T}')
    emit(f'#define B_SIZE {B}')
    emit(f'#define MAX_STREAMS {min(T, 4)}')
    emit()

    emit('struct LayerCfg { int Hi,Wi,Ci,Co,K,S,P,CiG,Ho,Wo; };')
    emit(f'static const LayerCfg cfg[L] = {{')
    for layer in layers:
        emit(f'    {{{layer["Hi"]},{layer["Wi"]},{layer["Ci"]},{layer["Co"]},'
             f'{layer["K"]},{layer["S"]},{layer["P"]},{layer["Ci"]//layer["G"]},'
             f'{layer["Ho"]},{layer["Wo"]}}},')
    emit('};')
    emit()

    # ---- Global state ----
    emit('static DC* conv_ops[L];')
    emit('static cudaStream_t streams[MAX_STREAMS];')
    emit(f'static cudaEvent_t done[L][T_STEPS]; // done[layer][timestep]')
    emit('static cudaGraph_t graph = nullptr;')
    emit('static cudaGraphExec_t graph_exec = nullptr;')
    emit('static bool graph_captured = false;')
    emit()

    # ---- Build CUTLASS params helper ----
    emit('void build_conv(int i, half* w, half* bias, half* in_ptr, half* out_ptr) {')
    emit('    auto& c = cfg[i];')
    emit('    conv::Conv2dProblemSize ps({B_SIZE,c.Hi,c.Wi,c.Ci},{c.Co,c.K,c.K,c.CiG},')
    emit('        {c.P,c.P,c.P,c.P},{c.S,c.S},{1,1},{B_SIZE,c.Ho,c.Wo,c.Co});')
    emit('    auto ref=[](half_t*p,int a,int b,int c,int d){')
    emit('        return TensorRef<half_t,layout::TensorNHWC>(p,layout::TensorNHWC::packed({a,b,c,d}));};')
    emit('    DC::Arguments args(ps, ref((half_t*)in_ptr,B_SIZE,c.Hi,c.Wi,c.Ci),')
    emit('        ref((half_t*)w,c.Co,c.K,c.K,c.CiG),')
    emit('        ref((half_t*)bias,B_SIZE,c.Ho,c.Wo,c.Co),')
    emit('        ref((half_t*)out_ptr,B_SIZE,c.Ho,c.Wo,c.Co), {1.f,1.f});')
    emit('    conv_ops[i] = new DC();')
    emit('    // Skip layers that CUTLASS can_implement fails (e.g., Ci=3 stem)')
    emit('    if (conv_ops[i]->can_implement(args) != cutlass::Status::kSuccess) {')
    emit('        printf("  Layer %d: CUTLASS not supported (Ci=%d), skipping\\n", i, c.Ci);')
    emit('        delete conv_ops[i]; conv_ops[i] = nullptr; return;')
    emit('    }')
    emit('    size_t ws = conv_ops[i]->get_workspace_size(args);')
    emit('    void* d_ws = nullptr;')
    emit('    if (ws > 0) cudaMalloc(&d_ws, ws);')
    emit('    conv_ops[i]->initialize(args, d_ws);')
    emit('}')
    emit()

    # ---- cuda_init ----
    emit('extern "C" void cuda_init() {')
    emit('    for (int i = 0; i < MAX_STREAMS; i++) cudaStreamCreate(&streams[i]);')
    emit('    for (int l = 0; l < L; l++)')
    emit('        for (int t = 0; t < T_STEPS; t++) cudaEventCreate(&done[l][t]);')
    emit('}')
    emit()

    # ---- setup_layers ----
    emit('extern "C" void setup_layers(half** weights, half** biases, half** inputs, half** outputs) {')
    emit('    for (int i = 0; i < L; i++) build_conv(i, weights[i], biases[i], inputs[i], outputs[i]);')
    emit('}')
    emit()

    # ---- kernel_entry: wavefront scheduling ----
    emit('extern "C" int kernel_entry() {')
    emit('    // TAIL diagonal wavefront with multi-stream parallelism')
    emit('    for (int w = 0; w < L + T_STEPS - 1; w++) {')
    emit('        for (int l = 0; l < L; l++) {')
    emit('            int t = w - l;')
    emit('            if (t < 0 || t >= T_STEPS) continue;')
    emit('            int sid = l % MAX_STREAMS; // stream assignment')
    emit()
    emit('            // Wait for predecessor: layer l-1 must finish timestep t')
    emit('            if (l > 0) cudaStreamWaitEvent(streams[sid], done[l-1][t], 0);')
    emit('            // Wait for temporal predecessor: same layer, previous timestep')
    emit('            if (t > 0) cudaStreamWaitEvent(streams[sid], done[l][t-1], 0);')
    emit()
    emit('            // Launch CUTLASS Conv+BN+IF on this stream')
    emit('            if (conv_ops[l]) conv_ops[l]->run(streams[sid]);')
    emit('            cudaEventRecord(done[l][t], streams[sid]);')
    emit('        }')
    emit('    }')
    emit('    // Sync all streams')
    emit('    for (int i = 0; i < MAX_STREAMS; i++) cudaStreamSynchronize(streams[i]);')
    emit('    return 0;')
    emit('}')
    emit()

    # ---- kernel_entry_graph: CUDA Graph captured version ----
    emit('extern "C" int kernel_entry_graph() {')
    emit('    if (!graph_captured) {')
    emit('        // Capture the wavefront execution into a CUDA graph')
    emit('        cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeRelaxed);')
    emit('        kernel_entry(); // runs the wavefront schedule on streams')
    emit('        cudaStreamEndCapture(streams[0], &graph);')
    emit('        cudaGraphInstantiate(&graph_exec, graph, 0);')
    emit('        graph_captured = true;')
    emit('    }')
    emit('    cudaGraphLaunch(graph_exec, streams[0]);')
    emit('    cudaStreamSynchronize(streams[0]);')
    emit('    return 0;')
    emit('}')
    emit()

    # ---- Benchmark main ----
    emit('int main() {')
    emit(f'    printf("SNN Compiler v4: CUTLASS Fused + CUDA Graph + Wavefront\\n");')
    emit(f'    printf("SEW-ResNet-34: L=%d, T=%d, B=%d\\n\\n", L, T_STEPS, B_SIZE);')
    emit('    cuda_init();')
    emit()

    # Allocate per-layer buffers
    emit(f'    half* weights[L], *biases[L], *inputs[L], *outputs[L];')
    emit(f'    for (int i = 0; i < L; i++) {{')
    emit(f'        int wt = cfg[i].Co * cfg[i].K * cfg[i].K * cfg[i].CiG;')
    emit(f'        int act_in = B_SIZE * cfg[i].Hi * cfg[i].Wi * cfg[i].Ci;')
    emit(f'        int act_out = B_SIZE * cfg[i].Ho * cfg[i].Wo * cfg[i].Co;')
    emit(f'        cudaMalloc(&weights[i], wt*2); cudaMemset(weights[i], 0, wt*2);')
    emit(f'        cudaMalloc(&biases[i], cfg[i].Co*2); cudaMemset(biases[i], 0, cfg[i].Co*2);')
    emit(f'        cudaMalloc(&inputs[i], act_in*2); cudaMemset(inputs[i], 0, act_in*2);')
    emit(f'        cudaMalloc(&outputs[i], act_out*2); cudaMemset(outputs[i], 0, act_out*2);')
    emit(f'    }}')
    emit(f'    setup_layers(weights, biases, inputs, outputs);')
    emit()

    # Set max shared memory for CUTLASS kernels
    emit(f'    size_t smem_sz = sizeof(CK::SharedStorage);')
    emit(f'    printf("Shared memory: %zu bytes\\n", smem_sz);')
    emit(f'    // Set max shared memory for CUTLASS kernel')
    emit(f'    cudaFuncSetAttribute(cutlass::Kernel<CK>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_sz);')
    emit()

    # Warmup
    emit('    printf("Warming up...\\n");')
    emit('    for (int i = 0; i < 5; i++) kernel_entry();')
    emit('    cudaDeviceSynchronize();')
    emit('    auto err = cudaGetLastError();')
    emit('    if (err) { printf("CUDA error: %s\\n", cudaGetErrorString(err)); return 1; }')
    emit()

    # Benchmark: wavefront streams (no graph)
    emit('    printf("\\nBenchmark: wavefront streams\\n");')
    emit('    { cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);')
    emit('      int NI = 50;')
    emit('      cudaEventRecord(t0);')
    emit('      for (int i=0;i<NI;i++) kernel_entry();')
    emit('      cudaEventRecord(t1); cudaDeviceSynchronize();')
    emit('      float ms; cudaEventElapsedTime(&ms, t0, t1);')
    emit(f'      printf("  Wavefront: %.2f ms (%.0f img/s)\\n", ms/NI, {B}*1000.0f*NI/ms);')
    emit('    }')
    emit()

    # Benchmark: CUDA Graph
    emit('    printf("\\nBenchmark: CUDA Graph\\n");')
    emit('    kernel_entry_graph(); // first call captures')
    emit('    { cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);')
    emit('      int NI = 50;')
    emit('      cudaEventRecord(t0);')
    emit('      for (int i=0;i<NI;i++) kernel_entry_graph();')
    emit('      cudaEventRecord(t1); cudaDeviceSynchronize();')
    emit('      float ms; cudaEventElapsedTime(&ms, t0, t1);')
    emit(f'      printf("  CUDA Graph: %.2f ms (%.0f img/s)\\n", ms/NI, {B}*1000.0f*NI/ms);')
    emit('    }')
    emit()

    emit('    return 0;')
    emit('}')

    cu_path = os.path.join(output_dir, 'snn_rt.cu')
    with open(cu_path, 'w') as f:
        f.write('\n'.join(lines))
    return cu_path


def compile_and_run(cu_path, output_dir, gpu=0):
    import subprocess
    # Find CUTLASS headers from the installed package
    try:
        import cutlass_library
        cutlass_inc = os.path.join(os.path.dirname(cutlass_library.__file__), 'source', 'include')
    except ImportError:
        # Fallback: try tilelang's bundled CUTLASS
        import pathlib, tilelang
        cutlass_inc = str(pathlib.Path(tilelang.__file__).parent / '3rdparty' / 'cutlass' / 'include')
    exe_path = os.path.join(output_dir, 'snn_rt')
    cmd = ['nvcc', '-std=c++17', '-arch=sm_89', '-O2', '--expt-relaxed-constexpr',
           f'-I{cutlass_inc}', '-o', exe_path, cu_path, '-lcudart']
    print(f'Compiling (CUTLASS templates, may take 5-10 min)...')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(f'Compile failed:\n{result.stderr[:500]}')
        return
    print(f'Compiled: {exe_path}')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    result = subprocess.run([exe_path], env=env, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:300])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sew_resnet34')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', default='/tmp/snn_v4')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

    print(f'SNN Compiler v4: {args.model} T={args.T} B={args.batch}')
    print('=' * 60)

    from tengine.utils import build_model
    model = build_model(args.model, num_classes=1000, T=args.T, in_channels=3)
    model.eval()

    print('Analyzing...')
    layers = analyze_model(model, args.T, args.batch)
    print(f'  {len(layers)} layers')

    print('Generating...')
    cu_path = generate_snn_rt(layers, args.T, args.batch, args.output)
    n = sum(1 for _ in open(cu_path))
    print(f'  {cu_path}: {n} lines')

    if args.benchmark:
        compile_and_run(cu_path, args.output, args.gpu)

if __name__ == '__main__':
    main()
