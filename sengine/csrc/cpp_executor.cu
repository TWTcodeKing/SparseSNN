/**
 * Standalone C++ CUDA Executor for sengine.
 *
 * ZERO dependency on PyTorch or TVM. Only needs:
 *   - CUDA runtime (kernels, streams, graphs)
 *   - dlopen/dlsym (load TileLang standalone .so kernels)
 *
 * TileLang kernels are exported as standalone .so with call() wrapper.
 * IF/LIF/Add/Pool/Gemm/Mean kernels are native CUDA.
 * CUDA Graph for zero-overhead replay.
 *
 * Exposed to Python via ctypes (no pybind11, no torch dependency).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <vector>
#include <string>

// ─── Native CUDA Kernels ───

__global__ void if_neuron_kernel(
    const half* __restrict__ input, float* __restrict__ membrane,
    half* __restrict__ spikes, int total_elems, int spatial_elems, float v_threshold
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float v = membrane[s];
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        float h = v + __half2float(input[g]);
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g] = __float2half(spike);
    }
    membrane[s] = v;
}

__global__ void lif_neuron_kernel(
    const half* __restrict__ input, float* __restrict__ membrane,
    half* __restrict__ spikes, int total_elems, int spatial_elems,
    float v_threshold, float recip_tau
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float decay = 1.0f - recip_tau;
    float v = membrane[s];
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        float h = decay * v + recip_tau * __half2float(input[g]);
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g] = __float2half(spike);
    }
    membrane[s] = v;
}

__global__ void add_fp16_kernel(const half* a, const half* b, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void zero_fp32_kernel(float* ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}

// MaxPool2d NHWC: (N, H, W, C) → (N, OH, OW, C)
__global__ void maxpool2d_nhwc_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int N, int H, int W, int C, int OH, int OW,
    int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW * C;
    if (idx >= total) return;

    int c = idx % C;
    int rem = idx / C;
    int ow = rem % OW;
    rem = rem / OW;
    int oh = rem % OH;
    int n = rem / OH;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_size; kw++) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;
            float v = __half2float(input[((n * H + ih) * W + iw) * C + c]);
            if (v > max_val) max_val = v;
        }
    }
    output[idx] = __float2half(max_val);
}

// Global Average Pool NHWC: (N, H, W, C) → (N, 1, 1, C)
__global__ void global_avgpool_nhwc_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;
    int n = idx / C;

    float sum = 0.0f;
    int hw = H * W;
    for (int i = 0; i < hw; i++) {
        sum += __half2float(input[(n * hw + i) * C + c]);
    }
    output[idx] = __float2half(sum / (float)hw);
}

// Temporal Mean: (TB, ...) → reshape (T, B, ...) → mean over T → (B, ...)
__global__ void temporal_mean_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int T, int spatial_elems  // spatial_elems = B * rest
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        sum += __half2float(input[t * spatial_elems + s]);
    }
    output[s] = __float2half(sum / (float)T);
}

// FP16 GEMM: output = input @ weight^T  (row-major)
// input: (M, K), weight: (N, K), output: (M, N)
__global__ void gemm_fp16_kernel(
    const half* __restrict__ input, const half* __restrict__ weight,
    half* __restrict__ output, int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(input[row * K + k]) * __half2float(weight[col * K + k]);
    }
    output[row * N + col] = __float2half(sum);
}

// ─── Kernel types ───
enum KernelType {
    KT_TILELANG = 0, KT_IF = 1, KT_LIF = 2, KT_ADD = 3, KT_SKIP = 4,
    KT_MAXPOOL = 5, KT_GLOBAL_AVGPOOL = 6, KT_TEMPORAL_MEAN = 7,
    KT_GEMM = 8, KT_ALIAS = 9
};

// ─── TileLang standalone kernel (loaded via dlopen) ───
struct TileLangKernel {
    void* dl_handle;
    // int init() — one-time setup (dynamic smem)
    typedef int (*InitFn)();
    InitFn init_fn;
    // int call(ptr0, ptr1, ..., ptrN, cudaStream_t) — kernel launch
    // We store it as a generic function pointer; actual calling uses per-node info
    void* call_fn;
    int initialized;
};

// ─── Node descriptor ───
struct NodeDesc {
    KernelType type;
    int tilelang_idx;         // index into tilelang_kernels

    // TileLang call convention: array of void* arg pointers + stream
    // Stored as raw GPU pointers in arg order matching call() signature
    void** tl_args;           // array of GPU pointers (in call() arg order)
    int tl_n_args;            // number of pointer args (excl. stream)

    // For neuron kernels
    half* input_ptr;
    half* output_ptr;
    float* membrane_ptr;
    int total_elems, spatial_elems;
    float v_threshold, recip_tau;

    // For add kernel
    half* add_a_ptr;
    half* add_b_ptr;
    half* add_out_ptr;
    int add_n;

    // For maxpool
    half* pool_in;
    half* pool_out;
    int pool_N, pool_H, pool_W, pool_C, pool_OH, pool_OW;
    int pool_ks, pool_stride, pool_pad;

    // For global avgpool
    half* gavg_in;
    half* gavg_out;
    int gavg_N, gavg_H, gavg_W, gavg_C;

    // For temporal mean
    half* tmean_in;
    half* tmean_out;
    int tmean_T, tmean_spatial;

    // For GEMM: output = input @ weight^T
    half* gemm_in;
    half* gemm_w;
    half* gemm_out;
    int gemm_M, gemm_K, gemm_N;

    // For alias (zero-cost: reshape/transpose — just pointer copy)
    half** alias_src;
    half** alias_dst;
};

// ─── Executor ───
struct SEngineExecutor {
    int* schedule;
    int schedule_len;
    NodeDesc* nodes;
    int max_node_id;
    TileLangKernel* tl_kernels;
    int n_tl_kernels;
    float** membranes;
    int* membrane_sizes;
    int n_membranes;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    int captured;
    cublasHandle_t cublas;
};

// ─── C API (exposed via ctypes) ───
extern "C" {

SEngineExecutor* sengine_create() {
    auto* e = new SEngineExecutor();
    memset(e, 0, sizeof(SEngineExecutor));
    cudaStreamCreate(&e->stream);
    cublasCreate(&e->cublas);
    cublasSetStream(e->cublas, e->stream);
    cublasSetMathMode(e->cublas, CUBLAS_TENSOR_OP_MATH);
    return e;
}

void sengine_destroy(SEngineExecutor* e) {
    if (!e) return;
    if (e->exec) cudaGraphExecDestroy(e->exec);
    if (e->graph) cudaGraphDestroy(e->graph);
    if (e->cublas) cublasDestroy(e->cublas);
    cudaStreamDestroy(e->stream);
    for (int i = 0; i < e->n_tl_kernels; i++) {
        if (e->tl_kernels[i].dl_handle) dlclose(e->tl_kernels[i].dl_handle);
    }
    free(e->tl_kernels);
    free(e->schedule);
    if (e->nodes) {
        for (int i = 0; i <= e->max_node_id; i++) {
            free(e->nodes[i].tl_args);
        }
        free(e->nodes);
    }
    free(e->membranes);
    free(e->membrane_sizes);
    delete e;
}

int sengine_load_tilelang(SEngineExecutor* e, const char* so_path) {
    void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
    }
    auto init_fn = (TileLangKernel::InitFn)dlsym(handle, "init");
    void* call_fn = dlsym(handle, "call");
    if (!call_fn) {
        fprintf(stderr, "No call() in %s\n", so_path);
        dlclose(handle);
        return -1;
    }
    // Run init() to set up dynamic shared memory
    if (init_fn) {
        int ret = init_fn();
        if (ret != 0) {
            fprintf(stderr, "init() failed in %s\n", so_path);
            dlclose(handle);
            return -1;
        }
    }
    int idx = e->n_tl_kernels;
    e->n_tl_kernels++;
    e->tl_kernels = (TileLangKernel*)realloc(e->tl_kernels,
                                              e->n_tl_kernels * sizeof(TileLangKernel));
    e->tl_kernels[idx].dl_handle = handle;
    e->tl_kernels[idx].init_fn = init_fn;
    e->tl_kernels[idx].call_fn = call_fn;
    e->tl_kernels[idx].initialized = 1;
    return idx;
}

void sengine_set_schedule(SEngineExecutor* e, int* sched, int len) {
    e->schedule = (int*)malloc(len * sizeof(int));
    memcpy(e->schedule, sched, len * sizeof(int));
    e->schedule_len = len;
}

void sengine_alloc_nodes(SEngineExecutor* e, int max_id) {
    e->max_node_id = max_id;
    e->nodes = (NodeDesc*)calloc(max_id + 1, sizeof(NodeDesc));
}

// ─── TileLang Conv+BN node (5 args: data, weight, scale, bias, output) ───
void sengine_set_tilelang_node_5(SEngineExecutor* e, int nid, int tl_idx,
                                  void* arg0, void* arg1, void* arg2, void* arg3, void* arg4) {
    auto& n = e->nodes[nid];
    n.type = KT_TILELANG;
    n.tilelang_idx = tl_idx;
    n.tl_n_args = 5;
    n.tl_args = (void**)malloc(5 * sizeof(void*));
    n.tl_args[0] = arg0;  // data
    n.tl_args[1] = arg1;  // weight
    n.tl_args[2] = arg2;  // bn_scale
    n.tl_args[3] = arg3;  // bn_bias
    n.tl_args[4] = arg4;  // output
}

// ─── TileLang fused Conv+BN+IF node (6 args: data, weight, membrane, scale, bias, spikes) ───
void sengine_set_tilelang_node_6(SEngineExecutor* e, int nid, int tl_idx,
                                  void* arg0, void* arg1, void* arg2,
                                  void* arg3, void* arg4, void* arg5) {
    auto& n = e->nodes[nid];
    n.type = KT_TILELANG;
    n.tilelang_idx = tl_idx;
    n.tl_n_args = 6;
    n.tl_args = (void**)malloc(6 * sizeof(void*));
    n.tl_args[0] = arg0;
    n.tl_args[1] = arg1;
    n.tl_args[2] = arg2;
    n.tl_args[3] = arg3;
    n.tl_args[4] = arg4;
    n.tl_args[5] = arg5;
}

void sengine_set_if_node(SEngineExecutor* e, int nid,
                          half* input, half* output, float* membrane,
                          int total_elems, int spatial_elems, float v_threshold) {
    auto& n = e->nodes[nid];
    n.type = KT_IF;
    n.input_ptr = input;
    n.output_ptr = output;
    n.membrane_ptr = membrane;
    n.total_elems = total_elems;
    n.spatial_elems = spatial_elems;
    n.v_threshold = v_threshold;
}

void sengine_set_lif_node(SEngineExecutor* e, int nid,
                           half* input, half* output, float* membrane,
                           int total, int spatial, float v_thr, float recip_tau) {
    auto& n = e->nodes[nid];
    n.type = KT_LIF;
    n.input_ptr = input;
    n.output_ptr = output;
    n.membrane_ptr = membrane;
    n.total_elems = total;
    n.spatial_elems = spatial;
    n.v_threshold = v_thr;
    n.recip_tau = recip_tau;
}

void sengine_set_add_node(SEngineExecutor* e, int nid,
                           half* a, half* b, half* out, int n) {
    auto& nd = e->nodes[nid];
    nd.type = KT_ADD;
    nd.add_a_ptr = a;
    nd.add_b_ptr = b;
    nd.add_out_ptr = out;
    nd.add_n = n;
}

void sengine_set_maxpool_node(SEngineExecutor* e, int nid,
                               half* input, half* output,
                               int N, int H, int W, int C,
                               int OH, int OW, int ks, int stride, int pad) {
    auto& nd = e->nodes[nid];
    nd.type = KT_MAXPOOL;
    nd.pool_in = input;
    nd.pool_out = output;
    nd.pool_N = N; nd.pool_H = H; nd.pool_W = W; nd.pool_C = C;
    nd.pool_OH = OH; nd.pool_OW = OW;
    nd.pool_ks = ks; nd.pool_stride = stride; nd.pool_pad = pad;
}

void sengine_set_global_avgpool_node(SEngineExecutor* e, int nid,
                                      half* input, half* output,
                                      int N, int H, int W, int C) {
    auto& nd = e->nodes[nid];
    nd.type = KT_GLOBAL_AVGPOOL;
    nd.gavg_in = input;
    nd.gavg_out = output;
    nd.gavg_N = N; nd.gavg_H = H; nd.gavg_W = W; nd.gavg_C = C;
}

void sengine_set_temporal_mean_node(SEngineExecutor* e, int nid,
                                     half* input, half* output,
                                     int T, int spatial_elems) {
    auto& nd = e->nodes[nid];
    nd.type = KT_TEMPORAL_MEAN;
    nd.tmean_in = input;
    nd.tmean_out = output;
    nd.tmean_T = T;
    nd.tmean_spatial = spatial_elems;
}

void sengine_set_gemm_node(SEngineExecutor* e, int nid,
                             half* input, half* weight, half* output,
                             int M, int K, int N) {
    auto& nd = e->nodes[nid];
    nd.type = KT_GEMM;
    nd.gemm_in = input;
    nd.gemm_w = weight;
    nd.gemm_out = output;
    nd.gemm_M = M; nd.gemm_K = K; nd.gemm_N = N;
}

void sengine_set_skip_node(SEngineExecutor* e, int nid) {
    e->nodes[nid].type = KT_SKIP;
}

void sengine_set_alias_node(SEngineExecutor* e, int nid,
                              half* src, half* dst, int n_elems) {
    auto& nd = e->nodes[nid];
    nd.type = KT_ALIAS;
    nd.input_ptr = src;
    nd.output_ptr = dst;
    nd.total_elems = n_elems;
}

void sengine_add_membrane(SEngineExecutor* e, float* ptr, int size) {
    int idx = e->n_membranes++;
    e->membranes = (float**)realloc(e->membranes, e->n_membranes * sizeof(float*));
    e->membrane_sizes = (int*)realloc(e->membrane_sizes, e->n_membranes * sizeof(int));
    e->membranes[idx] = ptr;
    e->membrane_sizes[idx] = size;
}

// ─── Execute schedule (tight C loop, no Python) ───
void sengine_execute(SEngineExecutor* e) {
    cudaStream_t s = e->stream;
    for (int i = 0; i < e->schedule_len; i++) {
        int nid = e->schedule[i];
        if (nid < 0 || nid > e->max_node_id) continue;
        auto& nd = e->nodes[nid];

        switch (nd.type) {
        case KT_TILELANG: {
            auto& tl = e->tl_kernels[nd.tilelang_idx];
            // call(arg0, arg1, ..., argN, stream)
            // We use function pointer cast based on arg count
            if (nd.tl_n_args == 5) {
                typedef int (*CallFn5)(void*, void*, void*, void*, void*, cudaStream_t);
                auto fn = (CallFn5)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2],
                   nd.tl_args[3], nd.tl_args[4], s);
            } else if (nd.tl_n_args == 6) {
                typedef int (*CallFn6)(void*, void*, void*, void*, void*, void*, cudaStream_t);
                auto fn = (CallFn6)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2],
                   nd.tl_args[3], nd.tl_args[4], nd.tl_args[5], s);
            }
            break;
        }
        case KT_IF: {
            int thr = 256, blk = (nd.spatial_elems + thr - 1) / thr;
            if_neuron_kernel<<<blk, thr, 0, s>>>(
                nd.input_ptr, nd.membrane_ptr, nd.output_ptr,
                nd.total_elems, nd.spatial_elems, nd.v_threshold);
            break;
        }
        case KT_LIF: {
            int thr = 256, blk = (nd.spatial_elems + thr - 1) / thr;
            lif_neuron_kernel<<<blk, thr, 0, s>>>(
                nd.input_ptr, nd.membrane_ptr, nd.output_ptr,
                nd.total_elems, nd.spatial_elems, nd.v_threshold, nd.recip_tau);
            break;
        }
        case KT_ADD: {
            int thr = 256, blk = (nd.add_n + thr - 1) / thr;
            add_fp16_kernel<<<blk, thr, 0, s>>>(
                nd.add_a_ptr, nd.add_b_ptr, nd.add_out_ptr, nd.add_n);
            break;
        }
        case KT_MAXPOOL: {
            int total = nd.pool_N * nd.pool_OH * nd.pool_OW * nd.pool_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            maxpool2d_nhwc_kernel<<<blk, thr, 0, s>>>(
                nd.pool_in, nd.pool_out,
                nd.pool_N, nd.pool_H, nd.pool_W, nd.pool_C,
                nd.pool_OH, nd.pool_OW,
                nd.pool_ks, nd.pool_stride, nd.pool_pad);
            break;
        }
        case KT_GLOBAL_AVGPOOL: {
            int total = nd.gavg_N * nd.gavg_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            global_avgpool_nhwc_kernel<<<blk, thr, 0, s>>>(
                nd.gavg_in, nd.gavg_out,
                nd.gavg_N, nd.gavg_H, nd.gavg_W, nd.gavg_C);
            break;
        }
        case KT_TEMPORAL_MEAN: {
            int thr = 256, blk = (nd.tmean_spatial + thr - 1) / thr;
            temporal_mean_kernel<<<blk, thr, 0, s>>>(
                nd.tmean_in, nd.tmean_out, nd.tmean_T, nd.tmean_spatial);
            break;
        }
        case KT_GEMM: {
            // Use cuBLAS for FP16 GEMM: C = A @ B^T
            // cuBLAS is column-major, so we compute: C^T = B @ A^T
            // which gives us row-major C = A @ B^T
            const half alpha_h = __float2half(1.0f);
            const half beta_h = __float2half(0.0f);
            cublasHgemm(e->cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                nd.gemm_N, nd.gemm_M, nd.gemm_K,
                &alpha_h,
                nd.gemm_w, nd.gemm_K,  // B^T: (N, K) stored row-major
                nd.gemm_in, nd.gemm_K, // A: (M, K) stored row-major
                &beta_h,
                nd.gemm_out, nd.gemm_N);
            break;
        }
        case KT_ALIAS: {
            // Just memcpy (for reshape/transpose that need contiguous copy)
            if (nd.input_ptr != nd.output_ptr && nd.total_elems > 0) {
                cudaMemcpyAsync(nd.output_ptr, nd.input_ptr,
                    nd.total_elems * sizeof(half), cudaMemcpyDeviceToDevice, s);
            }
            break;
        }
        case KT_SKIP:
            break;
        }
    }
}

void sengine_reset_membranes(SEngineExecutor* e) {
    for (int i = 0; i < e->n_membranes; i++) {
        int thr = 256, blk = (e->membrane_sizes[i] + thr - 1) / thr;
        zero_fp32_kernel<<<blk, thr, 0, e->stream>>>(
            e->membranes[i], e->membrane_sizes[i]);
    }
}

// Debug: execute with error checking per node
void sengine_execute_checked(SEngineExecutor* e) {
    sengine_execute(e);
    cudaError_t err = cudaStreamSynchronize(e->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after execute: %s\n", cudaGetErrorString(err));
        // Find which node failed by running one at a time
        for (int i = 0; i < e->schedule_len; i++) {
            int nid = e->schedule[i];
            if (nid < 0 || nid > e->max_node_id) continue;
            // Save and restore schedule to run single node
            int orig_len = e->schedule_len;
            int orig_sched = e->schedule[0];
            e->schedule[0] = nid;
            e->schedule_len = 1;
            cudaGetLastError(); // clear
            sengine_execute(e);
            cudaError_t nerr = cudaStreamSynchronize(e->stream);
            if (nerr != cudaSuccess) {
                fprintf(stderr, "  Node %d (type %d) FAILED: %s\n",
                        nid, e->nodes[nid].type, cudaGetErrorString(nerr));
            }
            e->schedule[0] = orig_sched;
            e->schedule_len = orig_len;
        }
    }
}

void sengine_capture_graph(SEngineExecutor* e) {
    // Warm up with error checking
    cudaGetLastError(); // clear any prior errors
    sengine_execute_checked(e);
    for (int i = 1; i < 3; i++) sengine_execute(e);
    cudaStreamSynchronize(e->stream);

    cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeGlobal);
    sengine_execute(e);
    cudaStreamEndCapture(e->stream, &e->graph);
    cudaGraphInstantiate(&e->exec, e->graph, 0);
    e->captured = 1;
    // Clear any deferred CUDA errors from warmup/capture
    cudaGetLastError();
}

void sengine_replay(SEngineExecutor* e) {
    if (e->captured) {
        cudaGraphLaunch(e->exec, e->stream);
    } else {
        sengine_execute(e);
    }
}

void sengine_sync(SEngineExecutor* e) {
    cudaStreamSynchronize(e->stream);
}

float sengine_benchmark(SEngineExecutor* e, int warmup, int n_iters) {
    for (int i = 0; i < warmup; i++) sengine_replay(e);
    cudaStreamSynchronize(e->stream);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, e->stream);
    for (int i = 0; i < n_iters; i++) sengine_replay(e);
    cudaEventRecord(end, e->stream);
    cudaStreamSynchronize(e->stream);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / n_iters;
}

} // extern "C"
