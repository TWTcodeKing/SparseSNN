/**
 * sengine_cpu executor implementation.
 *
 * Self-contained C runtime that loads compiled .so kernel files via
 * dlopen and dispatches execution through a pre-computed schedule.
 * No CUDA, no Python in the hot loop.
 *
 * Exposed to Python via ctypes as libsengine_cpu.so.
 */

#include "cpu_executor.h"
#include "native_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─── Initial capacities for dynamically-growing arrays ─── */
#define INIT_TVM_CAP      16
#define INIT_MEMBRANE_CAP 32

/* ─────────────────────────────────────────────────────────────────────
 * Create / Destroy
 * ───────────────────────────────────────────────────────────────────── */

CPUExecutor* sengine_cpu_create(int n_threads)
{
    CPUExecutor* e = (CPUExecutor*)calloc(1, sizeof(CPUExecutor));
    if (!e) {
        fprintf(stderr, "[sengine_cpu] allocation failed\n");
        return NULL;
    }

    /* Thread count */
    if (n_threads <= 0) {
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
#else
        n_threads = 1;
#endif
    }
    e->n_threads = n_threads;
#ifdef _OPENMP
    omp_set_num_threads(n_threads);
#endif

    /* TVM kernel array */
    e->max_tvm_kernels = INIT_TVM_CAP;
    e->tvm_kernels = (TVMKernel*)calloc(e->max_tvm_kernels, sizeof(TVMKernel));
    e->n_tvm_kernels = 0;

    /* Membrane tracking */
    e->max_membranes = INIT_MEMBRANE_CAP;
    e->membranes = (float**)calloc(e->max_membranes, sizeof(float*));
    e->membrane_sizes = (int*)calloc(e->max_membranes, sizeof(int));
    e->n_membranes = 0;

    return e;
}

void sengine_cpu_destroy(CPUExecutor* e)
{
    if (!e) return;

    /* Close TVM .so handles */
    for (int i = 0; i < e->n_tvm_kernels; i++) {
        if (e->tvm_kernels[i].dl_handle) {
            dlclose(e->tvm_kernels[i].dl_handle);
        }
    }
    free(e->tvm_kernels);

    /* Free node descriptors */
    if (e->nodes) {
        for (int i = 0; i <= e->max_node_id; i++) {
            free(e->nodes[i].tvm_args);
            free(e->nodes[i].conv_col);
        }
        free(e->nodes);
    }

    /* Free schedule */
    free(e->schedule);

    /* Free membrane tracking arrays (NOT the buffers themselves —
     * those are owned by the Python caller via numpy) */
    free(e->membranes);
    free(e->membrane_sizes);

    free(e);
}

/* ─────────────────────────────────────────────────────────────────────
 * TVM Kernel Loading
 * ───────────────────────────────────────────────────────────────────── */

int sengine_cpu_load_tvm(CPUExecutor* e, const char* so_path)
{
    /* Grow array if needed */
    if (e->n_tvm_kernels >= e->max_tvm_kernels) {
        e->max_tvm_kernels *= 2;
        e->tvm_kernels = (TVMKernel*)realloc(
            e->tvm_kernels, e->max_tvm_kernels * sizeof(TVMKernel));
        if (!e->tvm_kernels) {
            fprintf(stderr, "[sengine_cpu] realloc tvm_kernels failed\n");
            return -1;
        }
        /* Zero new entries */
        memset(e->tvm_kernels + e->n_tvm_kernels, 0,
               (e->max_tvm_kernels - e->n_tvm_kernels) * sizeof(TVMKernel));
    }

    int idx = e->n_tvm_kernels;
    TVMKernel* k = &e->tvm_kernels[idx];

    k->dl_handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!k->dl_handle) {
        fprintf(stderr, "[sengine_cpu] dlopen failed: %s\n", dlerror());
        return -1;
    }

    /* Look up init() and call() symbols (TVM convention) */
    k->init_fn = (int (*)(void))dlsym(k->dl_handle, "init");
    k->call_fn = dlsym(k->dl_handle, "call");

    if (!k->call_fn) {
        fprintf(stderr, "[sengine_cpu] no call() symbol in %s\n", so_path);
        dlclose(k->dl_handle);
        k->dl_handle = NULL;
        return -1;
    }

    /* Run init() if present */
    k->initialized = 0;
    if (k->init_fn) {
        int ret = k->init_fn();
        if (ret != 0) {
            fprintf(stderr, "[sengine_cpu] init() failed in %s (ret=%d)\n",
                    so_path, ret);
            dlclose(k->dl_handle);
            k->dl_handle = NULL;
            return -1;
        }
    }
    k->initialized = 1;

    e->n_tvm_kernels++;
    return idx;
}

/* ─────────────────────────────────────────────────────────────────────
 * Schedule & Node Allocation
 * ───────────────────────────────────────────────────────────────────── */

void sengine_cpu_set_schedule(CPUExecutor* e, const int* sched, int len)
{
    free(e->schedule);
    e->schedule = (int*)malloc(len * sizeof(int));
    memcpy(e->schedule, sched, len * sizeof(int));
    e->schedule_len = len;
}

void sengine_cpu_alloc_nodes(CPUExecutor* e, int max_id)
{
    if (e->nodes) {
        for (int i = 0; i <= e->max_node_id; i++) {
            free(e->nodes[i].tvm_args);
            free(e->nodes[i].conv_col);
        }
        free(e->nodes);
    }
    e->max_node_id = max_id;
    e->nodes = (NodeDesc*)calloc(max_id + 1, sizeof(NodeDesc));
}

/* ─────────────────────────────────────────────────────────────────────
 * Membrane Management
 * ───────────────────────────────────────────────────────────────────── */

void sengine_cpu_register_membrane(CPUExecutor* e, float* ptr, int size)
{
    if (e->n_membranes >= e->max_membranes) {
        e->max_membranes *= 2;
        e->membranes = (float**)realloc(
            e->membranes, e->max_membranes * sizeof(float*));
        e->membrane_sizes = (int*)realloc(
            e->membrane_sizes, e->max_membranes * sizeof(int));
    }
    int idx = e->n_membranes++;
    e->membranes[idx] = ptr;
    e->membrane_sizes[idx] = size;
}

void sengine_cpu_reset_membranes(CPUExecutor* e)
{
    for (int i = 0; i < e->n_membranes; i++) {
        memset(e->membranes[i], 0, e->membrane_sizes[i] * sizeof(float));
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Node Registration — one function per kernel type
 * ───────────────────────────────────────────────────────────────────── */

void sengine_cpu_set_tvm_node(CPUExecutor* e, int nid, int tvm_idx,
                               void** args, int n_args)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_TVM;
    nd->tvm_idx = tvm_idx;
    nd->tvm_n_args = n_args;

    /* Copy the args array (caller may free theirs) */
    free(nd->tvm_args);
    nd->tvm_args = (void**)malloc(n_args * sizeof(void*));
    memcpy(nd->tvm_args, args, n_args * sizeof(void*));
}

void sengine_cpu_set_if_node(CPUExecutor* e, int nid,
                              float* input, float* output, float* membrane,
                              int total, int spatial, float v_thresh)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_IF;
    nd->input_ptr = input;
    nd->output_ptr = output;
    nd->membrane_ptr = membrane;
    nd->total_elems = total;
    nd->spatial_elems = spatial;
    nd->v_threshold = v_thresh;
}

void sengine_cpu_set_lif_node(CPUExecutor* e, int nid,
                               float* input, float* output, float* membrane,
                               int total, int spatial, float v_thresh,
                               float recip_tau)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_LIF;
    nd->input_ptr = input;
    nd->output_ptr = output;
    nd->membrane_ptr = membrane;
    nd->total_elems = total;
    nd->spatial_elems = spatial;
    nd->v_threshold = v_thresh;
    nd->recip_tau = recip_tau;
}

void sengine_cpu_set_add_node(CPUExecutor* e, int nid,
                               float* a, float* b, float* out, int n)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_ADD;
    nd->add_a = a;
    nd->add_b = b;
    nd->add_out = out;
    nd->add_n = n;
}

void sengine_cpu_set_maxpool_node(CPUExecutor* e, int nid,
                                   float* input, float* output,
                                   int N, int H, int W, int C,
                                   int OH, int OW,
                                   int kh, int kw, int sh, int sw,
                                   int ph, int pw)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_MAXPOOL;
    nd->pool_in = input;
    nd->pool_out = output;
    nd->pool_N = N;
    nd->pool_H = H;
    nd->pool_W = W;
    nd->pool_C = C;
    nd->pool_OH = OH;
    nd->pool_OW = OW;
    nd->pool_kh = kh;
    nd->pool_kw = kw;
    nd->pool_sh = sh;
    nd->pool_sw = sw;
    nd->pool_ph = ph;
    nd->pool_pw = pw;
}

void sengine_cpu_set_gavg_node(CPUExecutor* e, int nid,
                                float* input, float* output,
                                int N, int H, int W, int C)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_GLOBAL_AVGPOOL;
    nd->input_ptr = input;
    nd->output_ptr = output;
    nd->gavg_N = N;
    nd->gavg_H = H;
    nd->gavg_W = W;
    nd->gavg_C = C;
}

void sengine_cpu_set_tmean_node(CPUExecutor* e, int nid,
                                 float* input, float* output,
                                 int T, int spatial)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_TEMPORAL_MEAN;
    nd->input_ptr = input;
    nd->output_ptr = output;
    nd->tmean_T = T;
    nd->tmean_spatial = spatial;
}

void sengine_cpu_set_gemm_node(CPUExecutor* e, int nid,
                                float* a, float* b, float* c,
                                int M, int K, int N)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_GEMM;
    nd->gemm_a = a;
    nd->gemm_b = b;
    nd->gemm_c = c;
    nd->gemm_M = M;
    nd->gemm_K = K;
    nd->gemm_N = N;
}

void sengine_cpu_set_gemm_bias(CPUExecutor* e, int nid, float* bias)
{
    e->nodes[nid].gemm_bias = bias;
}

void sengine_cpu_set_conv_node(CPUExecutor* e, int nid,
                               float* input, float* weight, float* scale, float* bias,
                               float* membrane, float* output,
                               int B, int H, int W, int C_in, int F, int T,
                               int KH, int KW, int pad, int stride,
                               int neuron, float v_threshold, float v_reset, float recip_tau)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_CONV;
    nd->conv_in = input; nd->conv_w = weight; nd->conv_scale = scale; nd->conv_bias = bias;
    nd->conv_mem = membrane; nd->conv_out = output;
    nd->conv_B = B; nd->conv_H = H; nd->conv_W = W; nd->conv_Cin = C_in; nd->conv_F = F;
    nd->conv_T = T; nd->conv_KH = KH; nd->conv_KW = KW; nd->conv_pad = pad; nd->conv_stride = stride;
    nd->conv_neuron = neuron; nd->conv_vth = v_threshold; nd->conv_vreset = v_reset; nd->conv_rt = recip_tau;
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    size_t M = (size_t)B * OH * OW, K = (size_t)KH * KW * C_in;
    free(nd->conv_col);
    nd->conv_col = (float*)aligned_alloc(64, ((M * K * sizeof(float) + 63) / 64) * 64);
    if (!nd->conv_col) fprintf(stderr, "[sengine_cpu] im2col alloc failed (node %d)\n", nid);
}

void sengine_cpu_set_softmax_node(CPUExecutor* e, int nid,
                                   float* input, float* output,
                                   int outer, int inner)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_SOFTMAX;
    nd->input_ptr = input;
    nd->output_ptr = output;
    nd->sm_outer = outer;
    nd->sm_inner = inner;
}

void sengine_cpu_set_skip_node(CPUExecutor* e, int nid)
{
    e->nodes[nid].type = KT_SKIP;
}

void sengine_cpu_set_alias_node(CPUExecutor* e, int nid,
                                 float* src, float* dst, int n_elems)
{
    NodeDesc* nd = &e->nodes[nid];
    nd->type = KT_ALIAS;
    nd->input_ptr = src;
    nd->output_ptr = dst;
    nd->total_elems = n_elems;
}

/* ─────────────────────────────────────────────────────────────────────
 * Execute — main dispatch loop
 *
 * Iterates through the schedule, dispatching each node by kernel type.
 * TVM kernels are called via function pointers loaded from .so files.
 * Native kernels use hand-optimized C implementations.
 * ───────────────────────────────────────────────────────────────────── */

/* TVM call function typedefs by arity (no stream arg on CPU) */
typedef int (*TVMCallFn2)(void*, void*);
typedef int (*TVMCallFn3)(void*, void*, void*);
typedef int (*TVMCallFn4)(void*, void*, void*, void*);
typedef int (*TVMCallFn5)(void*, void*, void*, void*, void*);
typedef int (*TVMCallFn6)(void*, void*, void*, void*, void*, void*);
typedef int (*TVMCallFn7)(void*, void*, void*, void*, void*, void*, void*);

void sengine_cpu_execute(CPUExecutor* e)
{
    for (int i = 0; i < e->schedule_len; i++) {
        int nid = e->schedule[i];
        if (nid < 0 || nid > e->max_node_id) continue;

        NodeDesc* nd = &e->nodes[nid];

        switch (nd->type) {

        case KT_TVM: {
            TVMKernel* k = &e->tvm_kernels[nd->tvm_idx];
            /* Dispatch by arity — CPU TVM kernels have no stream arg */
            switch (nd->tvm_n_args) {
            case 2:
                ((TVMCallFn2)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1]);
                break;
            case 3:
                ((TVMCallFn3)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1], nd->tvm_args[2]);
                break;
            case 4:
                ((TVMCallFn4)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1], nd->tvm_args[2],
                    nd->tvm_args[3]);
                break;
            case 5:
                ((TVMCallFn5)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1], nd->tvm_args[2],
                    nd->tvm_args[3], nd->tvm_args[4]);
                break;
            case 6:
                ((TVMCallFn6)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1], nd->tvm_args[2],
                    nd->tvm_args[3], nd->tvm_args[4], nd->tvm_args[5]);
                break;
            case 7:
                ((TVMCallFn7)k->call_fn)(
                    nd->tvm_args[0], nd->tvm_args[1], nd->tvm_args[2],
                    nd->tvm_args[3], nd->tvm_args[4], nd->tvm_args[5],
                    nd->tvm_args[6]);
                break;
            default:
                fprintf(stderr,
                    "[sengine_cpu] unsupported TVM arity %d at node %d\n",
                    nd->tvm_n_args, nid);
                break;
            }
            break;
        }

        case KT_IF:
            native_if_neuron(nd->input_ptr, nd->membrane_ptr, nd->output_ptr,
                             nd->total_elems, nd->spatial_elems,
                             nd->v_threshold);
            break;

        case KT_LIF:
            native_lif_neuron(nd->input_ptr, nd->membrane_ptr, nd->output_ptr,
                              nd->total_elems, nd->spatial_elems,
                              nd->v_threshold, nd->recip_tau);
            break;

        case KT_ADD:
            native_add(nd->add_a, nd->add_b, nd->add_out, nd->add_n);
            break;

        case KT_MAXPOOL:
            native_maxpool2d(nd->pool_in, nd->pool_out,
                             nd->pool_N, nd->pool_H, nd->pool_W, nd->pool_C,
                             nd->pool_OH, nd->pool_OW,
                             nd->pool_kh, nd->pool_kw,
                             nd->pool_sh, nd->pool_sw,
                             nd->pool_ph, nd->pool_pw);
            break;

        case KT_GLOBAL_AVGPOOL:
            native_global_avgpool(nd->input_ptr, nd->output_ptr,
                                  nd->gavg_N, nd->gavg_H, nd->gavg_W,
                                  nd->gavg_C);
            break;

        case KT_TEMPORAL_MEAN:
            native_temporal_mean(nd->input_ptr, nd->output_ptr,
                                 nd->tmean_T, nd->tmean_spatial);
            break;

        case KT_GEMM:
            /* C(M,N) = A(M,K) @ B(N,K)^T (+ bias) */
            sengine_sgemm(1, nd->gemm_M, nd->gemm_N, nd->gemm_K,
                          nd->gemm_a, nd->gemm_K, nd->gemm_b, nd->gemm_K,
                          nd->gemm_c, nd->gemm_N);
            if (nd->gemm_bias) {
                for (int m = 0; m < nd->gemm_M; m++)
                    for (int n = 0; n < nd->gemm_N; n++)
                        nd->gemm_c[m * nd->gemm_N + n] += nd->gemm_bias[n];
            }
            break;

        case KT_CONV:
            native_conv_bn_neuron(nd->conv_in, nd->conv_w, nd->conv_scale, nd->conv_bias,
                                  nd->conv_mem, nd->conv_out, nd->conv_col,
                                  nd->conv_B, nd->conv_H, nd->conv_W, nd->conv_Cin, nd->conv_F,
                                  nd->conv_T, nd->conv_KH, nd->conv_KW, nd->conv_pad, nd->conv_stride,
                                  nd->conv_neuron, nd->conv_vth, nd->conv_vreset, nd->conv_rt);
            break;

        case KT_SOFTMAX:
            native_softmax(nd->input_ptr, nd->output_ptr,
                           nd->sm_outer, nd->sm_inner);
            break;

        case KT_SKIP:
            /* No-op: node was absorbed into a fusion group */
            break;

        case KT_ALIAS:
            /* Zero-cost reshape/flatten: output_ptr already points to the
             * correct buffer. If src != dst, do a memcpy. */
            if (nd->input_ptr != nd->output_ptr && nd->total_elems > 0) {
                memcpy(nd->output_ptr, nd->input_ptr,
                       nd->total_elems * sizeof(float));
            }
            break;

        default:
            fprintf(stderr,
                "[sengine_cpu] unknown kernel type %d at node %d\n",
                nd->type, nid);
            break;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Benchmark
 *
 * Runs warmup iterations (un-timed), then timed iterations.
 * Membranes are reset before each iteration so neuron state is clean.
 * Returns average time in milliseconds.
 * ───────────────────────────────────────────────────────────────────── */

double sengine_cpu_benchmark(CPUExecutor* e, int warmup, int iters)
{
    /* Warmup (un-timed) */
    for (int i = 0; i < warmup; i++) {
        sengine_cpu_reset_membranes(e);
        sengine_cpu_execute(e);
    }

    /* Timed iterations */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < iters; i++) {
        sengine_cpu_reset_membranes(e);
        sengine_cpu_execute(e);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                      + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    return (iters > 0) ? (elapsed_ms / iters) : 0.0;
}
