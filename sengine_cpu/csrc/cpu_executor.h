/**
 * sengine_cpu: Self-contained C runtime for SNN inference on CPU.
 *
 * CPU port of sengine/csrc/cpp_executor.cu. No CUDA, no Python in the
 * hot loop. Uses OpenMP for thread parallelism, AVX2 for vectorized
 * element-wise ops, and dlopen for loading TVM-compiled kernel .so files.
 *
 * All data is FP32. Membrane potentials are FP32.
 * Exposed to Python via ctypes.
 */

#ifndef SENGINE_CPU_EXECUTOR_H
#define SENGINE_CPU_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Kernel type enum ─── */
typedef enum {
    KT_TVM           = 0,   /* TVM-compiled .so (fused or decomposed)    */
    KT_IF             = 1,   /* Native IF neuron                          */
    KT_LIF            = 2,   /* Native LIF neuron                         */
    KT_ADD            = 3,   /* Element-wise add                          */
    KT_MAXPOOL        = 4,   /* MaxPool2d (NHWC)                          */
    KT_GLOBAL_AVGPOOL = 5,   /* Global average pool (NHWC)                */
    KT_TEMPORAL_MEAN  = 6,   /* Average over T timesteps                  */
    KT_GEMM           = 7,   /* BLAS GEMM (or naive fallback)             */
    KT_SOFTMAX        = 8,   /* Numerically stable softmax                */
    KT_SKIP           = 9,   /* No-op (absorbed into fusion)              */
    KT_ALIAS          = 10,  /* Pointer alias (reshape/flatten, zero-cost)*/
    KT_CONV           = 11   /* Native NHWC Conv+BN(+IF/LIF), im2col+BLAS  */
} KernelType;

/* ─── TVM kernel handle (loaded via dlopen) ─── */
typedef struct {
    void* dl_handle;          /* dlopen handle                             */
    int (*init_fn)(void);     /* int init(void) — optional one-time setup  */
    void* call_fn;            /* int call(void*, void*, ...) — cast at use */
    int initialized;          /* 1 after init() has been called            */
} TVMKernel;

/* ─── Node descriptor (one per graph node) ─── */
typedef struct {
    KernelType type;

    /* TVM kernel dispatch */
    int tvm_idx;              /* index into tvm_kernels array              */
    void** tvm_args;          /* pointer array for TVM call                */
    int tvm_n_args;           /* number of pointer args                    */

    /* Neuron kernels (IF / LIF) */
    float* input_ptr;
    float* output_ptr;
    float* membrane_ptr;
    int total_elems;          /* T * spatial_elems                         */
    int spatial_elems;        /* B * C * H * W (per-timestep)              */
    float v_threshold;
    float recip_tau;          /* 1/tau for LIF decay                       */

    /* Add kernel */
    float* add_a;
    float* add_b;
    float* add_out;
    int add_n;

    /* MaxPool2d (NHWC) */
    float* pool_in;
    float* pool_out;
    int pool_N, pool_H, pool_W, pool_C;
    int pool_OH, pool_OW;
    int pool_kh, pool_kw, pool_sh, pool_sw, pool_ph, pool_pw;

    /* Global average pool (NHWC) */
    int gavg_N, gavg_H, gavg_W, gavg_C;

    /* Temporal mean */
    int tmean_T, tmean_spatial;

    /* GEMM: C = A @ B^T (row-major) */
    float* gemm_a;
    float* gemm_b;
    float* gemm_c;
    int gemm_M, gemm_K, gemm_N;

    /* Softmax */
    int sm_outer, sm_inner;
    /* GEMM bias (optional, (N,) added after the GEMM) */
    float* gemm_bias;
    /* Native conv (NHWC, fused BN + neuron) */
    float* conv_in; float* conv_w; float* conv_scale; float* conv_bias;
    float* conv_mem; float* conv_out; float* conv_col;   /* conv_col owned by executor */
    int conv_B, conv_H, conv_W, conv_Cin, conv_F, conv_T, conv_KH, conv_KW, conv_pad, conv_stride;
    int conv_neuron;          /* 0 none, 1 IF, 2 LIF */
    float conv_vth, conv_vreset, conv_rt;
} NodeDesc;

/* ─── Executor state ─── */
typedef struct {
    int* schedule;            /* node IDs in execution order               */
    int schedule_len;

    NodeDesc* nodes;          /* indexed by node ID                        */
    int max_node_id;

    TVMKernel* tvm_kernels;   /* dynamically-growing array                 */
    int n_tvm_kernels;
    int max_tvm_kernels;

    int n_threads;            /* OpenMP thread count                       */

    /* Membrane tracking for reset */
    float** membranes;
    int* membrane_sizes;      /* size in floats                            */
    int n_membranes;
    int max_membranes;
} CPUExecutor;

/* ─── Public API ─── */

/** Create executor with given thread count (0 = auto-detect). */
CPUExecutor* sengine_cpu_create(int n_threads);

/** Destroy executor and free all resources. */
void sengine_cpu_destroy(CPUExecutor* e);

/** Load a TVM-compiled .so kernel. Returns kernel index, or -1 on error. */
int sengine_cpu_load_tvm(CPUExecutor* e, const char* so_path);

/** Set the execution schedule (copied internally). */
void sengine_cpu_set_schedule(CPUExecutor* e, const int* sched, int len);

/** Allocate node descriptor array for IDs [0, max_id]. */
void sengine_cpu_alloc_nodes(CPUExecutor* e, int max_id);

/** Execute the full schedule once. */
void sengine_cpu_execute(CPUExecutor* e);

/** Benchmark: warmup + timed iters. Returns average milliseconds. */
double sengine_cpu_benchmark(CPUExecutor* e, int warmup, int iters);

/** Zero all registered membrane buffers. */
void sengine_cpu_reset_membranes(CPUExecutor* e);

/** Register a membrane buffer for reset tracking. */
void sengine_cpu_register_membrane(CPUExecutor* e, float* ptr, int size);

/* Native fused conv node: allocates the im2col scratch (M*K floats). */
void sengine_cpu_set_conv_node(CPUExecutor* e, int nid,
                               float* input, float* weight, float* scale, float* bias,
                               float* membrane, float* output,
                               int B, int H, int W, int C_in, int F, int T,
                               int KH, int KW, int pad, int stride,
                               int neuron, float v_threshold, float v_reset, float recip_tau);
/** Optional bias for a GEMM node (call after sengine_cpu_set_gemm_node). */
void sengine_cpu_set_gemm_bias(CPUExecutor* e, int nid, float* bias);
/** Set BLAS (OpenBLAS) thread count if the symbol is available. */
void sengine_blas_set_threads(int n);

/* ─── Node registration (one per kernel type) ─── */

void sengine_cpu_set_tvm_node(CPUExecutor* e, int nid, int tvm_idx,
                               void** args, int n_args);

void sengine_cpu_set_if_node(CPUExecutor* e, int nid,
                              float* input, float* output, float* membrane,
                              int total, int spatial, float v_thresh);

void sengine_cpu_set_lif_node(CPUExecutor* e, int nid,
                               float* input, float* output, float* membrane,
                               int total, int spatial, float v_thresh,
                               float recip_tau);

void sengine_cpu_set_add_node(CPUExecutor* e, int nid,
                               float* a, float* b, float* out, int n);

void sengine_cpu_set_maxpool_node(CPUExecutor* e, int nid,
                                   float* input, float* output,
                                   int N, int H, int W, int C,
                                   int OH, int OW,
                                   int kh, int kw, int sh, int sw,
                                   int ph, int pw);

void sengine_cpu_set_gavg_node(CPUExecutor* e, int nid,
                                float* input, float* output,
                                int N, int H, int W, int C);

void sengine_cpu_set_tmean_node(CPUExecutor* e, int nid,
                                 float* input, float* output,
                                 int T, int spatial);

void sengine_cpu_set_gemm_node(CPUExecutor* e, int nid,
                                float* a, float* b, float* c,
                                int M, int K, int N);

void sengine_cpu_set_softmax_node(CPUExecutor* e, int nid,
                                   float* input, float* output,
                                   int outer, int inner);

void sengine_cpu_set_skip_node(CPUExecutor* e, int nid);

void sengine_cpu_set_alias_node(CPUExecutor* e, int nid,
                                 float* src, float* dst, int n_elems);

#ifdef __cplusplus
}
#endif

#endif /* SENGINE_CPU_EXECUTOR_H */
