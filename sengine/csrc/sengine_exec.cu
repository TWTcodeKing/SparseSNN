/**
 * sengine_exec — Standalone C++ inference benchmark binary.
 *
 * Zero dependency on Python/PyTorch. Loads a .sengine file + TileLang .so
 * kernels, captures a CUDA Graph, and benchmarks inference.
 *
 * Usage: ./sengine_exec --engine model.sengine [--input data.bin]
 *                       [--warmup 200] [--iter 1000]
 *
 * Architecture: Unified function pointer dispatch.
 *   typedef void (*KernelFn)(void** args, int n_args, cudaStream_t stream);
 *   No enum, no switch/case. Each kernel (native or TileLang .so) is a
 *   function pointer with a uniform interface.
 */

#include "kernels.cuh"
#include <cublas_v2.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ═══════════════════════════════════════════════════════════════════
// Section 1: Minimal JSON Parser
// ═══════════════════════════════════════════════════════════════════

enum JType { J_NULL, J_BOOL, J_INT, J_FLOAT, J_STRING, J_ARRAY, J_OBJECT };

struct JVal {
    JType type;
    int64_t ival;
    double fval;
    char* str;
    JVal* items; int n_items;     // array
    char** keys; JVal* vals; int n_keys; // object
};

static JVal* jval_new(JType t) {
    JVal* v = (JVal*)calloc(1, sizeof(JVal));
    v->type = t;
    return v;
}

static void skip_ws(const char*& p) { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; }

static char* parse_string_raw(const char*& p) {
    if (*p != '"') return nullptr;
    p++;
    const char* start = p;
    // Find end, handling escapes
    std::string buf;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (*p == '"') buf += '"';
            else if (*p == '\\') buf += '\\';
            else if (*p == '/') buf += '/';
            else if (*p == 'n') buf += '\n';
            else if (*p == 't') buf += '\t';
            else buf += *p;
            p++;
        } else {
            buf += *p++;
        }
    }
    if (*p == '"') p++;
    char* s = (char*)malloc(buf.size() + 1);
    memcpy(s, buf.c_str(), buf.size() + 1);
    return s;
}

static JVal* jparse(const char*& p);

static JVal* parse_array(const char*& p) {
    p++; // skip '['
    skip_ws(p);
    std::vector<JVal*> items;
    while (*p && *p != ']') {
        items.push_back(jparse(p));
        skip_ws(p);
        if (*p == ',') p++;
        skip_ws(p);
    }
    if (*p == ']') p++;
    JVal* v = jval_new(J_ARRAY);
    v->n_items = (int)items.size();
    v->items = (JVal*)malloc(sizeof(JVal) * v->n_items);
    for (int i = 0; i < v->n_items; i++) {
        v->items[i] = *items[i];
        free(items[i]); // shallow free, data moved
    }
    return v;
}

static JVal* parse_object(const char*& p) {
    p++; // skip '{'
    skip_ws(p);
    std::vector<char*> keys;
    std::vector<JVal*> vals;
    while (*p && *p != '}') {
        char* key = parse_string_raw(p);
        skip_ws(p);
        if (*p == ':') p++;
        skip_ws(p);
        JVal* val = jparse(p);
        keys.push_back(key);
        vals.push_back(val);
        skip_ws(p);
        if (*p == ',') p++;
        skip_ws(p);
    }
    if (*p == '}') p++;
    JVal* v = jval_new(J_OBJECT);
    v->n_keys = (int)keys.size();
    v->keys = (char**)malloc(sizeof(char*) * v->n_keys);
    v->vals = (JVal*)malloc(sizeof(JVal) * v->n_keys);
    for (int i = 0; i < v->n_keys; i++) {
        v->keys[i] = keys[i];
        v->vals[i] = *vals[i];
        free(vals[i]);
    }
    return v;
}

static JVal* parse_number(const char*& p) {
    const char* start = p;
    bool is_float = false;
    if (*p == '-') p++;
    while (*p >= '0' && *p <= '9') p++;
    if (*p == '.') { is_float = true; p++; while (*p >= '0' && *p <= '9') p++; }
    if (*p == 'e' || *p == 'E') { is_float = true; p++; if (*p == '+' || *p == '-') p++; while (*p >= '0' && *p <= '9') p++; }
    JVal* v;
    if (is_float) {
        v = jval_new(J_FLOAT);
        v->fval = strtod(start, nullptr);
    } else {
        v = jval_new(J_INT);
        v->ival = strtoll(start, nullptr, 10);
    }
    return v;
}

static JVal* jparse(const char*& p) {
    skip_ws(p);
    if (*p == '"') {
        JVal* v = jval_new(J_STRING);
        v->str = parse_string_raw(p);
        return v;
    }
    if (*p == '{') return parse_object(p);
    if (*p == '[') return parse_array(p);
    if (*p == 't') { p += 4; JVal* v = jval_new(J_BOOL); v->ival = 1; return v; }
    if (*p == 'f') { p += 5; JVal* v = jval_new(J_BOOL); v->ival = 0; return v; }
    if (*p == 'n') { p += 4; return jval_new(J_NULL); }
    // Handle Infinity/NaN (Python json.dumps with allow_nan=True)
    if (*p == 'I' && strncmp(p, "Infinity", 8) == 0) {
        p += 8; JVal* v = jval_new(J_FLOAT); v->fval = 1e308; return v;
    }
    if (*p == '-' && strncmp(p, "-Infinity", 9) == 0) {
        p += 9; JVal* v = jval_new(J_FLOAT); v->fval = -1e308; return v;
    }
    if (*p == 'N' && strncmp(p, "NaN", 3) == 0) {
        p += 3; JVal* v = jval_new(J_FLOAT); v->fval = 0.0; return v;
    }
    return parse_number(p);
}

JVal* json_parse(const char* text, int len) {
    const char* p = text;
    return jparse(p);
}

void json_free(JVal* v) {
    if (!v) return;
    if (v->str) free(v->str);
    if (v->items) {
        for (int i = 0; i < v->n_items; i++) json_free(&v->items[i]);
        free(v->items);
    }
    if (v->keys) {
        for (int i = 0; i < v->n_keys; i++) { free(v->keys[i]); json_free(&v->vals[i]); }
        free(v->keys); free(v->vals);
    }
}

JVal* json_get(JVal* obj, const char* key) {
    if (!obj || obj->type != J_OBJECT) return nullptr;
    for (int i = 0; i < obj->n_keys; i++) {
        if (strcmp(obj->keys[i], key) == 0) return &obj->vals[i];
    }
    return nullptr;
}

JVal* json_idx(JVal* arr, int i) {
    if (!arr || arr->type != J_ARRAY || i < 0 || i >= arr->n_items) return nullptr;
    return &arr->items[i];
}

int64_t json_int(JVal* v) {
    if (!v) return 0;
    if (v->type == J_INT) return v->ival;
    if (v->type == J_FLOAT) return (int64_t)v->fval;
    return 0;
}

double json_float(JVal* v) {
    if (!v) return 0.0;
    if (v->type == J_FLOAT) return v->fval;
    if (v->type == J_INT) return (double)v->ival;
    return 0.0;
}

const char* json_str(JVal* v) {
    if (!v || v->type != J_STRING) return "";
    return v->str ? v->str : "";
}

int json_len(JVal* v) {
    if (!v) return 0;
    if (v->type == J_ARRAY) return v->n_items;
    if (v->type == J_OBJECT) return v->n_keys;
    return 0;
}

bool json_is_null(JVal* v) { return !v || v->type == J_NULL; }

// ═══════════════════════════════════════════════════════════════════
// Section 2: Unified Dispatch Executor
// ═══════════════════════════════════════════════════════════════════

typedef void (*KernelFn)(void** args, int n_args, cudaStream_t stream);

struct NodeDesc {
    KernelFn fn;
    void** args;
    int n_args;
};

struct SEngineExec {
    NodeDesc* nodes;
    int max_nid;
    int* schedule;
    int schedule_len;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int captured;
    cublasHandle_t cublas;
    std::vector<void*> dl_handles;
    std::vector<void*> call_fns;
    std::vector<float*> membranes;
    std::vector<int> membrane_sizes;
};

SEngineExec* exec_create() {
    SEngineExec* e = new SEngineExec();
    e->nodes = nullptr; e->max_nid = 0;
    e->schedule = nullptr; e->schedule_len = 0;
    e->captured = 0; e->graph = nullptr; e->graph_exec = nullptr;
    CHECK_CUDA(cudaStreamCreate(&e->stream));
    cublasCreate(&e->cublas);
    cublasSetStream(e->cublas, e->stream);
    cublasSetMathMode(e->cublas, CUBLAS_TENSOR_OP_MATH);
    return e;
}

void exec_destroy(SEngineExec* e) {
    if (!e) return;
    if (e->graph_exec) cudaGraphExecDestroy(e->graph_exec);
    if (e->graph) cudaGraphDestroy(e->graph);
    if (e->stream) cudaStreamDestroy(e->stream);
    if (e->cublas) cublasDestroy(e->cublas);
    for (auto h : e->dl_handles) if (h) dlclose(h);
    if (e->nodes) {
        for (int i = 0; i <= e->max_nid; i++) if (e->nodes[i].args) free(e->nodes[i].args);
        free(e->nodes);
    }
    if (e->schedule) free(e->schedule);
    delete e;
}

void exec_run(SEngineExec* e) {
    for (int i = 0; i < e->schedule_len; i++) {
        auto& nd = e->nodes[e->schedule[i]];
        if (nd.fn) nd.fn(nd.args, nd.n_args, e->stream);
    }
}

void exec_reset_membranes(SEngineExec* e) {
    for (size_t i = 0; i < e->membranes.size(); i++) {
        int n = e->membrane_sizes[i];
        int blocks = (n + 255) / 256;
        zero_fp32_kernel<<<blocks, 256, 0, e->stream>>>(e->membranes[i], n);
    }
}

void exec_capture_graph(SEngineExec* e) {
    // Warmup
    exec_reset_membranes(e);
    exec_run(e);
    CHECK_CUDA(cudaStreamSynchronize(e->stream));
    CHECK_CUDA(cudaGetLastError());
    // Capture (membrane reset is NOT inside the graph — matches Python path)
    exec_reset_membranes(e);
    CHECK_CUDA(cudaStreamSynchronize(e->stream));
    CHECK_CUDA(cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeGlobal));
    exec_run(e);
    CHECK_CUDA(cudaStreamEndCapture(e->stream, &e->graph));
    CHECK_CUDA(cudaGraphInstantiate(&e->graph_exec, e->graph, nullptr, nullptr, 0));
    e->captured = 1;
    CHECK_CUDA(cudaStreamSynchronize(e->stream));
}

void exec_replay(SEngineExec* e) {
    if (e->captured)
        CHECK_CUDA(cudaGraphLaunch(e->graph_exec, e->stream));
    else {
        exec_reset_membranes(e);
        exec_run(e);
    }
}

float exec_benchmark(SEngineExec* e, int warmup, int n_iter) {
    for (int i = 0; i < warmup; i++) exec_replay(e);
    CHECK_CUDA(cudaStreamSynchronize(e->stream));
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start, e->stream);
    for (int i = 0; i < n_iter; i++) exec_replay(e);
    cudaEventRecord(end, e->stream);
    CHECK_CUDA(cudaStreamSynchronize(e->stream));
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);
    return ms / n_iter;
}

// ═══════════════════════════════════════════════════════════════════
// Section 3: Kernel Wrappers (uniform KernelFn interface)
// ═══════════════════════════════════════════════════════════════════

// Helper: read int/float from void* (heap-allocated scalar)
static inline int    rd_int(void* p) { return *(int*)p; }
static inline float  rd_flt(void* p) { return *(float*)p; }

// Allocate scalar on heap (lifetime = plan lifetime)
static void* alloc_int(int v) { int* p = (int*)malloc(sizeof(int)); *p = v; return p; }
static void* alloc_flt(float v) { float* p = (float*)malloc(sizeof(float)); *p = v; return p; }

void kern_if_neuron(void** a, int n, cudaStream_t s) {
    int total = rd_int(a[3]), spatial = rd_int(a[4]);
    int blocks = (spatial + 255) / 256;
    if_neuron_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (float*)a[2], (half*)a[1], total, spatial, rd_flt(a[5]));
}

void kern_lif_neuron(void** a, int n, cudaStream_t s) {
    int total = rd_int(a[3]), spatial = rd_int(a[4]);
    int blocks = (spatial + 255) / 256;
    lif_neuron_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (float*)a[2], (half*)a[1], total, spatial, rd_flt(a[5]), rd_flt(a[6]));
}

void kern_add(void** a, int n, cudaStream_t s) {
    int ne = rd_int(a[3]);
    int blocks = (ne + 255) / 256;
    add_fp16_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (half*)a[1], (half*)a[2], ne);
}

void kern_maxpool(void** a, int n, cudaStream_t s) {
    int N = rd_int(a[2]), H = rd_int(a[3]), W = rd_int(a[4]), C = rd_int(a[5]);
    int OH = rd_int(a[6]), OW = rd_int(a[7]), ks = rd_int(a[8]), st = rd_int(a[9]), pad = rd_int(a[10]);
    int total = N * OH * OW * C;
    int blocks = (total + 255) / 256;
    maxpool2d_nhwc_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (half*)a[1], N, H, W, C, OH, OW, ks, st, pad);
}

void kern_global_avgpool(void** a, int n, cudaStream_t s) {
    int N = rd_int(a[2]), H = rd_int(a[3]), W = rd_int(a[4]), C = rd_int(a[5]);
    int total = N * C;
    int blocks = (total + 255) / 256;
    global_avgpool_nhwc_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (half*)a[1], N, H, W, C);
}

void kern_temporal_mean(void** a, int n, cudaStream_t s) {
    int T = rd_int(a[2]), spatial = rd_int(a[3]);
    int blocks = (spatial + 255) / 256;
    temporal_mean_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (half*)a[1], T, spatial);
}

void kern_gemm(void** a, int n, cudaStream_t s) {
    int M = rd_int(a[3]), K = rd_int(a[4]), N_dim = rd_int(a[5]);
    cublasHandle_t cb = *(cublasHandle_t*)a[6];
    cublasSetStream(cb, s);
    half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    cublasHgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                N_dim, M, K,
                &alpha_h,
                (half*)a[1], K,    // weight (N, K) → transposed
                (half*)a[0], K,    // input (M, K)
                &beta_h,
                (half*)a[2], N_dim);  // output (M, N)
}

void kern_naive_conv(void** a, int n, cudaStream_t s) {
    int N = rd_int(a[5]), H = rd_int(a[6]), W = rd_int(a[7]);
    int Cin = rd_int(a[8]), Cout = rd_int(a[9]);
    int KH = rd_int(a[10]), KW = rd_int(a[11]), stride = rd_int(a[12]), pad = rd_int(a[13]);
    int OH = rd_int(a[14]), OW = rd_int(a[15]), groups = rd_int(a[16]);
    int total = N * OH * OW * Cout;
    int blocks = (total + 255) / 256;
    naive_conv2d_bn_nhwc_kernel<<<blocks, 256, 0, s>>>(
        (half*)a[0], (half*)a[1], (float*)a[2], (float*)a[3], (half*)a[4],
        N, H, W, Cin, Cout, KH, KW, stride, pad, OH, OW, groups);
}

void kern_layout_transpose(void** a, int n, cudaStream_t s) {
    int N = rd_int(a[2]), H = rd_int(a[3]), W = rd_int(a[4]), C = rd_int(a[5]), dir = rd_int(a[6]);
    int total = N * H * W * C;
    int blocks = (total + 255) / 256;
    layout_transpose_kernel<<<blocks, 256, 0, s>>>((half*)a[0], (half*)a[1], N, H, W, C, dir);
}

void kern_alias(void** a, int n, cudaStream_t s) {
    int ne = rd_int(a[2]);
    if (a[0] != a[1] && a[0] && a[1])
        cudaMemcpyAsync(a[1], a[0], ne * sizeof(half), cudaMemcpyDeviceToDevice, s);
}

// TileLang .so dispatch — args[0] = call_fn, args[1..] = GPU buffers
typedef int (*TLCallFn3)(half*, half*, half*, cudaStream_t);
typedef int (*TLCallFn5)(half*, half*, float*, float*, half*, cudaStream_t);
typedef int (*TLCallFn6)(half*, half*, float*, float*, float*, half*, cudaStream_t);

void kern_tilelang(void** a, int n, cudaStream_t s) {
    void* fn = a[0];
    int narg = n - 1;
    if (narg == 3)
        ((TLCallFn3)fn)((half*)a[1], (half*)a[2], (half*)a[3], s);
    else if (narg == 5)
        ((TLCallFn5)fn)((half*)a[1], (half*)a[2], (float*)a[3], (float*)a[4], (half*)a[5], s);
    else if (narg == 6)
        ((TLCallFn6)fn)((half*)a[1], (half*)a[2], (float*)a[3], (float*)a[4], (float*)a[5], (half*)a[6], s);
}

// ═══════════════════════════════════════════════════════════════════
// Section 4: Kernel Registry
// ═══════════════════════════════════════════════════════════════════

static std::unordered_map<std::string, KernelFn> g_registry;

void register_kernels() {
    g_registry["if_neuron"] = kern_if_neuron;
    g_registry["lif_neuron"] = kern_lif_neuron;
    g_registry["add"] = kern_add;
    g_registry["maxpool"] = kern_maxpool;
    g_registry["global_avgpool"] = kern_global_avgpool;
    g_registry["temporal_mean"] = kern_temporal_mean;
    g_registry["gemm"] = kern_gemm;
    g_registry["naive_conv"] = kern_naive_conv;
    g_registry["layout_transpose"] = kern_layout_transpose;
    g_registry["alias"] = kern_alias;
    g_registry["tilelang_3"] = kern_tilelang;
    g_registry["tilelang_5"] = kern_tilelang;
    g_registry["tilelang_6"] = kern_tilelang;
    g_registry["fused_attn"] = nullptr; // TODO: port attention
    g_registry["skip"] = nullptr;
}

// ═══════════════════════════════════════════════════════════════════
// Section 5: Plan Loader
// ═══════════════════════════════════════════════════════════════════

struct SEnginePlan {
    int T, batch_size;
    void** gpu_bufs;
    size_t* buf_sizes;
    int n_buffers;
    int* schedule;
    int schedule_len;
    int input_buf_id, output_buf_id;
    size_t input_bytes, output_bytes;
    JVal* root;  // keep alive for string refs
    char* json_buf;
    char* engine_dir;
};

static std::string dir_of(const char* path) {
    std::string s(path);
    size_t pos = s.find_last_of('/');
    if (pos == std::string::npos) return ".";
    return s.substr(0, pos);
}

static int shape_elems(JVal* shape_arr) {
    int n = 1;
    for (int i = 0; i < json_len(shape_arr); i++) {
        int d = (int)json_int(json_idx(shape_arr, i));
        if (d <= 0) d = 1;
        n *= d;
    }
    return n;
}

SEnginePlan* load_plan(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    // Read header
    char magic[4]; fread(magic, 1, 4, f);
    if (memcmp(magic, "SENG", 4) != 0) { fprintf(stderr, "Bad magic\n"); exit(1); }
    uint32_t version, header_len;
    fread(&version, 4, 1, f);
    fread(&header_len, 4, 1, f);

    // Read JSON
    char* json_buf = (char*)malloc(header_len + 1);
    fread(json_buf, 1, header_len, f);
    json_buf[header_len] = 0;
    long weight_offset = 12 + header_len;

    JVal* root = json_parse(json_buf, header_len);
    JVal* ep = json_get(root, "exec_plan");
    if (!ep) { fprintf(stderr, "No exec_plan in .sengine\n"); exit(1); }

    SEnginePlan* plan = new SEnginePlan();
    plan->root = root;
    plan->json_buf = json_buf;
    plan->engine_dir = strdup(dir_of(path).c_str());
    plan->T = (int)json_int(json_get(root, "T"));
    plan->batch_size = (int)json_int(json_get(root, "batch_size"));

    // Parse schedule
    JVal* sched = json_get(root, "schedule");
    plan->schedule_len = json_len(sched);
    plan->schedule = (int*)malloc(sizeof(int) * plan->schedule_len);
    for (int i = 0; i < plan->schedule_len; i++)
        plan->schedule[i] = (int)json_int(json_idx(sched, i));

    // Parse buffers → cudaMalloc
    JVal* bufs = json_get(ep, "buffers");
    plan->n_buffers = json_len(bufs);
    plan->gpu_bufs = (void**)calloc(plan->n_buffers, sizeof(void*));
    plan->buf_sizes = (size_t*)calloc(plan->n_buffers, sizeof(size_t));

    for (int i = 0; i < plan->n_buffers; i++) {
        JVal* b = json_idx(bufs, i);
        int buf_id = (int)json_int(json_get(b, "buf_id"));
        JVal* shape = json_get(b, "shape");
        int n_elems = shape_elems(shape);
        const char* dtype = json_str(json_get(b, "dtype"));
        int elem_size = (strcmp(dtype, "fp32") == 0) ? 4 : 2;
        size_t bytes = (size_t)n_elems * elem_size;
        if (bytes > 0 && buf_id < plan->n_buffers) {
            CHECK_CUDA(cudaMalloc(&plan->gpu_bufs[buf_id], bytes));
            CHECK_CUDA(cudaMemset(plan->gpu_bufs[buf_id], 0, bytes));
            plan->buf_sizes[buf_id] = bytes;
        }
    }

    // Load weights from binary blob section
    JVal* wm = json_get(root, "weight_manifest");
    int n_weights = json_len(wm);
    fseek(f, weight_offset, SEEK_SET);

    // Build weight_name → weight_blob map
    struct WeightBlob { float* data; int nbytes; std::string name; };
    std::vector<WeightBlob> weight_blobs;
    for (int i = 0; i < n_weights; i++) {
        JVal* entry = json_idx(wm, i);
        const char* name = json_str(json_get(entry, "name"));
        int nbytes = (int)json_int(json_get(entry, "nbytes"));
        float* data = (float*)malloc(nbytes);
        fread(data, 1, nbytes, f);
        weight_blobs.push_back({data, nbytes, std::string(name)});
    }
    fclose(f);

    // Map weight blobs to GPU buffers via buffer manifest
    // Weight buffers have category "weight" or "weight_1x1" and a weight_key
    for (int i = 0; i < plan->n_buffers; i++) {
        JVal* b = json_idx(bufs, i);
        int buf_id = (int)json_int(json_get(b, "buf_id"));
        const char* cat = json_str(json_get(b, "category"));
        JVal* wk = json_get(b, "weight_key");
        if (json_is_null(wk)) continue;
        if (strcmp(cat, "weight") != 0 && strcmp(cat, "weight_1x1") != 0
            && strcmp(cat, "bn_scale") != 0 && strcmp(cat, "bn_bias") != 0) continue;

        int source_nid = (int)json_int(json_get(b, "source_nid"));

        // Determine weight name: BN params use weight_key directly,
        // conv weights use the IR node's weight_name
        const char* weight_name = nullptr;
        const char* wk_str = json_str(json_get(b, "weight_key"));
        if (wk_str && wk_str[0]) {
            // BN scale/bias: weight_key IS the weight name (e.g., "__bn_scale_42")
            weight_name = wk_str;
        } else {
            // Conv weight: look up the IR node's weight_name
            JVal* nodes_arr = json_get(root, "nodes");
            for (int j = 0; j < json_len(nodes_arr); j++) {
                JVal* nd = json_idx(nodes_arr, j);
                if ((int)json_int(json_get(nd, "id")) == source_nid) {
                    JVal* wn = json_get(nd, "weight_name");
                    if (wn && wn->type == J_STRING) weight_name = json_str(wn);
                    break;
                }
            }
        }
        if (!weight_name) continue;

        // Find blob by name
        bool is_bn = (strcmp(cat, "bn_scale") == 0 || strcmp(cat, "bn_bias") == 0);
        for (auto& wb : weight_blobs) {
            if (wb.name == weight_name) {
                int n_floats = wb.nbytes / 4;
                if (is_bn) {
                    // BN: keep as fp32
                    CHECK_CUDA(cudaMemcpy(plan->gpu_bufs[buf_id], wb.data,
                                           n_floats * sizeof(float), cudaMemcpyHostToDevice));
                } else {
                    // Weight: convert fp32 → fp16
                    half* h_buf = (half*)malloc(n_floats * sizeof(half));
                    for (int k = 0; k < n_floats; k++)
                        h_buf[k] = __float2half(wb.data[k]);
                    CHECK_CUDA(cudaMemcpy(plan->gpu_bufs[buf_id], h_buf,
                                           n_floats * sizeof(half), cudaMemcpyHostToDevice));
                    free(h_buf);
                }
                break;
            }
        }
    }

    // BN scale/bias are now stored as weight blobs (__bn_scale_NID, __bn_bias_NID)
    // and loaded by the weight matching loop above.

    // Free weight blobs (CPU copies)
    for (auto& wb : weight_blobs) free(wb.data);

    // Identify input/output buffer IDs
    // Input: first activation in schedule; Output: last activation
    plan->input_buf_id = -1;
    plan->output_buf_id = -1;
    JVal* ep_nodes = json_get(ep, "nodes");
    for (int i = 0; i < json_len(ep_nodes); i++) {
        JVal* nd = json_idx(ep_nodes, i);
        const char* kt = json_str(json_get(nd, "kernel_type"));
        if (strcmp(kt, "skip") == 0) continue;
        JVal* in_bufs = json_get(nd, "input_bufs");
        if (plan->input_buf_id < 0 && json_len(in_bufs) > 0) {
            int bid = (int)json_int(json_idx(in_bufs, 0));
            if (bid >= 0) {
                plan->input_buf_id = bid;
                plan->input_bytes = plan->buf_sizes[bid];
            }
        }
        int obid = (int)json_int(json_get(nd, "output_buf"));
        if (obid >= 0) {
            plan->output_buf_id = obid;
            plan->output_bytes = plan->buf_sizes[obid];
        }
    }

    printf("[sengine_exec] Loaded: %d buffers, %d schedule ops, T=%d, B=%d\n",
           plan->n_buffers, plan->schedule_len, plan->T, plan->batch_size);
    return plan;
}

// ═══════════════════════════════════════════════════════════════════
// Section 6: Plan Binder
// ═══════════════════════════════════════════════════════════════════

void bind_plan(SEngineExec* exe, SEnginePlan* plan) {
    register_kernels();

    JVal* ep = json_get(plan->root, "exec_plan");
    JVal* ep_nodes = json_get(ep, "nodes");
    JVal* ep_bufs = json_get(ep, "buffers");

    // Alloc node array
    int max_nid = 0;
    for (int i = 0; i < plan->schedule_len; i++)
        if (plan->schedule[i] > max_nid) max_nid = plan->schedule[i];
    exe->max_nid = max_nid;
    exe->nodes = (NodeDesc*)calloc(max_nid + 1, sizeof(NodeDesc));
    exe->schedule = (int*)malloc(sizeof(int) * plan->schedule_len);
    memcpy(exe->schedule, plan->schedule, sizeof(int) * plan->schedule_len);
    exe->schedule_len = plan->schedule_len;

    // Build so_key → call_fn map
    std::unordered_map<std::string, void*> so_call_map;

    auto load_so = [&](const char* so_key) -> void* {
        if (!so_key || !so_key[0]) return nullptr;
        auto it = so_call_map.find(so_key);
        if (it != so_call_map.end()) return it->second;

        // Resolve path relative to engine dir
        std::string so_path;
        if (so_key[0] == '/') so_path = so_key;
        else so_path = std::string(plan->engine_dir) + "/" + so_key;

        void* handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "[sengine_exec] dlopen failed: %s: %s\n", so_path.c_str(), dlerror());
            return nullptr;
        }
        exe->dl_handles.push_back(handle);

        typedef int (*InitFn)();
        auto init_fn = (InitFn)dlsym(handle, "init");
        if (init_fn) init_fn();

        void* call_fn = dlsym(handle, "call");
        so_call_map[so_key] = call_fn;
        exe->call_fns.push_back(call_fn);
        return call_fn;
    };

    auto B = [&](int buf_id) -> void* {
        if (buf_id < 0 || buf_id >= plan->n_buffers) return nullptr;
        return plan->gpu_bufs[buf_id];
    };

    // Bind each node
    for (int i = 0; i < json_len(ep_nodes); i++) {
        JVal* nd = json_idx(ep_nodes, i);
        int nid = (int)json_int(json_get(nd, "nid"));
        const char* kt = json_str(json_get(nd, "kernel_type"));
        if (nid < 0 || nid > max_nid) continue;

        auto reg_it = g_registry.find(kt);
        KernelFn fn = (reg_it != g_registry.end()) ? reg_it->second : nullptr;
        if (!fn) { exe->nodes[nid].fn = nullptr; continue; }

        JVal* in_bufs = json_get(nd, "input_bufs");
        int out_buf = (int)json_int(json_get(nd, "output_buf"));
        int weight_buf = (int)json_int(json_get(nd, "weight_buf"));
        int scale_buf = (int)json_int(json_get(nd, "scale_buf"));
        int bias_buf = (int)json_int(json_get(nd, "bias_buf"));
        int mem_buf = (int)json_int(json_get(nd, "membrane_buf"));
        JVal* params = json_get(nd, "params");

        std::vector<void*> args;

        if (strcmp(kt, "tilelang_3") == 0 || strcmp(kt, "tilelang_5") == 0 || strcmp(kt, "tilelang_6") == 0) {
            JVal* sk = json_get(nd, "so_key");
            void* call_fn = load_so(json_is_null(sk) ? "" : json_str(sk));
            if (!call_fn) { exe->nodes[nid].fn = nullptr; continue; }
            args.push_back(call_fn);
            if (strcmp(kt, "tilelang_3") == 0 && json_len(in_bufs) >= 2) {
                args.push_back(B((int)json_int(json_idx(in_bufs, 0))));
                args.push_back(B((int)json_int(json_idx(in_bufs, 1))));
                args.push_back(B(out_buf));
            } else if (strcmp(kt, "tilelang_5") == 0) {
                int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
                args.push_back(B(ib0));
                args.push_back(B(weight_buf));
                args.push_back(B(scale_buf));
                args.push_back(B(bias_buf));
                args.push_back(B(out_buf));
            } else { // tilelang_6
                int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
                args.push_back(B(ib0));
                args.push_back(B(weight_buf));
                args.push_back(B(mem_buf));
                args.push_back(B(scale_buf));
                args.push_back(B(bias_buf));
                args.push_back(B(out_buf));
            }
        }
        else if (strcmp(kt, "if_neuron") == 0 || strcmp(kt, "lif_neuron") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));      // input
            args.push_back(B(out_buf));  // output
            args.push_back(B(mem_buf));  // membrane
            int total = params ? (int)json_int(json_get(params, "total_elems")) : 0;
            int spatial = params ? (int)json_int(json_get(params, "spatial_elems")) : 0;
            float v_thr = params ? (float)json_float(json_get(params, "v_threshold")) : 1.0f;
            args.push_back(alloc_int(total));
            args.push_back(alloc_int(spatial));
            args.push_back(alloc_flt(v_thr));
            if (strcmp(kt, "lif_neuron") == 0) {
                float rtau = params ? (float)json_float(json_get(params, "recip_tau")) : 0.5f;
                args.push_back(alloc_flt(rtau));
            }
            // Register membrane for reset
            if (B(mem_buf)) {
                exe->membranes.push_back((float*)B(mem_buf));
                // Find membrane size from buffer manifest
                for (int j = 0; j < plan->n_buffers; j++) {
                    JVal* bj = json_idx(ep_bufs, j);
                    if ((int)json_int(json_get(bj, "buf_id")) == mem_buf) {
                        exe->membrane_sizes.push_back(shape_elems(json_get(bj, "shape")));
                        break;
                    }
                }
            }
        }
        else if (strcmp(kt, "add") == 0) {
            int a_id = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            int b_id = json_len(in_bufs) > 1 ? (int)json_int(json_idx(in_bufs, 1)) : -1;
            args.push_back(B(a_id));
            args.push_back(B(b_id));
            args.push_back(B(out_buf));
            int ne = params ? (int)json_int(json_get(params, "n_elems")) : 0;
            args.push_back(alloc_int(ne));
        }
        else if (strcmp(kt, "maxpool") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(out_buf));
            // Get input shape from buffer manifest
            JVal* ib = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                if ((int)json_int(json_get(bj, "buf_id")) == ib0) { ib = bj; break; }
            }
            JVal* is = ib ? json_get(ib, "shape") : nullptr;
            int N = is && json_len(is) >= 1 ? (int)json_int(json_idx(is, 0)) : 0;
            int H = is && json_len(is) >= 2 ? (int)json_int(json_idx(is, 1)) : 0;
            int W = is && json_len(is) >= 3 ? (int)json_int(json_idx(is, 2)) : 0;
            int C = is && json_len(is) >= 4 ? (int)json_int(json_idx(is, 3)) : 0;
            int OH = params ? (int)json_int(json_get(params, "OH")) : 0;
            int OW = params ? (int)json_int(json_get(params, "OW")) : 0;
            int ks = params ? (int)json_int(json_get(params, "kernel_size")) : 3;
            int st = params ? (int)json_int(json_get(params, "stride")) : 2;
            int pad = params ? (int)json_int(json_get(params, "padding")) : 1;
            args.push_back(alloc_int(N)); args.push_back(alloc_int(H));
            args.push_back(alloc_int(W)); args.push_back(alloc_int(C));
            args.push_back(alloc_int(OH)); args.push_back(alloc_int(OW));
            args.push_back(alloc_int(ks)); args.push_back(alloc_int(st));
            args.push_back(alloc_int(pad));
        }
        else if (strcmp(kt, "global_avgpool") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(out_buf));
            JVal* ib = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                if ((int)json_int(json_get(bj, "buf_id")) == ib0) { ib = bj; break; }
            }
            JVal* is = ib ? json_get(ib, "shape") : nullptr;
            args.push_back(alloc_int(is && json_len(is) >= 1 ? (int)json_int(json_idx(is, 0)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 2 ? (int)json_int(json_idx(is, 1)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 3 ? (int)json_int(json_idx(is, 2)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 4 ? (int)json_int(json_idx(is, 3)) : 0));
        }
        else if (strcmp(kt, "temporal_mean") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(out_buf));
            int T_val = params ? (int)json_int(json_get(params, "T")) : plan->T;
            JVal* ib = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                if ((int)json_int(json_get(bj, "buf_id")) == ib0) { ib = bj; break; }
            }
            int total = ib ? shape_elems(json_get(ib, "shape")) : 0;
            int spatial = T_val > 0 ? total / T_val : total;
            args.push_back(alloc_int(T_val));
            args.push_back(alloc_int(spatial));
        }
        else if (strcmp(kt, "gemm") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(weight_buf));
            args.push_back(B(out_buf));
            // Find shapes from buffers
            JVal* ib = nullptr; JVal* wb = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                int bid = (int)json_int(json_get(bj, "buf_id"));
                if (bid == ib0) ib = bj;
                if (bid == weight_buf) wb = bj;
            }
            JVal* is = ib ? json_get(ib, "shape") : nullptr;
            JVal* ws = wb ? json_get(wb, "shape") : nullptr;
            int M = is && json_len(is) >= 1 ? (int)json_int(json_idx(is, 0)) : 1;
            int K = is && json_len(is) >= 2 ? (int)json_int(json_idx(is, 1)) : 1;
            int N_dim = ws && json_len(ws) >= 1 ? (int)json_int(json_idx(ws, 0)) : 1;
            args.push_back(alloc_int(M));
            args.push_back(alloc_int(K));
            args.push_back(alloc_int(N_dim));
            cublasHandle_t* cb_ptr = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
            *cb_ptr = exe->cublas;
            args.push_back(cb_ptr);
        }
        else if (strcmp(kt, "layout_transpose") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(out_buf));
            JVal* ib = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                if ((int)json_int(json_get(bj, "buf_id")) == ib0) { ib = bj; break; }
            }
            JVal* is = ib ? json_get(ib, "shape") : nullptr;
            int dir = params ? (int)json_int(json_get(params, "direction")) : 0;
            args.push_back(alloc_int(is && json_len(is) >= 1 ? (int)json_int(json_idx(is, 0)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 2 ? (int)json_int(json_idx(is, 1)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 3 ? (int)json_int(json_idx(is, 2)) : 0));
            args.push_back(alloc_int(is && json_len(is) >= 4 ? (int)json_int(json_idx(is, 3)) : 0));
            args.push_back(alloc_int(dir));
        }
        else if (strcmp(kt, "alias") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));
            args.push_back(B(out_buf));
            JVal* ib = nullptr;
            for (int j = 0; j < plan->n_buffers; j++) {
                JVal* bj = json_idx(ep_bufs, j);
                if ((int)json_int(json_get(bj, "buf_id")) == ib0) { ib = bj; break; }
            }
            args.push_back(alloc_int(ib ? shape_elems(json_get(ib, "shape")) : 0));
        }
        else if (strcmp(kt, "naive_conv") == 0) {
            int ib0 = json_len(in_bufs) > 0 ? (int)json_int(json_idx(in_bufs, 0)) : -1;
            args.push_back(B(ib0));      // input
            args.push_back(B(weight_buf)); // weight
            args.push_back(B(scale_buf));  // bn_scale
            args.push_back(B(bias_buf));   // bn_bias
            args.push_back(B(out_buf));    // output
            // Find conv params from IR nodes
            JVal* nodes_arr = json_get(plan->root, "nodes");
            for (int j = 0; j < json_len(nodes_arr); j++) {
                JVal* irn = json_idx(nodes_arr, j);
                if ((int)json_int(json_get(irn, "id")) != nid) continue;
                JVal* cp = json_get(irn, "conv_params");
                JVal* os = json_get(irn, "output_shapes");
                JVal* iis = json_get(irn, "input_shapes");
                JVal* osh = os && json_len(os) > 0 ? json_idx(os, 0) : nullptr;
                JVal* ish = iis && json_len(iis) > 0 ? json_idx(iis, 0) : nullptr;
                int N_v = ish && json_len(ish) >= 1 ? (int)json_int(json_idx(ish, 0)) : 0;
                int H_v = ish && json_len(ish) >= 3 ? (int)json_int(json_idx(ish, 2)) : 0;
                int W_v = ish && json_len(ish) >= 4 ? (int)json_int(json_idx(ish, 3)) : 0;
                int Cin = cp ? (int)json_int(json_get(cp, "in_channels")) : 0;
                int Cout = cp ? (int)json_int(json_get(cp, "out_channels")) : 0;
                int KH = cp ? (int)json_int(json_get(cp, "kernel_h")) : 1;
                int KW = cp ? (int)json_int(json_get(cp, "kernel_w")) : 1;
                int stride = cp ? (int)json_int(json_get(cp, "stride_h")) : 1;
                int pad = cp ? (int)json_int(json_get(cp, "pad_h")) : 0;
                int OH_v = osh && json_len(osh) >= 3 ? (int)json_int(json_idx(osh, 2)) : 0;
                int OW_v = osh && json_len(osh) >= 4 ? (int)json_int(json_idx(osh, 3)) : 0;
                int groups = cp ? (int)json_int(json_get(cp, "groups")) : 1;
                args.push_back(alloc_int(N_v)); args.push_back(alloc_int(H_v));
                args.push_back(alloc_int(W_v)); args.push_back(alloc_int(Cin));
                args.push_back(alloc_int(Cout)); args.push_back(alloc_int(KH));
                args.push_back(alloc_int(KW)); args.push_back(alloc_int(stride));
                args.push_back(alloc_int(pad)); args.push_back(alloc_int(OH_v));
                args.push_back(alloc_int(OW_v)); args.push_back(alloc_int(groups));
                break;
            }
        }

        // Store args
        exe->nodes[nid].fn = fn;
        exe->nodes[nid].n_args = (int)args.size();
        exe->nodes[nid].args = (void**)malloc(sizeof(void*) * args.size());
        memcpy(exe->nodes[nid].args, args.data(), sizeof(void*) * args.size());
    }

    printf("[sengine_exec] Bound %d nodes, loaded %d .so kernels, %d membranes\n",
           json_len(ep_nodes), (int)exe->dl_handles.size(), (int)exe->membranes.size());
}

void free_plan(SEnginePlan* plan) {
    if (!plan) return;
    for (int i = 0; i < plan->n_buffers; i++)
        if (plan->gpu_bufs[i]) cudaFree(plan->gpu_bufs[i]);
    free(plan->gpu_bufs);
    free(plan->buf_sizes);
    free(plan->schedule);
    free(plan->engine_dir);
    json_free(plan->root);
    free(plan->root);
    free(plan->json_buf);
    delete plan;
}

// ═══════════════════════════════════════════════════════════════════
// Section 7: main()
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    const char* engine_path = nullptr;
    const char* input_path = nullptr;
    int warmup = 200, n_iter = 1000, gpu_id = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc) engine_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "--input-data") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) n_iter = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) gpu_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) gpu_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: sengine_exec --engine <path.sengine> [--input <data.bin>]\n"
                   "                    [--warmup N] [--iter N] [--device GPU_ID]\n");
            return 0;
        }
    }
    if (!engine_path) { fprintf(stderr, "Error: --engine required\n"); return 1; }

    // Set GPU device
    int n_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n_gpus));
    if (gpu_id >= n_gpus) {
        fprintf(stderr, "Error: GPU %d not available (%d GPUs found)\n", gpu_id, n_gpus);
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(gpu_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_id));
    printf("[sengine_exec] GPU %d: %s (SM %d.%d, %d SMs, %.0f MB)\n",
           gpu_id, prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1e6);

    printf("=== SEngine Standalone Executor ===\n");

    // Load plan
    SEnginePlan* plan = load_plan(engine_path);

    // Create executor
    SEngineExec* exe = exec_create();

    // Bind plan
    bind_plan(exe, plan);

    // Load input data
    if (input_path) {
        FILE* f = fopen(input_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            size_t fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            void* host = malloc(fsize);
            fread(host, 1, fsize, f);
            fclose(f);
            if (plan->input_buf_id >= 0) {
                size_t copy_size = fsize < plan->input_bytes ? fsize : plan->input_bytes;
                CHECK_CUDA(cudaMemcpy(plan->gpu_bufs[plan->input_buf_id], host,
                                       copy_size, cudaMemcpyHostToDevice));
                printf("[sengine_exec] Loaded input: %zu bytes → buf %d\n",
                       copy_size, plan->input_buf_id);
            }
            free(host);
        } else {
            fprintf(stderr, "Warning: cannot open input file %s\n", input_path);
        }
    }

    // Capture CUDA graph
    printf("[sengine_exec] Capturing CUDA graph...\n");
    exec_capture_graph(exe);
    printf("[sengine_exec] Graph captured.\n");

    // Benchmark
    float ms = exec_benchmark(exe, warmup, n_iter);

    // Report
    printf("\n=== SEngine Benchmark Report ===\n");
    printf("  Engine:      %s\n", engine_path);
    printf("  T=%d  B=%d\n", plan->T, plan->batch_size);
    printf("  Warmup:      %d\n", warmup);
    printf("  Iterations:  %d\n", n_iter);
    printf("  Latency:     %.3f ms\n", ms);
    printf("  Throughput:  %.1f inferences/sec\n", 1000.0 / ms * plan->batch_size);

    // Cleanup
    exec_destroy(exe);
    free_plan(plan);
    return 0;
}
