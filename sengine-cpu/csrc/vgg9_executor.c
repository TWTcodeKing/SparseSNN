/**
 * Self-contained VGG-9 SNN CPU executor.
 *
 * Zero Python in the hot loop. All buffers pre-allocated at init.
 * Dispatch: fused Conv+BN+LIF (implicit im2col) or
 *           OpenBLAS GEMM + im2col + native LIF (decomposed).
 *
 * Kernel selection per layer: FLOP-based threshold (same as optimizer.py).
 *
 * Public API:
 *   VGG9Engine* vgg9_create(int B, int T, int num_classes, int threads);
 *   void  vgg9_destroy(VGG9Engine* e);
 *   void  vgg9_set_input(VGG9Engine* e, const float* input);
 *   void  vgg9_execute(VGG9Engine* e);
 *   float vgg9_benchmark(VGG9Engine* e, int warmup, int iters);
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Link against scipy's OpenBLAS (symbols prefixed with scipy_) */
extern void scipy_cblas_sgemm(int order, int transA, int transB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
    const float* B, int ldb,
    float beta, float* C, int ldc);
extern void scipy_openblas_set_num_threads(int n);
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS 111
#define CBLAS_TRANS 112
#define HAS_BLAS 1

#include <immintrin.h>

/* ─── Architecture constants ─── */
#define NUM_CONV_LAYERS 6
#define VGG_KH 3
#define VGG_KW 3
#define VGG_PAD 1
#define VGG_CONV_STRIDE 1
#define POOL_K 2
#define POOL_S 2

#define FUSED_FLOP_LIMIT 50000000L

/* ─── Layer descriptor ─── */
typedef struct {
    int C_in, C_out;
    int H, W, OH, OW;      /* spatial dims at this layer */
    int M;                  /* B * OH * OW */
    int K;                  /* C_in * VGG_KH * VGG_KW */
    int F;                  /* C_out */
    int pool_after;         /* 1 if maxpool follows */
    int use_fused;          /* 1 = fused implicit im2col, 0 = BLAS */

    /* Weights (owned, pre-allocated at build time) */
    float* weight;          /* (K, F) row-major — original layout */
    float* packed_weight;   /* (num_panels, K, NR) GEBP panel layout, or NULL */
    float* bn_scale;        /* (F,) */
    float* bn_bias;         /* (F,) */

    /* Activation buffers (pointers into pool) */
    float* act_in;          /* input activation for this layer */
    float* act_out;         /* output activation (spikes reshaped to NCHW) */
    float* membrane;        /* (M, F) */
    float* spikes_flat;     /* (T*M, F) raw spike output */
    float* reshape_tmp;     /* (TB*F*OH*OW) temp for reshape+pool, or NULL */

    /* For decomposed path */
    float* im2col_buf;      /* (T*M, K) im2col buffer */
    float* gemm_buf;        /* (T*M, F) GEMM output */
} ConvLayer;

typedef struct {
    int B, T, num_classes;
    int n_threads;
    int TB;                 /* T * B */

    ConvLayer layers[NUM_CONV_LAYERS];

    /* Classifier */
    float* fc_weight;       /* (num_classes, last_C) */
    float* fc_output;       /* (B, num_classes) */

    /* Temporal mean + GAP output */
    float* tmean_buf;       /* (B, last_C, last_H, last_W) */
    float* gap_buf;         /* (B, last_C) */

    /* Input buffer (copied in) */
    float* input_buf;       /* (TB, 3, 32, 32) */

    /* Unified memory pool */
    float* pool;
    size_t pool_bytes;
} VGG9Engine;


/* ─── External: weight packing (from fused_conv_bn_if.c) ─── */
extern void sengine_pack_weight_panels(const float* weight, float* packed,
                                        int K, int F);

/* ─── External: fused Conv2d+BN+LIF kernel (from fused_conv_bn_if.c) ─── */
extern void fused_conv2d_bn_lif_tloop(
    const float* input, const float* weight,
    const float* bn_scale, const float* bn_bias,
    float* membrane, float* spikes,
    int B, int C_in, int H, int W, int F, int T,
    int kh, int kw, int pad, int stride_hw,
    float v_threshold, float v_reset,
    float decay, float recip_tau);

/* ─── External: im2col (from libim2col.so, or inline below) ─── */
static void im2col_nchw_inline(
    const float* __restrict__ data, float* __restrict__ col,
    int N, int C, int H, int W,
    int kH, int kW, int stride, int pad, int OH, int OW)
{
    int K = C * kH * kW;
    long total_rows = (long)N * OH * OW;

    #pragma omp parallel for schedule(static)
    for (long row = 0; row < total_rows; row++) {
        int ow = row % OW;
        int tmp = row / OW;
        int oh = tmp % OH;
        int n = tmp / OH;
        float* col_row = col + row * K;
        int k = 0;
        for (int c = 0; c < C; c++) {
            const float* dc = data + ((long)n * C + c) * H * W;
            for (int ky = 0; ky < kH; ky++) {
                int ih = oh * stride - pad + ky;
                for (int kx = 0; kx < kW; kx++) {
                    int iw = ow * stride - pad + kx;
                    col_row[k++] = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                                  ? dc[ih * W + iw] : 0.0f;
                }
            }
        }
    }
}

/* ─── Native LIF neuron (from native_kernels.c) ─── */
extern void native_lif_neuron(const float* input, float* membrane, float* spikes,
                               int total_elems, int spatial_elems,
                               float v_threshold, float recip_tau);

/* ─── MaxPool2d NCHW ─── */
static void maxpool2d_nchw(const float* __restrict__ in, float* __restrict__ out,
                            int N, int C, int H, int W) {
    int OH = H / POOL_S;
    int OW = W / POOL_S;
    long total = (long)N * C * OH * OW;

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < total; idx++) {
        int ow = idx % OW;
        long tmp = idx / OW;
        int oh = tmp % OH;
        tmp = tmp / OH;
        int c = tmp % C;
        int n = tmp / C;

        float mx = -1e30f;
        for (int ky = 0; ky < POOL_K; ky++)
            for (int kx = 0; kx < POOL_K; kx++) {
                float v = in[((long)n*C + c) * H * W + (oh*POOL_S+ky)*W + (ow*POOL_S+kx)];
                if (v > mx) mx = v;
            }
        out[((long)n*C + c) * OH * OW + oh*OW + ow] = mx;
    }
}

/* ─── Spike reshape: (T*M, F) flat → (TB, F, OH, OW) NCHW ─── */
static void reshape_spikes_to_nchw(
    const float* __restrict__ spk_flat,  /* (T*M, F) row-major */
    float* __restrict__ act_nchw,        /* (TB, F, OH, OW) */
    int T, int B, int OH, int OW, int F)
{
    int M = B * OH * OW;
    long total = (long)T * M * F;

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < total; idx++) {
        int f = idx % F;
        long tmp = idx / F;
        /* tmp = t * M + (b * OH * OW + oh * OW + ow) */
        int pos = tmp % M;
        int t = tmp / M;

        int ow = pos % OW;
        int rem = pos / OW;
        int oh = rem % OH;
        int b = rem / OH;

        /* Target: act_nchw[(t*B+b)*F*OH*OW + f*OH*OW + oh*OW + ow] */
        long dst = ((long)(t*B+b)*F + f) * (OH*OW) + oh*OW + ow;
        act_nchw[dst] = spk_flat[idx];
    }
}

/* ─── BLAS GEMM + BN + LIF (decomposed path) ─── */
static void decomposed_conv_bn_lif(
    const float* __restrict__ act_in,   /* (TB, C_in, H, W) NCHW */
    float* __restrict__ im2col_buf,     /* (TB*OH*OW, K) pre-allocated */
    float* __restrict__ gemm_buf,       /* (TB*OH*OW, F) pre-allocated */
    const float* __restrict__ weight,   /* (K, F) */
    const float* __restrict__ bn_scale, /* (F,) */
    const float* __restrict__ bn_bias,  /* (F,) */
    float* __restrict__ membrane,       /* (M*F,) flat */
    float* __restrict__ spikes,         /* (T*M*F,) flat */
    int TB, int C_in, int H, int W, int F, int T, int B)
{
    int OH = (H + 2*VGG_PAD - VGG_KH) / VGG_CONV_STRIDE + 1;
    int OW = (W + 2*VGG_PAD - VGG_KW) / VGG_CONV_STRIDE + 1;
    int K = C_in * VGG_KH * VGG_KW;
    int M = B * OH * OW;
    long TM = (long)T * M;
    long rows = (long)TB * OH * OW;  /* = TM */

    /* 1. im2col */
    im2col_nchw_inline(act_in, im2col_buf, TB, C_in, H, W,
                        VGG_KH, VGG_KW, VGG_CONV_STRIDE, VGG_PAD, OH, OW);

    /* 2. GEMM: (rows, K) @ (K, F) → (rows, F) */
#ifdef HAS_BLAS
    scipy_cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                (int)rows, F, K,
                1.0f, im2col_buf, K,
                weight, F,
                0.0f, gemm_buf, F);
#else
    /* Naive GEMM fallback */
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < rows; r++) {
        const float* a = im2col_buf + r * K;
        float* c = gemm_buf + r * F;
        for (int f = 0; f < F; f++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += a[k] * weight[k*F+f];
            c[f] = s;
        }
    }
#endif

    /* 3. BN: gemm_buf = gemm_buf * scale + bias */
    #pragma omp parallel for schedule(static)
    for (long r = 0; r < rows; r++) {
        float* row = gemm_buf + r * F;
        for (int f = 0; f < F; f++)
            row[f] = row[f] * bn_scale[f] + bn_bias[f];
    }

    /* 4. LIF neuron T-loop */
    int spatial = M * F;
    native_lif_neuron(gemm_buf, membrane, spikes,
                       (int)(TM * F), spatial, 1.0f, 0.5f);
}


/* ════════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════════ */

/* VGG-9 config: (C_in, C_out, pool_after) */
static const int VGG9_CFG[][3] = {
    {3,   64,  1},
    {64,  128, 1},
    {128, 256, 0},
    {256, 256, 1},
    {256, 512, 0},
    {512, 512, 1},
};


VGG9Engine* vgg9_create(int B, int T, int num_classes, int n_threads)
{
    VGG9Engine* e = (VGG9Engine*)calloc(1, sizeof(VGG9Engine));
    e->B = B;
    e->T = T;
    e->TB = T * B;
    e->num_classes = num_classes;
    e->n_threads = n_threads;

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#ifdef HAS_BLAS
    if (n_threads > 0) scipy_openblas_set_num_threads(n_threads);
#endif

    /* ── Compute buffer sizes ── */
    size_t total_bytes = 0;
    int H = 32, W = 32;

    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        ConvLayer* ly = &e->layers[l];
        ly->C_in  = VGG9_CFG[l][0];
        ly->C_out = VGG9_CFG[l][1];
        ly->pool_after = VGG9_CFG[l][2];
        ly->H = H; ly->W = W;
        ly->OH = (H + 2*VGG_PAD - VGG_KH) / VGG_CONV_STRIDE + 1;
        ly->OW = (W + 2*VGG_PAD - VGG_KW) / VGG_CONV_STRIDE + 1;
        ly->M = B * ly->OH * ly->OW;
        ly->K = ly->C_in * VGG_KH * VGG_KW;
        ly->F = ly->C_out;

        long flops = 2L * ly->M * ly->K * ly->F;
        /* Fused only for tiny GEMMs where our kernel is competitive with BLAS */
        ly->use_fused = (flops <= FUSED_FLOP_LIMIT) &&
                        (ly->K <= 64);  /* Only 1×1 or very small K */

        /* Weights */
        ly->weight   = (float*)calloc((size_t)ly->K * ly->F, sizeof(float));
        ly->bn_scale = (float*)malloc((size_t)ly->F * sizeof(float));
        ly->bn_bias  = (float*)calloc((size_t)ly->F, sizeof(float));
        for (int f = 0; f < ly->F; f++) ly->bn_scale[f] = 1.0f;

        /* Activation buffers: act_out = (TB, C_out, OH_after_pool, OW_after_pool) */
        int out_H = ly->pool_after ? ly->OH / POOL_S : ly->OH;
        int out_W = ly->pool_after ? ly->OW / POOL_S : ly->OW;
        total_bytes += (size_t)e->TB * ly->C_out * out_H * out_W * sizeof(float); /* act_out */
        total_bytes += (size_t)ly->M * ly->F * sizeof(float);                     /* membrane */
        total_bytes += (size_t)T * ly->M * ly->F * sizeof(float);                 /* spikes_flat */

        if (!ly->use_fused) {
            total_bytes += (size_t)e->TB * ly->OH * ly->OW * ly->K * sizeof(float); /* im2col */
            total_bytes += (size_t)e->TB * ly->OH * ly->OW * ly->F * sizeof(float); /* gemm_buf */
        }

        /* Pre-allocate reshape temp for pool layers (eliminate hot-loop malloc) */
        if (ly->pool_after) {
            total_bytes += (size_t)e->TB * ly->C_out * ly->OH * ly->OW * sizeof(float);
        }

        if (ly->pool_after) {
            H = ly->OH / POOL_S;
            W = ly->OW / POOL_S;
        } else {
            H = ly->OH;
            W = ly->OW;
        }
    }

    /* Input + tmean + gap + fc_output */
    total_bytes += (size_t)e->TB * 3 * 32 * 32 * sizeof(float);          /* input */
    total_bytes += (size_t)B * 512 * H * W * sizeof(float);              /* tmean */
    total_bytes += (size_t)B * 512 * sizeof(float);                       /* gap */
    total_bytes += (size_t)B * num_classes * sizeof(float);               /* fc_out */

    /* ── Allocate unified pool ── */
    total_bytes += 4096; /* alignment padding */
    e->pool_bytes = total_bytes;
    e->pool = (float*)aligned_alloc(64, total_bytes);
    if (!e->pool) { free(e); return NULL; }
    memset(e->pool, 0, total_bytes);

    /* ── Assign pointers from pool ── */
    float* ptr = e->pool;
#define ALLOC(n) ({ float* p = ptr; ptr += (n); p; })

    e->input_buf = ALLOC((size_t)e->TB * 3 * 32 * 32);

    H = 32; W = 32;
    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        ConvLayer* ly = &e->layers[l];
        int out_H = ly->pool_after ? ly->OH / POOL_S : ly->OH;
        int out_W = ly->pool_after ? ly->OW / POOL_S : ly->OW;

        ly->act_in = (l == 0) ? e->input_buf : e->layers[l-1].act_out;
        ly->spikes_flat = ALLOC((size_t)T * ly->M * ly->F);
        ly->act_out = ALLOC((size_t)e->TB * ly->C_out * out_H * out_W);
        ly->membrane = ALLOC((size_t)ly->M * ly->F);

        if (!ly->use_fused) {
            ly->im2col_buf = ALLOC((size_t)e->TB * ly->OH * ly->OW * ly->K);
            ly->gemm_buf = ALLOC((size_t)e->TB * ly->OH * ly->OW * ly->F);
        }

        if (ly->pool_after) {
            ly->reshape_tmp = ALLOC((size_t)e->TB * ly->C_out * ly->OH * ly->OW);
        }

        H = out_H; W = out_W;
    }

    e->tmean_buf = ALLOC((size_t)B * 512 * H * W);
    e->gap_buf   = ALLOC((size_t)B * 512);
    e->fc_output = ALLOC((size_t)B * num_classes);

    /* Classifier weight */
    e->fc_weight = (float*)calloc((size_t)num_classes * 512, sizeof(float));

#undef ALLOC

    return e;
}


void vgg9_destroy(VGG9Engine* e)
{
    if (!e) return;
    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        free(e->layers[l].weight);
        free(e->layers[l].bn_scale);
        free(e->layers[l].bn_bias);
        free(e->layers[l].packed_weight);
    }
    free(e->fc_weight);
    free(e->pool);
    free(e);
}


void vgg9_set_threads(VGG9Engine* e, int n_threads)
{
    e->n_threads = n_threads;
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#ifdef HAS_BLAS
    if (n_threads > 0) scipy_openblas_set_num_threads(n_threads);
#endif
}


void vgg9_set_input(VGG9Engine* e, const float* input)
{
    memcpy(e->input_buf, input, (size_t)e->TB * 3 * 32 * 32 * sizeof(float));
}


void vgg9_set_weights(VGG9Engine* e, int layer, const float* w,
                      const float* scale, const float* bias)
{
    ConvLayer* ly = &e->layers[layer];
    memcpy(ly->weight, w, (size_t)ly->K * ly->F * sizeof(float));
    memcpy(ly->bn_scale, scale, (size_t)ly->F * sizeof(float));
    memcpy(ly->bn_bias, bias, (size_t)ly->F * sizeof(float));

    /* Phase 4: pre-pack weight into GEBP panel layout at build time.
     * Fused layers need packed_weight for the GEBP path (K > KC_THRESH=64).
     * Decomposed layers don't use packed_weight (BLAS handles its own packing). */
    if (ly->use_fused && ly->K > 64) {
        int NR = 8;  /* AVX2 SIMD width */
        int num_panels = ly->F / NR;
        if (num_panels > 0) {
            free(ly->packed_weight);
            ly->packed_weight = (float*)aligned_alloc(64,
                (size_t)num_panels * ly->K * NR * sizeof(float));
            sengine_pack_weight_panels(ly->weight, ly->packed_weight, ly->K, ly->F);
        }
    }
}


void vgg9_set_fc_weights(VGG9Engine* e, const float* w)
{
    memcpy(e->fc_weight, w, (size_t)e->num_classes * 512 * sizeof(float));
}


void vgg9_execute(VGG9Engine* e)
{
    int T = e->T, B = e->B, TB = e->TB;

    for (int l = 0; l < NUM_CONV_LAYERS; l++) {
        ConvLayer* ly = &e->layers[l];
        int H = ly->H, W = ly->W;
        int OH = ly->OH, OW = ly->OW;
        int M = ly->M, K = ly->K, F = ly->F;

        /* Reset membrane */
        if (ly->use_fused)
            memset(ly->membrane, 0, (size_t)M * F * sizeof(float));
        else
            memset(ly->membrane, 0, (size_t)M * F * sizeof(float));

        if (ly->use_fused) {
            /* ── Fused: implicit im2col + GEBP + BN + LIF ── */
            fused_conv2d_bn_lif_tloop(
                ly->act_in, ly->weight, ly->bn_scale, ly->bn_bias,
                ly->membrane, ly->spikes_flat,
                B, ly->C_in, H, W, F, T,
                VGG_KH, VGG_KW, VGG_PAD, VGG_CONV_STRIDE,
                1.0f, 0.0f, 0.5f, 0.5f);
        } else {
            /* ── Decomposed: im2col + BLAS GEMM + BN + native LIF ── */
            decomposed_conv_bn_lif(
                ly->act_in, ly->im2col_buf, ly->gemm_buf,
                ly->weight, ly->bn_scale, ly->bn_bias,
                ly->membrane, ly->spikes_flat,
                TB, ly->C_in, H, W, F, T, B);
        }

        /* ── Reshape spikes → NCHW for next layer ── */
        if (ly->pool_after) {
            /* reshape_tmp is pre-allocated in pool (Phase 4: zero malloc in hot loop) */
            reshape_spikes_to_nchw(ly->spikes_flat, ly->reshape_tmp, T, B, OH, OW, F);
            maxpool2d_nchw(ly->reshape_tmp, ly->act_out, TB, F, OH, OW);
        } else {
            reshape_spikes_to_nchw(ly->spikes_flat, ly->act_out, T, B, OH, OW, F);
        }
    }

    /* ── Temporal mean: (T, B, C, H, W) → (B, C, H, W) ── */
    ConvLayer* last = &e->layers[NUM_CONV_LAYERS - 1];
    int last_H = last->pool_after ? last->OH / POOL_S : last->OH;
    int last_W = last->pool_after ? last->OW / POOL_S : last->OW;
    int last_C = last->C_out;
    long spatial = (long)B * last_C * last_H * last_W;
    float inv_T = 1.0f / (float)T;

    #pragma omp parallel for schedule(static)
    for (long s = 0; s < spatial; s++) {
        float sum = 0;
        for (int t = 0; t < T; t++)
            sum += last->act_out[t * spatial + s];
        e->tmean_buf[s] = sum * inv_T;
    }

    /* ── Global average pool: (B, C, H, W) → (B, C) ── */
    int hw = last_H * last_W;
    float inv_hw = 1.0f / (float)hw;
    #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < B * last_C; bc++) {
        float sum = 0;
        for (int i = 0; i < hw; i++)
            sum += e->tmean_buf[bc * hw + i];
        e->gap_buf[bc] = sum * inv_hw;
    }

    /* ── FC: (B, 512) @ (512, num_classes)^T ── */
#ifdef HAS_BLAS
    scipy_cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
                B, e->num_classes, 512,
                1.0f, e->gap_buf, 512,
                e->fc_weight, 512,
                0.0f, e->fc_output, e->num_classes);
#else
    for (int b = 0; b < B; b++)
        for (int c = 0; c < e->num_classes; c++) {
            float s = 0;
            for (int k = 0; k < 512; k++)
                s += e->gap_buf[b*512+k] * e->fc_weight[c*512+k];
            e->fc_output[b*e->num_classes+c] = s;
        }
#endif
}


double vgg9_benchmark(VGG9Engine* e, int warmup, int iters)
{
    for (int i = 0; i < warmup; i++) {
        vgg9_execute(e);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) {
        vgg9_execute(e);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return (iters > 0) ? ms / iters : 0.0;
}
