#define _GNU_SOURCE
/**
 * Native NHWC Conv2d + BN + neuron (IF / LIF) with a per-node T loop.
 *
 * Replaces the TVM-compiled conv kernels when no TVM toolchain is available.
 * Layout contract (matches the rest of the CPU executor):
 *   input    (T*B, H, W, C_in)     NHWC fp32
 *   weight   (K, F)                K = KH*KW*C_in, k ordered (kh, kw, cin)
 *   scale    (F,)  bias (F,)       folded BatchNorm
 *   membrane (M, F)                M = B*OH*OW, persistent across the T loop
 *   output   (T*B, OH, OW, F)      NHWC fp32 (spikes, or BN output if neuron=0)
 *   im2col   (M, K)                scratch, owned by the executor
 *
 * GEMM: cblas_sgemm resolved at runtime (scipy's bundled OpenBLAS, prefixed
 * "scipy_cblas_sgemm", or a system "cblas_sgemm"); naive OpenMP fallback.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "native_kernels.h"

typedef void (*cblas_sgemm_fn)(int order, int transA, int transB,
                               int M, int N, int K, float alpha,
                               const float* A, int lda,
                               const float* B, int ldb,
                               float beta, float* C, int ldc);
typedef void (*set_threads_fn)(int);

static cblas_sgemm_fn g_sgemm = NULL;
static int g_sgemm_resolved = 0;

static void resolve_sgemm(void)
{
    if (g_sgemm_resolved) return;
    g_sgemm_resolved = 1;
    const char* names[] = {"scipy_cblas_sgemm", "cblas_sgemm", NULL};
    for (int i = 0; names[i]; i++) {
        void* f = dlsym(RTLD_DEFAULT, names[i]);
        if (f) { g_sgemm = (cblas_sgemm_fn)f; break; }
    }
    if (!g_sgemm)
        fprintf(stderr, "[sengine-cpu] no cblas_sgemm symbol found: using naive GEMM\n");
}

/* Row-major C(M,N) = A(M,K) @ op(B); transB=0: B is (K,N); transB=1: B is (N,K). */
void sengine_sgemm(int transB, int M, int N, int K,
                   const float* A, int lda, const float* B, int ldb,
                   float* C, int ldc)
{
    resolve_sgemm();
    if (g_sgemm) {
        /* CblasRowMajor=101, CblasNoTrans=111, CblasTrans=112 */
        g_sgemm(101, 111, transB ? 112 : 111, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        float* crow = C + (long)m * ldc;
        for (int n = 0; n < N; n++) crow[n] = 0.0f;
        const float* arow = A + (long)m * lda;
        if (transB) {
            for (int n = 0; n < N; n++) {
                const float* brow = B + (long)n * ldb;
                float s = 0.0f;
                for (int k = 0; k < K; k++) s += arow[k] * brow[k];
                crow[n] = s;
            }
        } else {
            for (int k = 0; k < K; k++) {
                float a = arow[k];
                const float* brow = B + (long)k * ldb;
                for (int n = 0; n < N; n++) crow[n] += a * brow[n];
            }
        }
    }
}

void sengine_blas_set_threads(int n)
{
    const char* names[] = {"scipy_openblas_set_num_threads", "openblas_set_num_threads", NULL};
    for (int i = 0; names[i]; i++) {
        void* f = dlsym(RTLD_DEFAULT, names[i]);
        if (f) { ((set_threads_fn)f)(n); return; }
    }
}

void native_conv_bn_neuron(
    const float* input, const float* weight,
    const float* scale, const float* bias,
    float* membrane, float* output, float* im2col,
    int B, int H, int W, int C_in, int F, int T,
    int KH, int KW, int pad, int stride,
    int neuron, float v_threshold, float v_reset, float recip_tau)
{
    const int OH = (H + 2 * pad - KH) / stride + 1;
    const int OW = (W + 2 * pad - KW) / stride + 1;
    const int M = B * OH * OW;
    const int K = KH * KW * C_in;
    const float decay = 1.0f - recip_tau;

    for (int t = 0; t < T; t++) {
        /* im2col for timestep t: row m = (b, oh, ow), k = (kh, kw, cin) */
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            int ow = m % OW, tmp = m / OW;
            int oh = tmp % OH, b = tmp / OH;
            const float* img = input + ((long)(t * B + b) * H) * W * C_in;
            float* row = im2col + (long)m * K;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * stride + kh - pad;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * stride + kw - pad;
                    float* dst = row + (kh * KW + kw) * C_in;
                    if ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                        memcpy(dst, img + ((long)ih * W + iw) * C_in, C_in * sizeof(float));
                    else
                        memset(dst, 0, C_in * sizeof(float));
                }
            }
        }
        float* out_t = output + (long)t * M * F;
        sengine_sgemm(0, M, F, K, im2col, K, weight, F, out_t, F);

        /* BN + neuron epilogue (membrane persists across t) */
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float* o = out_t + (long)m * F;
            float* mem = membrane + (long)m * F;
            for (int f = 0; f < F; f++) {
                float v = o[f] * scale[f] + bias[f];
                if (neuron == 0) { o[f] = v; continue; }
                float h = (neuron == 2) ? (decay * mem[f] + recip_tau * v) : (mem[f] + v);
                float sp = (h >= v_threshold) ? 1.0f : 0.0f;
                mem[f] = sp * v_reset + (1.0f - sp) * h;
                o[f] = sp;
            }
        }
    }
}
