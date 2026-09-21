/**
 * Native CPU kernels for sengine_cpu.
 *
 * Hand-optimized C kernels with OpenMP thread parallelism and
 * AVX2 SIMD intrinsics for vectorized element-wise operations.
 *
 * All data is FP32, NHWC layout for spatial ops.
 * Membrane potentials are FP32, updated in-place.
 *
 * GEMM delegates to OpenBLAS cblas_sgemm when available,
 * with a naive triple-loop fallback otherwise.
 */

#include "native_kernels.h"

#include <math.h>
#include <float.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <immintrin.h>

/* ─────────────────────────────────────────────────────────────────────
 * BLAS header (optional — link with -lopenblas)
 * ───────────────────────────────────────────────────────────────────── */
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

/* ─────────────────────────────────────────────────────────────────────
 * AVX2 helpers
 * ─────────────────────────────────────────────────────────────────────
 * Functions decorated with __attribute__((target("avx2"))) will use
 * AVX2 even when the translation unit is compiled without -mavx2.
 * The Makefile uses -march=native which enables AVX2 on supporting
 * CPUs anyway, but the attribute provides a safe fallback.
 * ───────────────────────────────────────────────────────────────────── */

/* ─────────────────────────────────────────────────────────────────────
 * IF Neuron — Integrate-and-Fire
 *
 * For each spatial element s, iterate over T timesteps:
 *   h = v + input[t * spatial + s]
 *   spike = (h >= v_threshold) ? 1.0 : 0.0
 *   v = (1.0 - spike) * h          // branchless reset
 *   spikes[t * spatial + s] = spike
 *
 * OpenMP parallel over spatial dim (each spatial elem is independent
 * because the T-loop carries membrane state sequentially).
 * ───────────────────────────────────────────────────────────────────── */
void native_if_neuron(const float* input, float* membrane, float* spikes,
                      int total_elems, int spatial_elems, float v_threshold)
{
    int T = total_elems / spatial_elems;

    #pragma omp parallel for schedule(static)
    for (int s = 0; s < spatial_elems; s++) {
        float v = membrane[s];
        for (int t = 0; t < T; t++) {
            int idx = t * spatial_elems + s;
            float h = v + input[idx];
            float spike = (h >= v_threshold) ? 1.0f : 0.0f;
            v = (1.0f - spike) * h;
            spikes[idx] = spike;
        }
        membrane[s] = v;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * LIF Neuron — Leaky Integrate-and-Fire
 *
 * Same as IF but with exponential leak:
 *   decay = 1.0 - recip_tau
 *   h = decay * v + recip_tau * input
 * ───────────────────────────────────────────────────────────────────── */
void native_lif_neuron(const float* input, float* membrane, float* spikes,
                       int total_elems, int spatial_elems,
                       float v_threshold, float recip_tau)
{
    int T = total_elems / spatial_elems;
    float decay = 1.0f - recip_tau;

    #pragma omp parallel for schedule(static)
    for (int s = 0; s < spatial_elems; s++) {
        float v = membrane[s];
        for (int t = 0; t < T; t++) {
            int idx = t * spatial_elems + s;
            float h = decay * v + recip_tau * input[idx];
            float spike = (h >= v_threshold) ? 1.0f : 0.0f;
            v = (1.0f - spike) * h;
            spikes[idx] = spike;
        }
        membrane[s] = v;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Element-wise Add — AVX2 vectorized + OpenMP
 *
 * Processes 8 floats per AVX2 iteration. Scalar tail for remainder.
 * ───────────────────────────────────────────────────────────────────── */
__attribute__((target("avx2")))
void native_add(const float* a, const float* b, float* out, int n)
{
    int i = 0;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < (n & ~7); i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vc);
    }

    /* Scalar tail */
    for (i = (n & ~7); i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * MaxPool2d — NHWC layout
 *
 * input:  (N, H, W, C)
 * output: (N, OH, OW, C)
 *
 * Supports rectangular kernel, stride, and padding.
 * ───────────────────────────────────────────────────────────────────── */
void native_maxpool2d(const float* input, float* output,
                      int N, int H, int W, int C,
                      int OH, int OW,
                      int kh, int kw, int sh, int sw, int ph, int pw)
{
    int total_out = N * OH * OW * C;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_out; idx++) {
        int c  = idx % C;
        int rem = idx / C;
        int ow = rem % OW;
        rem = rem / OW;
        int oh = rem % OH;
        int n  = rem / OH;

        float max_val = -FLT_MAX;
        for (int ky = 0; ky < kh; ky++) {
            int ih = oh * sh - ph + ky;
            if (ih < 0 || ih >= H) continue;
            for (int kx = 0; kx < kw; kx++) {
                int iw = ow * sw - pw + kx;
                if (iw < 0 || iw >= W) continue;
                float v = input[((n * H + ih) * W + iw) * C + c];
                if (v > max_val) max_val = v;
            }
        }
        output[idx] = max_val;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Global Average Pool — NHWC layout
 *
 * input:  (N, H, W, C)
 * output: (N, C)      [equivalently (N, 1, 1, C)]
 *
 * Reduces over spatial H*W dimensions per (n, c) pair.
 * ───────────────────────────────────────────────────────────────────── */
void native_global_avgpool(const float* input, float* output,
                           int N, int H, int W, int C)
{
    int total_out = N * C;
    int hw = H * W;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_out; idx++) {
        int c = idx % C;
        int n = idx / C;
        float sum = 0.0f;
        for (int i = 0; i < hw; i++) {
            sum += input[(n * hw + i) * C + c];
        }
        output[idx] = sum / (float)hw;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * Temporal Mean
 *
 * input:  (T * spatial_elems) laid out as T contiguous blocks
 * output: (spatial_elems)
 *
 * output[s] = (1/T) * sum_{t=0}^{T-1} input[t * spatial + s]
 * ───────────────────────────────────────────────────────────────────── */
__attribute__((target("avx2")))
void native_temporal_mean(const float* input, float* output,
                          int T, int spatial_elems)
{
    float inv_T = 1.0f / (float)T;

    #pragma omp parallel for schedule(static)
    for (int s = 0; s < spatial_elems; s++) {
        float sum = 0.0f;
        for (int t = 0; t < T; t++) {
            sum += input[t * spatial_elems + s];
        }
        output[s] = sum * inv_T;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * GEMM — C = A @ B^T  (row-major)
 *
 * A: (M, K), B: (N, K), C: (M, N)
 * Weight layout matches the GPU executor: B is stored row-major with
 * shape (N, K), so the operation is C[m,n] = sum_k A[m,k] * B[n,k].
 *
 * Uses OpenBLAS cblas_sgemm when available, otherwise falls back
 * to a naive triple-loop with OpenMP.
 * ───────────────────────────────────────────────────────────────────── */
void native_gemm(const float* A, const float* B, float* C_out,
                 int M, int K, int N)
{
#ifdef HAVE_OPENBLAS
    /* C = A @ B^T
     * cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
     *             M, N, K, 1.0, A, K, B, K, 0.0, C_out, N)
     *
     * A:   (M, K) row-major, ld=K
     * B^T: (K, N) — B stored as (N, K) row-major, ld=K, transposed
     * C:   (M, N) row-major, ld=N
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K,
                B, K,
                0.0f, C_out, N);
#else
    /* Naive triple loop with OpenMP parallelism over rows. */
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C_out[m * N + n] = sum;
        }
    }
#endif
}

/* ─────────────────────────────────────────────────────────────────────
 * Softmax — numerically stable
 *
 * input/output: (outer, inner)
 * For each row r in [0, outer):
 *   max_val = max(input[r, :])
 *   sum_exp = sum(exp(input[r, i] - max_val) for i in inner)
 *   output[r, i] = exp(input[r, i] - max_val) / sum_exp
 *
 * OpenMP over outer dimension. AVX2 for inner loops on large rows.
 * ───────────────────────────────────────────────────────────────────── */
void native_softmax(const float* input, float* output,
                    int outer, int inner)
{
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < outer; o++) {
        const float* row_in = input + o * inner;
        float* row_out = output + o * inner;

        /* Pass 1: find max */
        float max_val = -FLT_MAX;
        for (int i = 0; i < inner; i++) {
            if (row_in[i] > max_val) max_val = row_in[i];
        }

        /* Pass 2: exp(x - max) and sum */
        float sum_exp = 0.0f;
        for (int i = 0; i < inner; i++) {
            float e = expf(row_in[i] - max_val);
            row_out[i] = e;
            sum_exp += e;
        }

        /* Pass 3: normalize */
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < inner; i++) {
            row_out[i] *= inv_sum;
        }
    }
}
