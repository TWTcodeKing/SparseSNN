/**
 * Fused Conv2d + BN + LIF using MLAS GEMM backend.
 *
 * Fork of MlasConvOperation from MLAS convolve.cpp.
 * Replaces MlasActivation (ReLU) with temporal-interleaved BN+LIF:
 *
 *   For each output segment (N-tile):
 *     For each K-tile:
 *       MlasConvIm2Col()         ← MLAS im2col (optimized)
 *       MlasSgemmOperation()     ← MLAS GEMM (FMA3 assembly micro-kernel)
 *     ──── GEMM result in SegmentOutput (L2-hot) ────
 *     For t in T timesteps:       ← temporal interleaving
 *       BN: h = decay * membrane + recip_tau * (gemm * scale + bias)
 *       LIF: spike = (h >= thresh), membrane = (1-spike)*h
 *
 * The segment (FilterCount × CountN floats) stays L2-hot between GEMM and
 * the LIF T-loop. Membrane persists across T iterations in L1/L2.
 *
 * Compile: link against libmlas.a from openvinotoolkit/mlas
 */

#include <cstddef>
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#define SIMD_WIDTH 8
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4
#else
#define SIMD_WIDTH 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* MLAS public API */
#include "mlas.h"

/* ═══════════════════════════════════════════════════════════════════
 * OpenMP-based MLAS ThreadPool — enables multi-threaded GEMM/im2col
 * ═══════════════════════════════════════════════════════════════════ */
class OmpMlasThreadPool : public IMlasThreadPool {
    int n_threads_;
public:
    explicit OmpMlasThreadPool(int n) : n_threads_(n > 0 ? n : 1) {}

    size_t DegreeOfParallelism() override {
        return (size_t)n_threads_;
    }

    void TrySimpleParallelFor(
        const std::ptrdiff_t total,
        const std::function<void(std::ptrdiff_t)>& fn) override
    {
        if (total <= 1 || n_threads_ <= 1) {
            for (std::ptrdiff_t i = 0; i < total; i++) fn(i);
            return;
        }
        #pragma omp parallel for schedule(static) num_threads(n_threads_)
        for (std::ptrdiff_t i = 0; i < total; i++) {
            fn(i);
        }
    }
};

/* Internal MLAS functions (C++ linkage, defined in convolve.cpp / sgemm.cpp) */
void MlasConvIm2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input, float* ColumnBuffer,
    size_t k, size_t CountK, size_t n, size_t CountN);

void MlasSgemmOperation(
    CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    size_t M, size_t N, size_t K,
    float alpha, const float* A, size_t lda,
    const float* B, size_t ldb,
    float beta, float* C, size_t ldc);

/* ═══════════════════════════════════════════════════════════════════
 * BN + LIF epilogue: applied to GEMM output segment
 *
 * Processes one N-segment of the GEMM output across T timesteps.
 * GEMM output layout: (FilterCount, OutputSize) — one column per output position.
 * We process CountN columns starting at SegmentOutput.
 *
 * For SNN temporal interleaving:
 *   - Input GEMM was computed for ALL T*B frames merged (im2col merges T and B)
 *   - But membrane state requires per-spatial-position T-loop
 *   - So we reinterpret the GEMM output as T slices of B*OH*OW positions
 * ═══════════════════════════════════════════════════════════════════ */
static void apply_bn_lif_segment(
    float* SegmentOutput,    /* (FilterCount, OutputSize) col-major segment */
    const float* Bias,       /* (FilterCount,) BN bias (scale=1 after fold) */
    const float* BnScale,    /* (FilterCount,) */
    float* Membrane,         /* (FilterCount, SpatialPerT) flat */
    float* Spikes,           /* (FilterCount, OutputSize) output spikes */
    size_t FilterCount,      /* number of output channels */
    size_t OutputSize,       /* T * B * OH * OW total output positions */
    size_t CountN,           /* number of columns in this segment */
    size_t SegmentStart,     /* column offset into output */
    size_t SpatialPerT,      /* B * OH * OW (positions per timestep) */
    int T,
    float v_threshold,
    float decay,
    float recip_tau)
{
    /*
     * MLAS GEMM output is column-major:
     *   SegmentOutput[f * OutputSize + (SegmentStart + col)]
     * where f = filter (0..FilterCount-1), col = spatial position (0..CountN-1)
     *
     * For the T-loop, column index maps to (t, spatial_pos):
     *   t = (SegmentStart + col) / SpatialPerT
     *   spatial_pos = (SegmentStart + col) % SpatialPerT
     */

    for (size_t col = 0; col < CountN; col++) {
        size_t abs_col = SegmentStart + col;
        size_t t = abs_col / SpatialPerT;
        size_t sp = abs_col % SpatialPerT;

        for (size_t f = 0; f < FilterCount; f++) {
            float gemm_val = SegmentOutput[f * OutputSize + col];
            float bn = gemm_val * BnScale[f] + (Bias ? Bias[f] : 0.0f);
            float* mem = &Membrane[f * SpatialPerT + sp];

            float h = decay * (*mem) + recip_tau * bn;
            float spike = (h >= v_threshold) ? 1.0f : 0.0f;
            *mem = (1.0f - spike) * h;

            Spikes[f * OutputSize + abs_col] = spike;
        }
    }
}

/* SIMD-vectorized BN+LIF epilogue — spatial-major membrane/spike layout
 *
 * Membrane: (SpatialPerT, FilterCount)  — membrane[sp * F + f] CONTIGUOUS per position
 * Spikes:   (OutputSize, FilterCount)    — spikes[col * F + f]  CONTIGUOUS per position
 * GEMM out: (FilterCount, OutputSize)    — filter-major (MLAS output, can't change)
 *
 * For each column (spatial position), we:
 *   1. Gather SIMD_WIDTH GEMM values from filter-major layout (stride = OutputSize)
 *   2. Load SIMD_WIDTH membrane values contiguously (spatial-major)
 *   3. Compute BN+LIF
 *   4. Store membrane + spikes contiguously
 *
 * Dual ISA: AVX2 (8-wide) on x86, NEON (4-wide) on AArch64, scalar fallback.
 */
static void apply_bn_lif_segment_simd(
    float* SegmentOutput,    /* (FilterCount, OutputSize) filter-major */
    const float* Bias,
    const float* BnScale,
    float* Membrane,         /* (SpatialPerT, FilterCount) spatial-major */
    float* Spikes,           /* (OutputSize, FilterCount) spatial-major */
    size_t FilterCount,
    size_t OutputSize,
    size_t CountN,
    size_t SegmentStart,
    size_t SpatialPerT,
    int T,
    float v_threshold,
    float decay,
    float recip_tau)
{
    size_t F = FilterCount;

    for (size_t col = 0; col < CountN; col++) {
        size_t abs_col = SegmentStart + col;
        size_t sp = abs_col % SpatialPerT;

        float* mem_row = &Membrane[sp * F];
        float* spk_row = &Spikes[abs_col * F];

        size_t f = 0;

#if defined(__AVX2__)
        /* ── AVX2: 8-wide ── */
        __m256 v_thr = _mm256_set1_ps(v_threshold);
        __m256 v_dec = _mm256_set1_ps(decay);
        __m256 v_rt  = _mm256_set1_ps(recip_tau);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_zero = _mm256_setzero_ps();

        for (; f + 7 < F; f += 8) {
            __m256 gemm = _mm256_set_ps(
                SegmentOutput[(f+7)*OutputSize + col],
                SegmentOutput[(f+6)*OutputSize + col],
                SegmentOutput[(f+5)*OutputSize + col],
                SegmentOutput[(f+4)*OutputSize + col],
                SegmentOutput[(f+3)*OutputSize + col],
                SegmentOutput[(f+2)*OutputSize + col],
                SegmentOutput[(f+1)*OutputSize + col],
                SegmentOutput[(f+0)*OutputSize + col]);

            __m256 scale = _mm256_loadu_ps(&BnScale[f]);
            __m256 bias  = Bias ? _mm256_loadu_ps(&Bias[f]) : v_zero;
            __m256 bn    = _mm256_fmadd_ps(gemm, scale, bias);
            __m256 mem   = _mm256_loadu_ps(&mem_row[f]);
            __m256 h     = _mm256_fmadd_ps(v_rt, bn, _mm256_mul_ps(v_dec, mem));
            __m256 cmp   = _mm256_cmp_ps(h, v_thr, _CMP_GE_OS);
            __m256 spike = _mm256_and_ps(cmp, v_one);
            __m256 nm    = _mm256_mul_ps(_mm256_sub_ps(v_one, spike), h);

            _mm256_storeu_ps(&mem_row[f], nm);
            _mm256_storeu_ps(&spk_row[f], spike);
        }

#elif defined(__aarch64__) || defined(__ARM_NEON)
        /* ── NEON: 4-wide ── */
        float32x4_t v_thr = vdupq_n_f32(v_threshold);
        float32x4_t v_dec = vdupq_n_f32(decay);
        float32x4_t v_rt  = vdupq_n_f32(recip_tau);
        float32x4_t v_one = vdupq_n_f32(1.0f);
        float32x4_t v_zero = vdupq_n_f32(0.0f);

        for (; f + 3 < F; f += 4) {
            /* Gather 4 GEMM values from filter-major layout */
            float g_buf[4] = {
                SegmentOutput[(f+0)*OutputSize + col],
                SegmentOutput[(f+1)*OutputSize + col],
                SegmentOutput[(f+2)*OutputSize + col],
                SegmentOutput[(f+3)*OutputSize + col]
            };
            float32x4_t gemm = vld1q_f32(g_buf);

            /* BN: gemm * scale + bias */
            float32x4_t scale = vld1q_f32(&BnScale[f]);
            float32x4_t bias  = Bias ? vld1q_f32(&Bias[f]) : v_zero;
            float32x4_t bn    = vfmaq_f32(bias, gemm, scale);

            /* Load membrane — contiguous */
            float32x4_t mem = vld1q_f32(&mem_row[f]);

            /* LIF: h = decay * mem + recip_tau * bn */
            float32x4_t h = vfmaq_f32(vmulq_f32(v_dec, mem), v_rt, bn);

            /* spike = (h >= threshold) ? 1.0 : 0.0 */
            uint32x4_t cmp   = vcgeq_f32(h, v_thr);
            float32x4_t spike = vreinterpretq_f32_u32(vandq_u32(
                                    cmp, vreinterpretq_u32_f32(v_one)));

            /* membrane = (1 - spike) * h */
            float32x4_t nm = vmulq_f32(vsubq_f32(v_one, spike), h);

            /* Store contiguous */
            vst1q_f32(&mem_row[f], nm);
            vst1q_f32(&spk_row[f], spike);
        }
#endif

        /* Scalar tail (handles remainder for any ISA) */
        for (; f < F; f++) {
            float gv = SegmentOutput[f * OutputSize + col];
            float bn_v = gv * BnScale[f] + (Bias ? Bias[f] : 0.0f);
            float h = decay * mem_row[f] + recip_tau * bn_v;
            float sp_v = (h >= v_threshold) ? 1.0f : 0.0f;
            mem_row[f] = (1.0f - sp_v) * h;
            spk_row[f] = sp_v;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * Public API: Conv2d + BN + LIF using MLAS GEMM
 *
 * Same interface as MLAS MlasConv but with BN+LIF instead of Activation.
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
void MlasConvBnLif(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* BnScale,
    const float* BnBias,
    float* WorkingBuffer,
    float* Output,
    float* Membrane,
    float* Spikes,
    size_t SpatialPerT,
    int T,
    float v_threshold,
    float decay,
    float recip_tau,
    MLAS_THREADPOOL* ThreadPool)
{
    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    int n_threads = ThreadPool ? (int)ThreadPool->DegreeOfParallelism() : 1;

    /* Adaptive stride selection (from MLAS convolve.cpp) */
    uint32_t StrideN = 128;
    uint32_t StrideK = 128;

    if (OutputSize >= K) {
        while (StrideK / 2 >= K) { StrideN *= 2; StrideK /= 2; }
    } else {
        while (StrideN > 16 && StrideN / 2 >= OutputSize) {
            StrideK *= 2; StrideN /= 2;
        }
    }

    /* Step through N-segments: im2col + GEMM (threaded GEMM within each segment) */
    size_t CountN;
    for (size_t n = 0; n < OutputSize; n += CountN) {
        CountN = std::min(OutputSize - n, (size_t)StrideN);

        size_t CountK;
        float beta = Parameters->Beta;
        float* SegmentOutput = Output + n;

        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, (size_t)StrideK);

            MlasConvIm2Col(Parameters, Input, WorkingBuffer,
                           k, CountK, n, CountN);

            MLAS_SGEMM_DATA_PARAMS GemmData;
            GemmData.A = Filter + k;
            GemmData.lda = K;
            GemmData.B = WorkingBuffer;
            GemmData.ldb = CountN;
            GemmData.C = SegmentOutput;
            GemmData.ldc = OutputSize;
            GemmData.alpha = 1.0f;
            GemmData.beta = beta;
            MlasGemm(CblasNoTrans, CblasNoTrans,
                      FilterCount, CountN, CountK,
                      GemmData, ThreadPool);

            beta = 1.0f;
        }
    }

    /* BN + LIF on full Conv output — parallel over spatial positions */
    #pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
    for (size_t col = 0; col < OutputSize; col++) {
        size_t sp = col % SpatialPerT;
        float* mem_row = &Membrane[sp * FilterCount];
        float* spk_row = &Spikes[col * FilterCount];

        size_t f = 0;
#if defined(__AVX2__)
        __m256 v_thr = _mm256_set1_ps(v_threshold);
        __m256 v_dec = _mm256_set1_ps(decay);
        __m256 v_rt  = _mm256_set1_ps(recip_tau);
        __m256 v_one = _mm256_set1_ps(1.0f);
        __m256 v_zero = _mm256_setzero_ps();

        for (; f + 7 < FilterCount; f += 8) {
            __m256 gemm = _mm256_set_ps(
                Output[(f+7)*OutputSize + col], Output[(f+6)*OutputSize + col],
                Output[(f+5)*OutputSize + col], Output[(f+4)*OutputSize + col],
                Output[(f+3)*OutputSize + col], Output[(f+2)*OutputSize + col],
                Output[(f+1)*OutputSize + col], Output[(f+0)*OutputSize + col]);
            __m256 scale = _mm256_loadu_ps(&BnScale[f]);
            __m256 bias  = BnBias ? _mm256_loadu_ps(&BnBias[f]) : v_zero;
            __m256 bn    = _mm256_fmadd_ps(gemm, scale, bias);
            __m256 mem   = _mm256_loadu_ps(&mem_row[f]);
            __m256 h     = _mm256_fmadd_ps(v_rt, bn, _mm256_mul_ps(v_dec, mem));
            __m256 cmp   = _mm256_cmp_ps(h, v_thr, _CMP_GE_OS);
            __m256 spike = _mm256_and_ps(cmp, v_one);
            __m256 nm    = _mm256_mul_ps(_mm256_sub_ps(v_one, spike), h);
            _mm256_storeu_ps(&mem_row[f], nm);
            _mm256_storeu_ps(&spk_row[f], spike);
        }
#elif defined(__aarch64__) || defined(__ARM_NEON)
        float32x4_t v_thr = vdupq_n_f32(v_threshold);
        float32x4_t v_dec = vdupq_n_f32(decay);
        float32x4_t v_rt  = vdupq_n_f32(recip_tau);
        float32x4_t v_one = vdupq_n_f32(1.0f);
        float32x4_t v_zero = vdupq_n_f32(0.0f);

        for (; f + 3 < FilterCount; f += 4) {
            float g_buf[4] = {
                Output[(f+0)*OutputSize + col], Output[(f+1)*OutputSize + col],
                Output[(f+2)*OutputSize + col], Output[(f+3)*OutputSize + col]};
            float32x4_t gemm = vld1q_f32(g_buf);
            float32x4_t scale = vld1q_f32(&BnScale[f]);
            float32x4_t bias  = BnBias ? vld1q_f32(&BnBias[f]) : v_zero;
            float32x4_t bn    = vfmaq_f32(bias, gemm, scale);
            float32x4_t mem = vld1q_f32(&mem_row[f]);
            float32x4_t h = vfmaq_f32(vmulq_f32(v_dec, mem), v_rt, bn);
            uint32x4_t cmp   = vcgeq_f32(h, v_thr);
            float32x4_t spike = vreinterpretq_f32_u32(vandq_u32(cmp, vreinterpretq_u32_f32(v_one)));
            float32x4_t nm = vmulq_f32(vsubq_f32(v_one, spike), h);
            vst1q_f32(&mem_row[f], nm);
            vst1q_f32(&spk_row[f], spike);
        }
#endif
        for (; f < FilterCount; f++) {
            float gv = Output[f * OutputSize + col];
            float bn_v = gv * BnScale[f] + (BnBias ? BnBias[f] : 0.0f);
            float h = decay * mem_row[f] + recip_tau * bn_v;
            float sp_v = (h >= v_threshold) ? 1.0f : 0.0f;
            mem_row[f] = (1.0f - sp_v) * h;
            spk_row[f] = sp_v;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * Convenience: single-layer benchmark entry point
 *
 * Sets up MLAS_CONV_PARAMETERS and calls MlasConvBnLif.
 * ═══════════════════════════════════════════════════════════════════ */
extern "C"
void mlas_conv2d_bn_lif(
    const float* input,      /* (TB, C_in, H, W) NCHW */
    const float* weight,     /* (C_out, C_in, KH, KW) OIHW */
    const float* bn_scale,   /* (C_out,) */
    const float* bn_bias,    /* (C_out,) */
    float* membrane,         /* (B*OH*OW, C_out) spatial-major */
    float* spikes,           /* (T*B*OH*OW, C_out) spatial-major */
    int B, int C_in, int H, int W, int C_out, int T,
    int KH, int KW, int pad, int stride,
    float v_threshold, float decay, float recip_tau,
    int n_threads)
{
    int OH = (H + 2*pad - KH) / stride + 1;
    int OW = (W + 2*pad - KW) / stride + 1;
    size_t SpatialPerT = (size_t)B * OH * OW;
    size_t OutputSize = (size_t)T * SpatialPerT;
    size_t FilterCount = C_out;

    /* Create thread pool */
    OmpMlasThreadPool* pool = nullptr;
    if (n_threads > 1) {
        pool = new OmpMlasThreadPool(n_threads);
    }

    /* Setup MLAS conv parameters */
    int64_t InputShape[2] = {H, W};
    int64_t KernelShape[2] = {KH, KW};
    int64_t DilationShape[2] = {1, 1};
    int64_t Padding[4] = {pad, pad, pad, pad};
    int64_t StrideShape[2] = {stride, stride};
    int64_t OutputShape[2] = {OH, OW};

    MLAS_CONV_PARAMETERS ConvParams;
    size_t WorkingBufferSize = 0;

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MlasConvPrepare(&ConvParams,
        2, T * B, 1, C_in,
        InputShape, KernelShape, DilationShape,
        Padding, StrideShape, OutputShape,
        FilterCount, &Activation, &WorkingBufferSize,
        0.0f, pool);

    float* WorkingBuffer = (float*)aligned_alloc(64,
        std::max(WorkingBufferSize, (size_t)64) * sizeof(float));
    float* GemmOutput = (float*)aligned_alloc(64,
        FilterCount * OutputSize * sizeof(float));

    memset(membrane, 0, FilterCount * SpatialPerT * sizeof(float));

    MlasConvBnLif(&ConvParams, input, weight, bn_scale, bn_bias,
                   WorkingBuffer, GemmOutput, membrane, spikes,
                   SpatialPerT, T, v_threshold, decay, recip_tau, pool);

    free(WorkingBuffer);
    free(GemmOutput);
    delete pool;
}

/* Stub for missing MLAS platform function */
size_t getCacheSizeMlas(int level, bool perCore) {
    (void)perCore;
    switch (level) {
        case 1: return 32768;      /* 32KB L1 */
        case 2: return 524288;     /* 512KB L2 */
        case 3: return 33554432;   /* 32MB L3 */
        default: return 524288;
    }
}

/* GEMM-only: same as fused but skip the LIF epilogue (write raw GEMM output) */
extern "C"
void mlas_conv2d_bn_only(
    const float* input, const float* weight,
    const float* bn_scale, const float* bn_bias,
    float* output,
    int B, int C_in, int H, int W, int C_out, int T,
    int KH, int KW, int pad, int stride_hw)
{
    int OH = (H + 2*pad - KH) / stride_hw + 1;
    int OW = (W + 2*pad - KW) / stride_hw + 1;
    size_t SpatialPerT = (size_t)B * OH * OW;
    size_t OutputSize = (size_t)T * SpatialPerT;
    size_t FilterCount = C_out;
    size_t K_gemm = (size_t)C_in * KH * KW;

    int64_t InputShape[2]={H,W}, KernelShape[2]={KH,KW}, DilationShape[2]={1,1};
    int64_t Padding[4]={pad,pad,pad,pad}, StrideShape[2]={stride_hw,stride_hw};
    int64_t OutputShape[2]={OH,OW};

    MLAS_ACTIVATION Act; Act.ActivationKind = MlasIdentityActivation;
    MLAS_CONV_PARAMETERS CP; size_t WBS=0;
    MlasConvPrepare(&CP,2,T*B,1,C_in,InputShape,KernelShape,DilationShape,
                     Padding,StrideShape,OutputShape,FilterCount,&Act,&WBS,0.0f,nullptr);

    float* WB=(float*)aligned_alloc(64, std::max(WBS,(size_t)64)*sizeof(float));

    /* Conv + BN (scale+bias), no activation */
    uint32_t StrideN=128, StrideK=128;
    if (OutputSize >= K_gemm) { while (StrideK/2>=K_gemm){StrideN*=2;StrideK/=2;} }
    else { while (StrideN>16&&StrideN/2>=OutputSize){StrideK*=2;StrideN/=2;} }

    size_t CountN;
    for (size_t n=0; n<OutputSize; n+=CountN) {
        CountN=std::min(OutputSize-n,(size_t)StrideN);
        size_t CountK; float beta=0.0f;
        float* SegOut = output + n;
        for (size_t k=0; k<K_gemm; k+=CountK) {
            CountK=std::min(K_gemm-k,(size_t)StrideK);
            MlasConvIm2Col(&CP,input,WB,k,CountK,n,CountN);
            MlasSgemmOperation(CblasNoTrans,CblasNoTrans,FilterCount,CountN,CountK,
                1.0f,weight+k,K_gemm,WB,CountN,beta,SegOut,OutputSize);
            beta=1.0f;
        }
        /* Apply BN only (no LIF) */
        for (size_t col=0; col<CountN; col++)
            for (size_t f=0; f<FilterCount; f++)
                SegOut[f*OutputSize+col] = SegOut[f*OutputSize+col]*bn_scale[f] + bn_bias[f];
    }
    free(WB);
}
