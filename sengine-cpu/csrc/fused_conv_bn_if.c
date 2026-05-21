/**
 * Fused Conv1x1 + BN + IF/LIF with GEBP-style cache blocking.
 *
 * Two paths per layer (auto-selected by K size):
 *
 *   DIRECT (K ≤ KC_THRESH):   original 6×8 micro-kernel, inline epilogue
 *   GEBP   (K > KC_THRESH):   weight packing → KC-blocked GEMM → epilogue
 *
 * GEBP solves the cache-thrashing problem for large K (e.g. K=4608 in
 * VGG-9 deep layers after im2col). Weight packing converts stride-F
 * access to sequential reads; KC-blocking keeps the working set in L1:
 *
 *   Working set per micro-kernel call:
 *     weight panel:  KC × NR × 4 =  8 KB  (L1)
 *     data rows:     MR × KC × 4 =  6 KB  (L1)
 *     accumulators:  MR × NR × 4 = 192  B  (registers)
 *     total:                       ~14 KB  ⊂ 32KB L1
 *
 * MR=6, NR=8 (AVX2), KC=256
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#define NR 8
#define MR 6
#define KC 256      /* KC × NR × 4 = 8KB ⊂ L1 */
#define KC_THRESH 64 /* below this K, skip packing overhead */
#endif

/* ════════════════════════════════════════════════════════════════════
 * Epilogue macros — applied to GEMM result in YMM registers
 * ════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__

#define EPILOGUE_IF(acc, j, bn_scale, bn_bias, mem_row, spike_row, v_thr, v_rst, v_one) \
    do { \
        __m256 _vs = _mm256_loadu_ps(&(bn_scale)[(j)]); \
        __m256 _vb = _mm256_loadu_ps(&(bn_bias)[(j)]); \
        __m256 _vm = _mm256_loadu_ps(&(mem_row)[(j)]); \
        __m256 _vh = _mm256_add_ps(_vm, _mm256_fmadd_ps((acc), _vs, _vb)); \
        __m256 _cmp = _mm256_cmp_ps(_vh, (v_thr), _CMP_GE_OS); \
        __m256 _sp = _mm256_and_ps(_cmp, (v_one)); \
        __m256 _inv = _mm256_sub_ps((v_one), _sp); \
        _mm256_storeu_ps(&(mem_row)[(j)], \
            _mm256_fmadd_ps(_sp, (v_rst), _mm256_mul_ps(_inv, _vh))); \
        _mm256_storeu_ps(&(spike_row)[(j)], _sp); \
    } while (0)

#define EPILOGUE_LIF(acc, j, bn_scale, bn_bias, mem_row, spike_row, \
                     v_thr, v_rst, v_one, v_decay, v_recip_tau) \
    do { \
        __m256 _vs = _mm256_loadu_ps(&(bn_scale)[(j)]); \
        __m256 _vb = _mm256_loadu_ps(&(bn_bias)[(j)]); \
        __m256 _vm = _mm256_loadu_ps(&(mem_row)[(j)]); \
        __m256 _bn = _mm256_fmadd_ps((acc), _vs, _vb); \
        __m256 _vh = _mm256_fmadd_ps((v_recip_tau), _bn, \
                                      _mm256_mul_ps((v_decay), _vm)); \
        __m256 _cmp = _mm256_cmp_ps(_vh, (v_thr), _CMP_GE_OS); \
        __m256 _sp = _mm256_and_ps(_cmp, (v_one)); \
        __m256 _inv = _mm256_sub_ps((v_one), _sp); \
        _mm256_storeu_ps(&(mem_row)[(j)], \
            _mm256_fmadd_ps(_sp, (v_rst), _mm256_mul_ps(_inv, _vh))); \
        _mm256_storeu_ps(&(spike_row)[(j)], _sp); \
    } while (0)


/* ════════════════════════════════════════════════════════════════════
 * Weight packing: (K, F) row-major → (num_panels, K, NR) panel-major
 *
 * Panel p covers columns [p*NR .. p*NR+NR-1].
 * packed[p * K * NR + k * NR + nr] = weight[k * F + p*NR + nr]
 *
 * Sequential access in the GEBP inner loop: stride = NR*4 = 32 bytes.
 * ════════════════════════════════════════════════════════════════════ */

static void pack_weight_panels(const float* __restrict__ weight,
                                float* __restrict__ packed,
                                int K, int F)
{
    int num_panels = F / NR;
    for (int p = 0; p < num_panels; p++) {
        const float* src_col = weight + p * NR;
        float* dst = packed + (long)p * K * NR;
        for (int k = 0; k < K; k++) {
            memcpy(dst + k * NR, src_col + k * F, NR * sizeof(float));
        }
    }
}

/* Public wrapper for vgg9_executor.c to call at build time */
void sengine_pack_weight_panels(const float* weight, float* packed, int K, int F)
{
    pack_weight_panels(weight, packed, K, F);
}


/* ════════════════════════════════════════════════════════════════════
 * GEBP path: KC-blocked GEMM accumulation into acc_buf
 *
 * Processes MR=6 rows × all full NR-panels of F columns.
 * acc_buf layout: [row * F + col], pre-zeroed by caller.
 * ════════════════════════════════════════════════════════════════════ */

static void gebp_gemm_6x8(
    const float* __restrict__ d0, const float* __restrict__ d1,
    const float* __restrict__ d2, const float* __restrict__ d3,
    const float* __restrict__ d4, const float* __restrict__ d5,
    const float* __restrict__ packed_w,
    float* __restrict__ acc_buf,
    int C_in, int F, int num_panels)
{
    for (int kb = 0; kb < C_in; kb += KC) {
        int klen = C_in - kb;
        if (klen > KC) klen = KC;

        for (int p = 0; p < num_panels; p++) {
            int j = p * NR;

            /* Load partial accumulators */
            __m256 a0 = _mm256_loadu_ps(&acc_buf[       j]);
            __m256 a1 = _mm256_loadu_ps(&acc_buf[  F  + j]);
            __m256 a2 = _mm256_loadu_ps(&acc_buf[2*F  + j]);
            __m256 a3 = _mm256_loadu_ps(&acc_buf[3*F  + j]);
            __m256 a4 = _mm256_loadu_ps(&acc_buf[4*F  + j]);
            __m256 a5 = _mm256_loadu_ps(&acc_buf[5*F  + j]);

            /* Packed weight panel: sequential stride = NR*4 = 32 bytes */
            const float* wp = &packed_w[((long)p * C_in + kb) * NR];

            for (int k = 0; k < klen; k++) {
                __m256 vw = _mm256_loadu_ps(&wp[k * NR]);
                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0[kb+k]), vw, a0);
                a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1[kb+k]), vw, a1);
                a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d2[kb+k]), vw, a2);
                a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d3[kb+k]), vw, a3);
                a4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d4[kb+k]), vw, a4);
                a5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d5[kb+k]), vw, a5);
            }

            /* Store partial accumulators */
            _mm256_storeu_ps(&acc_buf[       j], a0);
            _mm256_storeu_ps(&acc_buf[  F  + j], a1);
            _mm256_storeu_ps(&acc_buf[2*F  + j], a2);
            _mm256_storeu_ps(&acc_buf[3*F  + j], a3);
            _mm256_storeu_ps(&acc_buf[4*F  + j], a4);
            _mm256_storeu_ps(&acc_buf[5*F  + j], a5);
        }
    }
}

/* Single-row variant for M remainder */
static void gebp_gemm_1x8(
    const float* __restrict__ dr,
    const float* __restrict__ packed_w,
    float* __restrict__ acc_buf,
    int C_in, int F, int num_panels)
{
    for (int kb = 0; kb < C_in; kb += KC) {
        int klen = C_in - kb;
        if (klen > KC) klen = KC;

        for (int p = 0; p < num_panels; p++) {
            int j = p * NR;
            __m256 acc = _mm256_loadu_ps(&acc_buf[j]);
            const float* wp = &packed_w[((long)p * C_in + kb) * NR];
            for (int k = 0; k < klen; k++)
                acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dr[kb+k]),
                                      _mm256_loadu_ps(&wp[k * NR]), acc);
            _mm256_storeu_ps(&acc_buf[j], acc);
        }
    }
}


/* ════════════════════════════════════════════════════════════════════
 * DIRECT path: original 6×8 micro-kernel (inline GEMM + epilogue)
 *
 * Used when K is small enough that weight fits in cache without
 * packing. No acc_buf overhead.
 * ════════════════════════════════════════════════════════════════════ */

static inline void fused_gemm_bn_if_6x8(
    const float* __restrict__ d0, const float* __restrict__ d1,
    const float* __restrict__ d2, const float* __restrict__ d3,
    const float* __restrict__ d4, const float* __restrict__ d5,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ m0, float* __restrict__ s0,
    float* __restrict__ m1, float* __restrict__ s1,
    float* __restrict__ m2, float* __restrict__ s2,
    float* __restrict__ m3, float* __restrict__ s3,
    float* __restrict__ m4, float* __restrict__ s4,
    float* __restrict__ m5, float* __restrict__ s5,
    int C_in, int F,
    __m256 v_thr, __m256 v_rst, __m256 v_one
) {
    int j = 0;
    for (; j + NR - 1 < F; j += NR) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();

        for (int k = 0; k < C_in; k++) {
            __m256 vw = _mm256_loadu_ps(&weight[k * F + j]);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0[k]), vw, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1[k]), vw, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d2[k]), vw, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d3[k]), vw, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d4[k]), vw, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d5[k]), vw, acc5);
        }

        EPILOGUE_IF(acc0, j, bn_scale, bn_bias, m0, s0, v_thr, v_rst, v_one);
        EPILOGUE_IF(acc1, j, bn_scale, bn_bias, m1, s1, v_thr, v_rst, v_one);
        EPILOGUE_IF(acc2, j, bn_scale, bn_bias, m2, s2, v_thr, v_rst, v_one);
        EPILOGUE_IF(acc3, j, bn_scale, bn_bias, m3, s3, v_thr, v_rst, v_one);
        EPILOGUE_IF(acc4, j, bn_scale, bn_bias, m4, s4, v_thr, v_rst, v_one);
        EPILOGUE_IF(acc5, j, bn_scale, bn_bias, m5, s5, v_thr, v_rst, v_one);
    }
    for (; j < F; j++) {
        float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0;
        for (int k = 0; k < C_in; k++) {
            float wkj = weight[k*F+j];
            a0+=d0[k]*wkj; a1+=d1[k]*wkj; a2+=d2[k]*wkj;
            a3+=d3[k]*wkj; a4+=d4[k]*wkj; a5+=d5[k]*wkj;
        }
        float thr, rst;
        _mm_store_ss(&thr, _mm256_castps256_ps128(v_thr));
        _mm_store_ss(&rst, _mm256_castps256_ps128(v_rst));
#define SCALAR_IF(a, m, s) { \
    float h=(m)[j]+(a)*bn_scale[j]+bn_bias[j]; \
    float sp=(h>=thr)?1.0f:0.0f; \
    (m)[j]=(1.0f-sp)*h+sp*rst; (s)[j]=sp; }
        SCALAR_IF(a0,m0,s0); SCALAR_IF(a1,m1,s1); SCALAR_IF(a2,m2,s2);
        SCALAR_IF(a3,m3,s3); SCALAR_IF(a4,m4,s4); SCALAR_IF(a5,m5,s5);
#undef SCALAR_IF
    }
}

static inline void fused_gemm_bn_if_1x8(
    const float* __restrict__ data_row,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ mem_row, float* __restrict__ spike_row,
    int C_in, int F,
    __m256 v_thr, __m256 v_rst, __m256 v_one
) {
    int j = 0;
    for (; j + NR - 1 < F; j += NR) {
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < C_in; k++)
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&data_row[k]),
                                  _mm256_loadu_ps(&weight[k * F + j]), acc);
        EPILOGUE_IF(acc, j, bn_scale, bn_bias, mem_row, spike_row,
                     v_thr, v_rst, v_one);
    }
    for (; j < F; j++) {
        float a = 0;
        for (int k = 0; k < C_in; k++) a += data_row[k] * weight[k*F+j];
        float thr, rst;
        _mm_store_ss(&thr, _mm256_castps256_ps128(v_thr));
        _mm_store_ss(&rst, _mm256_castps256_ps128(v_rst));
        float h = mem_row[j] + a * bn_scale[j] + bn_bias[j];
        float sp = (h >= thr) ? 1.0f : 0.0f;
        mem_row[j] = (1.0f - sp) * h + sp * rst;
        spike_row[j] = sp;
    }
}

/* ── Direct path LIF variants ── */

static inline void fused_gemm_bn_lif_6x8(
    const float* __restrict__ d0, const float* __restrict__ d1,
    const float* __restrict__ d2, const float* __restrict__ d3,
    const float* __restrict__ d4, const float* __restrict__ d5,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ m0, float* __restrict__ s0,
    float* __restrict__ m1, float* __restrict__ s1,
    float* __restrict__ m2, float* __restrict__ s2,
    float* __restrict__ m3, float* __restrict__ s3,
    float* __restrict__ m4, float* __restrict__ s4,
    float* __restrict__ m5, float* __restrict__ s5,
    int C_in, int F,
    __m256 v_thr, __m256 v_rst, __m256 v_one,
    __m256 v_decay, __m256 v_recip_tau
) {
    int j = 0;
    for (; j + NR - 1 < F; j += NR) {
        __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
        __m256 acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
        __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps();
        for (int k = 0; k < C_in; k++) {
            __m256 vw = _mm256_loadu_ps(&weight[k*F+j]);
            acc0=_mm256_fmadd_ps(_mm256_broadcast_ss(&d0[k]),vw,acc0);
            acc1=_mm256_fmadd_ps(_mm256_broadcast_ss(&d1[k]),vw,acc1);
            acc2=_mm256_fmadd_ps(_mm256_broadcast_ss(&d2[k]),vw,acc2);
            acc3=_mm256_fmadd_ps(_mm256_broadcast_ss(&d3[k]),vw,acc3);
            acc4=_mm256_fmadd_ps(_mm256_broadcast_ss(&d4[k]),vw,acc4);
            acc5=_mm256_fmadd_ps(_mm256_broadcast_ss(&d5[k]),vw,acc5);
        }
        EPILOGUE_LIF(acc0,j,bn_scale,bn_bias,m0,s0,v_thr,v_rst,v_one,v_decay,v_recip_tau);
        EPILOGUE_LIF(acc1,j,bn_scale,bn_bias,m1,s1,v_thr,v_rst,v_one,v_decay,v_recip_tau);
        EPILOGUE_LIF(acc2,j,bn_scale,bn_bias,m2,s2,v_thr,v_rst,v_one,v_decay,v_recip_tau);
        EPILOGUE_LIF(acc3,j,bn_scale,bn_bias,m3,s3,v_thr,v_rst,v_one,v_decay,v_recip_tau);
        EPILOGUE_LIF(acc4,j,bn_scale,bn_bias,m4,s4,v_thr,v_rst,v_one,v_decay,v_recip_tau);
        EPILOGUE_LIF(acc5,j,bn_scale,bn_bias,m5,s5,v_thr,v_rst,v_one,v_decay,v_recip_tau);
    }
    for (; j < F; j++) {
        float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0;
        for (int k=0;k<C_in;k++){
            float wkj=weight[k*F+j];
            a0+=d0[k]*wkj;a1+=d1[k]*wkj;a2+=d2[k]*wkj;
            a3+=d3[k]*wkj;a4+=d4[k]*wkj;a5+=d5[k]*wkj;
        }
        float thr,rst,dec,rt;
        _mm_store_ss(&thr,_mm256_castps256_ps128(v_thr));
        _mm_store_ss(&rst,_mm256_castps256_ps128(v_rst));
        _mm_store_ss(&dec,_mm256_castps256_ps128(v_decay));
        _mm_store_ss(&rt, _mm256_castps256_ps128(v_recip_tau));
#define SCALAR_LIF(a,m,s){float bn=(a)*bn_scale[j]+bn_bias[j];\
    float h=dec*(m)[j]+rt*bn;float sp=(h>=thr)?1.0f:0.0f;\
    (m)[j]=(1.0f-sp)*h+sp*rst;(s)[j]=sp;}
        SCALAR_LIF(a0,m0,s0);SCALAR_LIF(a1,m1,s1);SCALAR_LIF(a2,m2,s2);
        SCALAR_LIF(a3,m3,s3);SCALAR_LIF(a4,m4,s4);SCALAR_LIF(a5,m5,s5);
#undef SCALAR_LIF
    }
}

static inline void fused_gemm_bn_lif_1x8(
    const float* __restrict__ dr, const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ mr, float* __restrict__ sr, int C_in, int F,
    __m256 v_thr, __m256 v_rst, __m256 v_one,
    __m256 v_decay, __m256 v_recip_tau
) {
    int j = 0;
    for (; j + NR - 1 < F; j += NR) {
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < C_in; k++)
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dr[k]),
                                  _mm256_loadu_ps(&weight[k*F+j]), acc);
        EPILOGUE_LIF(acc, j, bn_scale, bn_bias, mr, sr,
                      v_thr, v_rst, v_one, v_decay, v_recip_tau);
    }
    for (; j < F; j++) {
        float a = 0;
        for (int k = 0; k < C_in; k++) a += dr[k]*weight[k*F+j];
        float thr, rst, dec, rt;
        _mm_store_ss(&thr, _mm256_castps256_ps128(v_thr));
        _mm_store_ss(&rst, _mm256_castps256_ps128(v_rst));
        _mm_store_ss(&dec, _mm256_castps256_ps128(v_decay));
        _mm_store_ss(&rt,  _mm256_castps256_ps128(v_recip_tau));
        float bn = a*bn_scale[j]+bn_bias[j];
        float h = dec*mr[j]+rt*bn;
        float sp = (h>=thr)?1.0f:0.0f;
        mr[j]=(1.0f-sp)*h+sp*rst; sr[j]=sp;
    }
}

#endif /* __AVX2__ */


/* ════════════════════════════════════════════════════════════════════
 * Scalar helpers for F-remainder in GEBP path
 * ════════════════════════════════════════════════════════════════════ */

static inline void scalar_tail_if(
    const float* __restrict__ d, const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ mem, float* __restrict__ spike,
    int C_in, int F, int j_start,
    float v_threshold, float v_reset)
{
    for (int j = j_start; j < F; j++) {
        float a = 0;
        for (int k = 0; k < C_in; k++) a += d[k] * weight[k*F+j];
        float h = mem[j] + a * bn_scale[j] + bn_bias[j];
        float sp = (h >= v_threshold) ? 1.0f : 0.0f;
        mem[j] = (1.0f - sp) * h + sp * v_reset;
        spike[j] = sp;
    }
}

static inline void scalar_tail_lif(
    const float* __restrict__ d, const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ mem, float* __restrict__ spike,
    int C_in, int F, int j_start,
    float v_threshold, float v_reset, float decay, float recip_tau)
{
    for (int j = j_start; j < F; j++) {
        float a = 0;
        for (int k = 0; k < C_in; k++) a += d[k] * weight[k*F+j];
        float bn = a * bn_scale[j] + bn_bias[j];
        float h = decay * mem[j] + recip_tau * bn;
        float sp = (h >= v_threshold) ? 1.0f : 0.0f;
        mem[j] = (1.0f - sp) * h + sp * v_reset;
        spike[j] = sp;
    }
}


/* ════════════════════════════════════════════════════════════════════
 * Public API: IF tloop
 * ════════════════════════════════════════════════════════════════════ */

void fused_conv1x1_bn_if_tloop(
    const float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    float* __restrict__ membrane,
    float* __restrict__ spikes,
    int M, int C_in, int F, int T,
    float v_threshold, float v_reset
) {
    /* ── Data repack: (T,M) interleave → (M,T) per-position grouping ── */
    float* repack = (float*)malloc((size_t)M * T * C_in * sizeof(float));
    if (!repack) return;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        for (int t = 0; t < T; t++)
            memcpy(&repack[((long)i * T + t) * C_in],
                   &data[((long)t * M + i) * C_in],
                   (size_t)C_in * sizeof(float));

#ifdef __AVX2__
    __m256 v_thr = _mm256_set1_ps(v_threshold);
    __m256 v_rst = _mm256_set1_ps(v_reset);
    __m256 v_one = _mm256_set1_ps(1.0f);
    int M_body = M - (M % MR);
    int num_panels = F / NR;
    int F_tail = num_panels * NR;  /* start of scalar remainder */

    if (C_in > KC_THRESH) {
        /* ═══ GEBP path: weight packing + KC-blocked GEMM ═══ */
        float* packed_w = (float*)malloc((size_t)num_panels * C_in * NR * sizeof(float));
        if (!packed_w) { free(repack); return; }
        pack_weight_panels(weight, packed_w, C_in, F);

        #pragma omp parallel
        {
            float* acc_buf = (float*)malloc((size_t)MR * F * sizeof(float));

            #pragma omp for schedule(static)
            for (int i = 0; i < M_body; i += MR) {
                float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
                float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;

                for (int t = 0; t < T; t++) {
                    const float* d0=&repack[((long) i   *T+t)*C_in];
                    const float* d1=&repack[((long)(i+1)*T+t)*C_in];
                    const float* d2=&repack[((long)(i+2)*T+t)*C_in];
                    const float* d3=&repack[((long)(i+3)*T+t)*C_in];
                    const float* d4=&repack[((long)(i+4)*T+t)*C_in];
                    const float* d5=&repack[((long)(i+5)*T+t)*C_in];
                    float* s0=spikes+((long)t*M+i)*F;
                    float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                    float* s4=s3+F; float* s5=s4+F;

                    /* KC-blocked GEMM into acc_buf */
                    memset(acc_buf, 0, (size_t)MR * F * sizeof(float));
                    gebp_gemm_6x8(d0,d1,d2,d3,d4,d5,
                                  packed_w, acc_buf, C_in, F, num_panels);

                    /* Epilogue: BN + IF from acc_buf */
                    for (int j = 0; j + NR - 1 < F; j += NR) {
                        __m256 a0=_mm256_loadu_ps(&acc_buf[     j]);
                        __m256 a1=_mm256_loadu_ps(&acc_buf[  F +j]);
                        __m256 a2=_mm256_loadu_ps(&acc_buf[2*F+j]);
                        __m256 a3=_mm256_loadu_ps(&acc_buf[3*F+j]);
                        __m256 a4=_mm256_loadu_ps(&acc_buf[4*F+j]);
                        __m256 a5=_mm256_loadu_ps(&acc_buf[5*F+j]);
                        EPILOGUE_IF(a0,j,bn_scale,bn_bias,m0,s0,v_thr,v_rst,v_one);
                        EPILOGUE_IF(a1,j,bn_scale,bn_bias,m1,s1,v_thr,v_rst,v_one);
                        EPILOGUE_IF(a2,j,bn_scale,bn_bias,m2,s2,v_thr,v_rst,v_one);
                        EPILOGUE_IF(a3,j,bn_scale,bn_bias,m3,s3,v_thr,v_rst,v_one);
                        EPILOGUE_IF(a4,j,bn_scale,bn_bias,m4,s4,v_thr,v_rst,v_one);
                        EPILOGUE_IF(a5,j,bn_scale,bn_bias,m5,s5,v_thr,v_rst,v_one);
                    }
                    /* Scalar tail for F remainder */
                    if (F_tail < F) {
                        scalar_tail_if(d0,weight,bn_scale,bn_bias,m0,s0,C_in,F,F_tail,v_threshold,v_reset);
                        scalar_tail_if(d1,weight,bn_scale,bn_bias,m1,s1,C_in,F,F_tail,v_threshold,v_reset);
                        scalar_tail_if(d2,weight,bn_scale,bn_bias,m2,s2,C_in,F,F_tail,v_threshold,v_reset);
                        scalar_tail_if(d3,weight,bn_scale,bn_bias,m3,s3,C_in,F,F_tail,v_threshold,v_reset);
                        scalar_tail_if(d4,weight,bn_scale,bn_bias,m4,s4,C_in,F,F_tail,v_threshold,v_reset);
                        scalar_tail_if(d5,weight,bn_scale,bn_bias,m5,s5,C_in,F,F_tail,v_threshold,v_reset);
                    }
                }
            }
            free(acc_buf);
        }

        /* Remainder rows (M % MR) — GEBP 1×8 */
        for (int i = M_body; i < M; i++) {
            float* mr = membrane + (long)i * F;
            float* acc1 = (float*)malloc((size_t)F * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float* dr = &repack[((long)i*T+t)*C_in];
                float* sr = spikes + ((long)t*M+i)*F;
                memset(acc1, 0, (size_t)F * sizeof(float));
                gebp_gemm_1x8(dr, packed_w, acc1, C_in, F, num_panels);
                for (int j = 0; j + NR - 1 < F; j += NR) {
                    __m256 a = _mm256_loadu_ps(&acc1[j]);
                    EPILOGUE_IF(a,j,bn_scale,bn_bias,mr,sr,v_thr,v_rst,v_one);
                }
                if (F_tail < F)
                    scalar_tail_if(dr,weight,bn_scale,bn_bias,mr,sr,C_in,F,F_tail,v_threshold,v_reset);
            }
            free(acc1);
        }

        free(packed_w);

    } else {
        /* ═══ DIRECT path: original micro-kernels (small K) ═══ */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M_body; i += MR) {
            float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
            float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;
            for (int t = 0; t < T; t++) {
                const float* d0=&repack[((long) i   *T+t)*C_in];
                const float* d1=&repack[((long)(i+1)*T+t)*C_in];
                const float* d2=&repack[((long)(i+2)*T+t)*C_in];
                const float* d3=&repack[((long)(i+3)*T+t)*C_in];
                const float* d4=&repack[((long)(i+4)*T+t)*C_in];
                const float* d5=&repack[((long)(i+5)*T+t)*C_in];
                float* s0=spikes+((long)t*M+i)*F;
                float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                float* s4=s3+F; float* s5=s4+F;
                fused_gemm_bn_if_6x8(d0,d1,d2,d3,d4,d5, weight, bn_scale, bn_bias,
                                      m0,s0, m1,s1, m2,s2, m3,s3, m4,s4, m5,s5,
                                      C_in, F, v_thr, v_rst, v_one);
            }
        }
        for (int i = M_body; i < M; i++) {
            float* mr = membrane + (long)i * F;
            for (int t = 0; t < T; t++) {
                const float* dr = &repack[((long)i*T+t)*C_in];
                float* sr = spikes + ((long)t*M+i)*F;
                fused_gemm_bn_if_1x8(dr, weight, bn_scale, bn_bias,
                                      mr, sr, C_in, F, v_thr, v_rst, v_one);
            }
        }
    }

#else
    /* Scalar fallback (no AVX2) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float* mr = membrane + (long)i * F;
        for (int t = 0; t < T; t++) {
            const float* dr = &repack[((long)i*T+t)*C_in];
            float* sr = spikes + ((long)t*M+i)*F;
            for (int j = 0; j < F; j++) {
                float a = 0;
                for (int k = 0; k < C_in; k++) a += dr[k] * weight[k*F+j];
                float h = mr[j] + a * bn_scale[j] + bn_bias[j];
                float sp = (h >= v_threshold) ? 1.0f : 0.0f;
                mr[j] = (1.0f - sp) * h + sp * v_reset; sr[j] = sp;
            }
        }
    }
#endif
    free(repack);
}


/* ════════════════════════════════════════════════════════════════════
 * Public API: LIF tloop
 * ════════════════════════════════════════════════════════════════════ */

void fused_conv1x1_bn_lif_tloop(
    const float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    float* __restrict__ membrane,
    float* __restrict__ spikes,
    int M, int C_in, int F, int T,
    float v_threshold, float v_reset,
    float decay, float recip_tau
) {
    float* repack = (float*)malloc((size_t)M * T * C_in * sizeof(float));
    if (!repack) return;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        for (int t = 0; t < T; t++)
            memcpy(&repack[((long)i * T + t) * C_in],
                   &data[((long)t * M + i) * C_in],
                   (size_t)C_in * sizeof(float));

#ifdef __AVX2__
    __m256 v_thr = _mm256_set1_ps(v_threshold);
    __m256 v_rst = _mm256_set1_ps(v_reset);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_dec = _mm256_set1_ps(decay);
    __m256 v_rt  = _mm256_set1_ps(recip_tau);
    int M_body = M - (M % MR);
    int num_panels = F / NR;
    int F_tail = num_panels * NR;

    if (C_in > KC_THRESH) {
        /* ═══ GEBP path ═══ */
        float* packed_w = (float*)malloc((size_t)num_panels * C_in * NR * sizeof(float));
        if (!packed_w) { free(repack); return; }
        pack_weight_panels(weight, packed_w, C_in, F);

        #pragma omp parallel
        {
            float* acc_buf = (float*)malloc((size_t)MR * F * sizeof(float));

            #pragma omp for schedule(static)
            for (int i = 0; i < M_body; i += MR) {
                float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
                float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;
                for (int t = 0; t < T; t++) {
                    const float* d0=&repack[((long) i   *T+t)*C_in];
                    const float* d1=&repack[((long)(i+1)*T+t)*C_in];
                    const float* d2=&repack[((long)(i+2)*T+t)*C_in];
                    const float* d3=&repack[((long)(i+3)*T+t)*C_in];
                    const float* d4=&repack[((long)(i+4)*T+t)*C_in];
                    const float* d5=&repack[((long)(i+5)*T+t)*C_in];
                    float* s0=spikes+((long)t*M+i)*F;
                    float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                    float* s4=s3+F; float* s5=s4+F;

                    memset(acc_buf, 0, (size_t)MR * F * sizeof(float));
                    gebp_gemm_6x8(d0,d1,d2,d3,d4,d5,
                                  packed_w, acc_buf, C_in, F, num_panels);

                    for (int j = 0; j + NR - 1 < F; j += NR) {
                        __m256 a0=_mm256_loadu_ps(&acc_buf[     j]);
                        __m256 a1=_mm256_loadu_ps(&acc_buf[  F +j]);
                        __m256 a2=_mm256_loadu_ps(&acc_buf[2*F+j]);
                        __m256 a3=_mm256_loadu_ps(&acc_buf[3*F+j]);
                        __m256 a4=_mm256_loadu_ps(&acc_buf[4*F+j]);
                        __m256 a5=_mm256_loadu_ps(&acc_buf[5*F+j]);
                        EPILOGUE_LIF(a0,j,bn_scale,bn_bias,m0,s0,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a1,j,bn_scale,bn_bias,m1,s1,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a2,j,bn_scale,bn_bias,m2,s2,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a3,j,bn_scale,bn_bias,m3,s3,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a4,j,bn_scale,bn_bias,m4,s4,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a5,j,bn_scale,bn_bias,m5,s5,v_thr,v_rst,v_one,v_dec,v_rt);
                    }
                    if (F_tail < F) {
                        scalar_tail_lif(d0,weight,bn_scale,bn_bias,m0,s0,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                        scalar_tail_lif(d1,weight,bn_scale,bn_bias,m1,s1,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                        scalar_tail_lif(d2,weight,bn_scale,bn_bias,m2,s2,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                        scalar_tail_lif(d3,weight,bn_scale,bn_bias,m3,s3,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                        scalar_tail_lif(d4,weight,bn_scale,bn_bias,m4,s4,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                        scalar_tail_lif(d5,weight,bn_scale,bn_bias,m5,s5,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
                    }
                }
            }
            free(acc_buf);
        }

        for (int i = M_body; i < M; i++) {
            float* mr = membrane + (long)i * F;
            float* acc1 = (float*)malloc((size_t)F * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float* dr = &repack[((long)i*T+t)*C_in];
                float* sr = spikes + ((long)t*M+i)*F;
                memset(acc1, 0, (size_t)F * sizeof(float));
                gebp_gemm_1x8(dr, packed_w, acc1, C_in, F, num_panels);
                for (int j = 0; j + NR - 1 < F; j += NR) {
                    __m256 a = _mm256_loadu_ps(&acc1[j]);
                    EPILOGUE_LIF(a,j,bn_scale,bn_bias,mr,sr,v_thr,v_rst,v_one,v_dec,v_rt);
                }
                if (F_tail < F)
                    scalar_tail_lif(dr,weight,bn_scale,bn_bias,mr,sr,C_in,F,F_tail,v_threshold,v_reset,decay,recip_tau);
            }
            free(acc1);
        }

        free(packed_w);

    } else {
        /* ═══ DIRECT path ═══ */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M_body; i += MR) {
            float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
            float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;
            for (int t = 0; t < T; t++) {
                const float* d0=&repack[((long) i   *T+t)*C_in];
                const float* d1=&repack[((long)(i+1)*T+t)*C_in];
                const float* d2=&repack[((long)(i+2)*T+t)*C_in];
                const float* d3=&repack[((long)(i+3)*T+t)*C_in];
                const float* d4=&repack[((long)(i+4)*T+t)*C_in];
                const float* d5=&repack[((long)(i+5)*T+t)*C_in];
                float* s0=spikes+((long)t*M+i)*F;
                float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                float* s4=s3+F; float* s5=s4+F;
                fused_gemm_bn_lif_6x8(d0,d1,d2,d3,d4,d5, weight, bn_scale, bn_bias,
                                       m0,s0, m1,s1, m2,s2, m3,s3, m4,s4, m5,s5,
                                       C_in, F, v_thr, v_rst, v_one, v_dec, v_rt);
            }
        }
        for (int i = M_body; i < M; i++) {
            float* mr = membrane + (long)i * F;
            for (int t = 0; t < T; t++) {
                const float* dr = &repack[((long)i*T+t)*C_in];
                float* sr = spikes + ((long)t*M+i)*F;
                fused_gemm_bn_lif_1x8(dr, weight, bn_scale, bn_bias,
                                       mr, sr, C_in, F,
                                       v_thr, v_rst, v_one, v_dec, v_rt);
            }
        }
    }

#else
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float* mr = membrane + (long)i * F;
        for (int t = 0; t < T; t++) {
            const float* dr = &repack[((long)i*T+t)*C_in];
            float* sr = spikes + ((long)t*M+i)*F;
            for (int j = 0; j < F; j++) {
                float a = 0;
                for (int k = 0; k < C_in; k++) a += dr[k]*weight[k*F+j];
                float bn = a*bn_scale[j]+bn_bias[j];
                float h = decay*mr[j]+recip_tau*bn;
                float sp = (h>=v_threshold)?1.0f:0.0f;
                mr[j]=(1.0f-sp)*h+sp*v_reset; sr[j]=sp;
            }
        }
    }
#endif
    free(repack);
}


/* ════════════════════════════════════════════════════════════════════
 * Backward-compat single-timestep wrappers
 * ════════════════════════════════════════════════════════════════════ */

void fused_conv1x1_bn_if(
    const float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    float* __restrict__ membrane,
    float* __restrict__ spikes,
    int M, int C_in, int F,
    float v_threshold, float v_reset
) {
    fused_conv1x1_bn_if_tloop(data, weight, bn_scale, bn_bias,
                               membrane, spikes, M, C_in, F, 1,
                               v_threshold, v_reset);
}

void fused_conv1x1_bn_lif(
    const float* __restrict__ data,
    const float* __restrict__ weight,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    float* __restrict__ membrane,
    float* __restrict__ spikes,
    int M, int C_in, int F,
    float v_threshold, float v_reset,
    float decay, float recip_tau
) {
    fused_conv1x1_bn_lif_tloop(data, weight, bn_scale, bn_bias,
                                membrane, spikes, M, C_in, F, 1,
                                v_threshold, v_reset, decay, recip_tau);
}


/* ════════════════════════════════════════════════════════════════════
 * IMPLICIT IM2COL: Conv2d + BN + LIF (no external im2col buffer)
 *
 * Takes NCHW input directly, gathers 3×3 patches on-the-fly into
 * a small L1-resident "micro im2col" buffer per MR block.
 *
 * Eliminates:
 *   • External im2col buffer allocation (up to 37MB at B=16)
 *   • im2col memory bandwidth (write + read = 2× data traffic)
 *   • 25% of B=1 latency
 *
 * Gather buffer: MR × KC × 4 = 6KB ⊂ L1 (32KB)
 * ════════════════════════════════════════════════════════════════════ */

/* Gather one row of K-range values from NCHW input for one spatial position */
static inline void gather_conv_row(
    const float* __restrict__ img,  /* (C_in, H, W) one image frame */
    float* __restrict__ out,        /* klen output values */
    int oh, int ow,                 /* output spatial position */
    int C_in, int H, int W,
    int KH, int KW, int pad, int stride,
    int k_start, int klen)
{
    int KK = KH * KW;
    int HW = H * W;
    /* Compute starting (c, ky, kx) from k_start */
    int c  = k_start / KK;
    int rem = k_start - c * KK;
    int ky = rem / KW;
    int kx = rem - ky * KW;

    for (int ki = 0; ki < klen; ki++) {
        int ih = oh * stride + ky - pad;
        int iw = ow * stride + kx - pad;
        out[ki] = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                 ? img[(long)c * HW + ih * W + iw] : 0.0f;
        if (++kx >= KW) { kx = 0; if (++ky >= KH) { ky = 0; c++; } }
    }
}


void fused_conv2d_bn_lif_tloop(
    const float* __restrict__ input,     /* (TB, C_in, H, W) NCHW */
    const float* __restrict__ weight,    /* (K, F) K=C_in*KH*KW */
    const float* __restrict__ bn_scale,  /* (F,) */
    const float* __restrict__ bn_bias,   /* (F,) */
    float* __restrict__ membrane,        /* (M, F) M=B*OH*OW */
    float* __restrict__ spikes,          /* (T*M, F) */
    int B, int C_in, int H, int W, int F, int T,
    int KH, int KW, int pad, int stride_hw,
    float v_threshold, float v_reset,
    float decay, float recip_tau
) {
    int OH = (H + 2*pad - KH) / stride_hw + 1;
    int OW = (W + 2*pad - KW) / stride_hw + 1;
    int M = B * OH * OW;
    int K = C_in * KH * KW;
    long CHW = (long)C_in * H * W;

#ifdef __AVX2__
    __m256 v_thr = _mm256_set1_ps(v_threshold);
    __m256 v_rst = _mm256_set1_ps(v_reset);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_dec = _mm256_set1_ps(decay);
    __m256 v_rt  = _mm256_set1_ps(recip_tau);
    int M_body = M - (M % MR);
    int num_panels = F / NR;
    int F_tail = num_panels * NR;

    /* Weight packing (once per call — will move to build time in Phase 4) */
    float* packed_w = NULL;
    if (K > KC_THRESH && num_panels > 0) {
        packed_w = (float*)malloc((size_t)num_panels * K * NR * sizeof(float));
        if (!packed_w) return;
        pack_weight_panels(weight, packed_w, K, F);
    }

    if (packed_w) {
        /* ═══ GEBP path with micro im2col ═══ */
        #pragma omp parallel
        {
            float* acc_buf = (float*)malloc((size_t)MR * F * sizeof(float));
            float* gbuf = (float*)malloc((size_t)MR * KC * sizeof(float));

            #pragma omp for schedule(static)
            for (int i = 0; i < M_body; i += MR) {
                float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
                float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;

                /* Decode spatial positions for MR rows */
                int pos_b[MR], pos_oh[MR], pos_ow[MR];
                for (int r = 0; r < MR; r++) {
                    int p = i + r;
                    pos_b[r]  = p / (OH * OW);
                    int sp    = p - pos_b[r] * (OH * OW);
                    pos_oh[r] = sp / OW;
                    pos_ow[r] = sp - pos_oh[r] * OW;
                }

                for (int t = 0; t < T; t++) {
                    float* s0=spikes+((long)t*M+i)*F;
                    float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                    float* s4=s3+F; float* s5=s4+F;

                    memset(acc_buf, 0, (size_t)MR * F * sizeof(float));

                    for (int kb = 0; kb < K; kb += KC) {
                        int klen = K - kb;
                        if (klen > KC) klen = KC;

                        /* ── Micro im2col: gather MR × klen into gbuf ── */
                        for (int r = 0; r < MR; r++) {
                            const float* img = input + (long)(t*B + pos_b[r]) * CHW;
                            gather_conv_row(img, &gbuf[r * klen],
                                            pos_oh[r], pos_ow[r],
                                            C_in, H, W, KH, KW, pad, stride_hw,
                                            kb, klen);
                        }

                        /* ── GEBP: packed weight × gathered data ── */
                        for (int p = 0; p < num_panels; p++) {
                            int j = p * NR;
                            __m256 a0 = _mm256_loadu_ps(&acc_buf[       j]);
                            __m256 a1 = _mm256_loadu_ps(&acc_buf[  F  + j]);
                            __m256 a2 = _mm256_loadu_ps(&acc_buf[2*F  + j]);
                            __m256 a3 = _mm256_loadu_ps(&acc_buf[3*F  + j]);
                            __m256 a4 = _mm256_loadu_ps(&acc_buf[4*F  + j]);
                            __m256 a5 = _mm256_loadu_ps(&acc_buf[5*F  + j]);

                            const float* wp = &packed_w[((long)p * K + kb) * NR];
                            const float* d0 = &gbuf[0 * klen];
                            const float* d1 = &gbuf[1 * klen];
                            const float* d2 = &gbuf[2 * klen];
                            const float* d3 = &gbuf[3 * klen];
                            const float* d4 = &gbuf[4 * klen];
                            const float* d5 = &gbuf[5 * klen];

                            for (int k = 0; k < klen; k++) {
                                __m256 vw = _mm256_loadu_ps(&wp[k * NR]);
                                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0[k]), vw, a0);
                                a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1[k]), vw, a1);
                                a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d2[k]), vw, a2);
                                a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d3[k]), vw, a3);
                                a4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d4[k]), vw, a4);
                                a5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d5[k]), vw, a5);
                            }

                            _mm256_storeu_ps(&acc_buf[       j], a0);
                            _mm256_storeu_ps(&acc_buf[  F  + j], a1);
                            _mm256_storeu_ps(&acc_buf[2*F  + j], a2);
                            _mm256_storeu_ps(&acc_buf[3*F  + j], a3);
                            _mm256_storeu_ps(&acc_buf[4*F  + j], a4);
                            _mm256_storeu_ps(&acc_buf[5*F  + j], a5);
                        }
                    }

                    /* ── Epilogue: BN + LIF from acc_buf ── */
                    for (int j = 0; j + NR - 1 < F; j += NR) {
                        __m256 a0=_mm256_loadu_ps(&acc_buf[     j]);
                        __m256 a1=_mm256_loadu_ps(&acc_buf[  F +j]);
                        __m256 a2=_mm256_loadu_ps(&acc_buf[2*F+j]);
                        __m256 a3=_mm256_loadu_ps(&acc_buf[3*F+j]);
                        __m256 a4=_mm256_loadu_ps(&acc_buf[4*F+j]);
                        __m256 a5=_mm256_loadu_ps(&acc_buf[5*F+j]);
                        EPILOGUE_LIF(a0,j,bn_scale,bn_bias,m0,s0,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a1,j,bn_scale,bn_bias,m1,s1,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a2,j,bn_scale,bn_bias,m2,s2,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a3,j,bn_scale,bn_bias,m3,s3,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a4,j,bn_scale,bn_bias,m4,s4,v_thr,v_rst,v_one,v_dec,v_rt);
                        EPILOGUE_LIF(a5,j,bn_scale,bn_bias,m5,s5,v_thr,v_rst,v_one,v_dec,v_rt);
                    }
                    /* Scalar F tail — use original weight for remainder columns */
                    if (F_tail < F) {
                        for (int r = 0; r < MR; r++) {
                            const float* img = input + (long)(t*B + pos_b[r]) * CHW;
                            float* mr = membrane + (long)(i+r)*F;
                            float* sr = spikes + ((long)t*M + i + r)*F;
                            /* Full gather for this row for scalar columns */
                            float gbuf_sc[9*512]; /* max C_in*KH*KW for scalar tail */
                            gather_conv_row(img, gbuf_sc, pos_oh[r], pos_ow[r],
                                           C_in, H, W, KH, KW, pad, stride_hw, 0, K);
                            scalar_tail_lif(gbuf_sc, weight, bn_scale, bn_bias,
                                           mr, sr, K, F, F_tail,
                                           v_threshold, v_reset, decay, recip_tau);
                        }
                    }
                }
            }
            free(acc_buf);
            free(gbuf);
        }

        /* Remainder rows (M % MR) */
        for (int i = M_body; i < M; i++) {
            int b  = i / (OH * OW);
            int sp = i - b * (OH * OW);
            int oh = sp / OW;
            int ow = sp - oh * OW;
            float* mr = membrane + (long)i * F;
            float* acc1 = (float*)malloc((size_t)F * sizeof(float));
            float* gbuf1 = (float*)malloc((size_t)KC * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float* img = input + (long)(t*B + b) * CHW;
                float* sr = spikes + ((long)t*M + i) * F;
                memset(acc1, 0, (size_t)F * sizeof(float));
                for (int kb = 0; kb < K; kb += KC) {
                    int klen = K - kb; if (klen > KC) klen = KC;
                    gather_conv_row(img, gbuf1, oh, ow,
                                   C_in, H, W, KH, KW, pad, stride_hw, kb, klen);
                    for (int p = 0; p < num_panels; p++) {
                        int j = p * NR;
                        __m256 acc = _mm256_loadu_ps(&acc1[j]);
                        const float* wp = &packed_w[((long)p * K + kb) * NR];
                        for (int k = 0; k < klen; k++)
                            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&gbuf1[k]),
                                                  _mm256_loadu_ps(&wp[k * NR]), acc);
                        _mm256_storeu_ps(&acc1[j], acc);
                    }
                }
                for (int j = 0; j + NR - 1 < F; j += NR) {
                    __m256 a = _mm256_loadu_ps(&acc1[j]);
                    EPILOGUE_LIF(a,j,bn_scale,bn_bias,mr,sr,v_thr,v_rst,v_one,v_dec,v_rt);
                }
            }
            free(acc1);
            free(gbuf1);
        }

        free(packed_w);

    } else {
        /* ═══ DIRECT path (small K): micro im2col + inline epilogue ═══ */
        #pragma omp parallel
        {
            float* gbuf = (float*)malloc((size_t)MR * K * sizeof(float));

            #pragma omp for schedule(static)
            for (int i = 0; i < M_body; i += MR) {
                float* m0=membrane+(long)i*F; float* m1=m0+F; float* m2=m1+F;
                float* m3=m2+F; float* m4=m3+F; float* m5=m4+F;

                int pos_b[MR], pos_oh[MR], pos_ow[MR];
                for (int r = 0; r < MR; r++) {
                    int p = i + r;
                    pos_b[r]  = p / (OH * OW);
                    int sp    = p - pos_b[r] * (OH * OW);
                    pos_oh[r] = sp / OW;
                    pos_ow[r] = sp - pos_oh[r] * OW;
                }

                for (int t = 0; t < T; t++) {
                    float* s0=spikes+((long)t*M+i)*F;
                    float* s1=s0+F; float* s2=s1+F; float* s3=s2+F;
                    float* s4=s3+F; float* s5=s4+F;

                    for (int r = 0; r < MR; r++) {
                        const float* img = input + (long)(t*B + pos_b[r]) * CHW;
                        gather_conv_row(img, &gbuf[r * K],
                                       pos_oh[r], pos_ow[r],
                                       C_in, H, W, KH, KW, pad, stride_hw, 0, K);
                    }

                    fused_gemm_bn_lif_6x8(
                        &gbuf[0*K], &gbuf[1*K], &gbuf[2*K],
                        &gbuf[3*K], &gbuf[4*K], &gbuf[5*K],
                        weight, bn_scale, bn_bias,
                        m0,s0, m1,s1, m2,s2, m3,s3, m4,s4, m5,s5,
                        K, F, v_thr, v_rst, v_one, v_dec, v_rt);
                }
            }
            free(gbuf);
        }

        /* Remainder */
        for (int i = M_body; i < M; i++) {
            int b  = i / (OH * OW);
            int sp = i - b * (OH * OW);
            int oh = sp / OW;
            int ow = sp - oh * OW;
            float* mr = membrane + (long)i * F;
            float gbuf1[9*512]; /* stack: max K for direct path */
            for (int t = 0; t < T; t++) {
                const float* img = input + (long)(t*B + b) * CHW;
                float* sr = spikes + ((long)t*M + i) * F;
                gather_conv_row(img, gbuf1, oh, ow,
                               C_in, H, W, KH, KW, pad, stride_hw, 0, K);
                fused_gemm_bn_lif_1x8(gbuf1, weight, bn_scale, bn_bias,
                                       mr, sr, K, F,
                                       v_thr, v_rst, v_one, v_dec, v_rt);
            }
        }
    }

#else
    /* Scalar fallback */
    int OH = (H + 2*pad - KH) / stride_hw + 1;
    /* (already computed above, but needed here for non-AVX2 path) */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        int b  = i / (OH * OW);
        int sp = i - b * (OH * OW);
        int oh = sp / OW;
        int ow = sp - oh * OW;
        float* mr = membrane + (long)i * F;
        for (int t = 0; t < T; t++) {
            const float* img = input + (long)(t*B + b) * CHW;
            float* sr = spikes + ((long)t*M + i) * F;
            for (int j = 0; j < F; j++) {
                float a = 0;
                int k = 0;
                for (int c = 0; c < C_in; c++) {
                    const float* ch = img + (long)c * H * W;
                    for (int ky = 0; ky < KH; ky++) {
                        int ih = oh * stride_hw + ky - pad;
                        for (int kx = 0; kx < KW; kx++) {
                            int iw = ow * stride_hw + kx - pad;
                            float d = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                                     ? ch[ih * W + iw] : 0.0f;
                            a += d * weight[k * F + j];
                            k++;
                        }
                    }
                }
                float bn = a * bn_scale[j] + bn_bias[j];
                float h = decay * mr[j] + recip_tau * bn;
                float sp_v = (h >= v_threshold) ? 1.0f : 0.0f;
                mr[j] = (1.0f - sp_v) * h + sp_v * v_reset;
                sr[j] = sp_v;
            }
        }
    }
#endif
}
