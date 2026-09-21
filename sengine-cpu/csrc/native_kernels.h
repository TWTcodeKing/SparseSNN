/**
 * Native CPU kernels for sengine-cpu.
 *
 * All kernels operate on FP32 data in NHWC layout.
 * OpenMP for thread parallelism, AVX2 SIMD where beneficial.
 */

#ifndef SENGINE_CPU_NATIVE_KERNELS_H
#define SENGINE_CPU_NATIVE_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void native_if_neuron(const float* input, float* membrane, float* spikes,
                      int total_elems, int spatial_elems, float v_threshold);

void native_lif_neuron(const float* input, float* membrane, float* spikes,
                       int total_elems, int spatial_elems,
                       float v_threshold, float recip_tau);

void native_add(const float* a, const float* b, float* out, int n);

void native_maxpool2d(const float* input, float* output,
                      int N, int H, int W, int C,
                      int OH, int OW,
                      int kh, int kw, int sh, int sw, int ph, int pw);

void native_global_avgpool(const float* input, float* output,
                           int N, int H, int W, int C);

void native_temporal_mean(const float* input, float* output,
                          int T, int spatial_elems);

void native_gemm(const float* A, const float* B, float* C,
                 int M, int K, int N);

void native_conv_bn_neuron(
    const float* input, const float* weight,
    const float* scale, const float* bias,
    float* membrane, float* output, float* im2col,
    int B, int H, int W, int C_in, int F, int T,
    int KH, int KW, int pad, int stride,
    int neuron, float v_threshold, float v_reset, float recip_tau);
void sengine_sgemm(int transB, int M, int N, int K,
                   const float* A, int lda, const float* B, int ldb,
                   float* C, int ldc);

void native_softmax(const float* input, float* output,
                    int outer, int inner);

#ifdef __cplusplus
}
#endif

#endif /* SENGINE_CPU_NATIVE_KERNELS_H */
