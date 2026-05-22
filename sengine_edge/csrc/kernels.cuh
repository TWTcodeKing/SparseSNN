/**
 * Native CUDA kernels for sengine.
 *
 * Shared between cpp_executor.cu (Python ctypes path) and
 * sengine_exec.cu (standalone binary). All kernels are NHWC FP16
 * unless noted otherwise.
 */

#ifndef SENGINE_KERNELS_CUH
#define SENGINE_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

// ─── Neuron kernels ───

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

// ─── Element-wise kernels ───

__global__ void add_fp16_kernel(const half* a, const half* b, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void zero_fp32_kernel(float* ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}

// ─── Pool kernels (NHWC) ───

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

// ─── Temporal & layout kernels ───

__global__ void temporal_mean_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int T, int spatial_elems
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        sum += __half2float(input[t * spatial_elems + s]);
    }
    output[s] = __float2half(sum / (float)T);
}

__global__ void layout_transpose_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int N, int H, int W, int C, int direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    if (direction == 0) {
        int c = idx % C;
        int w = (idx / C) % W;
        int h = (idx / C / W) % H;
        int n = idx / (C * W * H);
        output[n*C*H*W + c*H*W + h*W + w] = input[idx];
    } else {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / W / H) % C;
        int n = idx / (W * H * C);
        output[n*H*W*C + h*W*C + w*C + c] = input[idx];
    }
}

// ─── Naive Conv2d+BN (NHWC, for stem/grouped) ───

__global__ void naive_conv2d_bn_nhwc_kernel(
    const half* __restrict__ input, const half* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    half* __restrict__ output,
    int N, int H, int W, int C_in, int C_out,
    int KH, int KW, int stride, int pad, int OH, int OW, int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW * C_out;
    if (idx >= total) return;
    int co = idx % C_out;
    int rem = idx / C_out;
    int ow = rem % OW;
    rem = rem / OW;
    int oh = rem % OH;
    int n = rem / OH;
    int C_in_per_g = C_in / groups;
    int C_out_per_g = C_out / groups;
    int g = co / C_out_per_g;
    int ci_start = g * C_in_per_g;
    float sum = 0.0f;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * stride - pad + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * stride - pad + kw;
            if (iw < 0 || iw >= W) continue;
            for (int ci_local = 0; ci_local < C_in_per_g; ci_local++) {
                int ci = ci_start + ci_local;
                float iv = __half2float(input[((n * H + ih) * W + iw) * C_in + ci]);
                float wv = __half2float(weight[((kh * KW + kw) * C_in_per_g + ci_local) * C_out + co]);
                sum += iv * wv;
            }
        }
    }
    sum = sum * bn_scale[co] + bn_bias[co];
    output[idx] = __float2half(sum);
}

// ─── FP16 GEMM (naive, for reference) ───

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

// ─── Attention helper kernels ───

__global__ void scale_fp16_kernel(half* data, int n, half scale_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = __hmul(data[i], scale_val);
}

__global__ void mul_fp16_kernel(const half* a, const half* b, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

__global__ void scale_tensor_broadcast_kernel(
    half* data, const half* scale, int total, int heads, int inner
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = (i / inner) % heads;
    data[i] = __float2half(__half2float(data[i]) * __half2float(scale[h]));
}

__global__ void scale_nhwc_broadcast_kernel(
    half* data, const half* scale, int total, int C, int hd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i % C;
    int h = col / hd;
    data[i] = __float2half(__half2float(data[i]) * __half2float(scale[h]));
}

__global__ void reduce_sum_head_dim_kernel(
    const half* input, half* output, int outer, int head_dim, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = outer * N;
    if (idx >= total_out) return;
    int n_idx = idx % N;
    int o_idx = idx / N;
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += __half2float(input[(o_idx * head_dim + d) * N + n_idx]);
    }
    output[o_idx * N + n_idx] = __float2half(sum);
}

#endif // SENGINE_KERNELS_CUH
