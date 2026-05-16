/**
 * Standalone C++ CUDA Executor for sengine.
 *
 * ZERO dependency on PyTorch or TVM. Only needs:
 *   - CUDA runtime (kernels, streams, graphs)
 *   - dlopen/dlsym (load TileLang standalone .so kernels)
 *
 * TileLang kernels are exported as standalone .so with call() wrapper.
 * IF/LIF/Add/Pool/Gemm/Mean kernels are native CUDA.
 * CUDA Graph for zero-overhead replay.
 *
 * Exposed to Python via ctypes (no pybind11, no torch dependency).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <vector>
#include <string>

// ─── Native CUDA Kernels ───

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

__global__ void add_fp16_kernel(const half* a, const half* b, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

__global__ void zero_fp32_kernel(float* ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = 0.0f;
}

// MaxPool2d NHWC: (N, H, W, C) → (N, OH, OW, C)
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

// Global Average Pool NHWC: (N, H, W, C) → (N, 1, 1, C)
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

// Temporal Mean: (TB, ...) → reshape (T, B, ...) → mean over T → (B, ...)
__global__ void temporal_mean_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int T, int spatial_elems  // spatial_elems = B * rest
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        sum += __half2float(input[t * spatial_elems + s]);
    }
    output[s] = __float2half(sum / (float)T);
}

// ─── FP32 variants of native kernels (for precision="fp32" mode) ───

__global__ void if_neuron_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ membrane,
    float* __restrict__ spikes, int total_elems, int spatial_elems, float v_threshold
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float v = membrane[s];
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        float h = v + input[g];
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g] = spike;
    }
    membrane[s] = v;
}

__global__ void lif_neuron_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ membrane,
    float* __restrict__ spikes, int total_elems, int spatial_elems,
    float v_threshold, float recip_tau
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float decay = 1.0f - recip_tau;
    float v = membrane[s];
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        float h = decay * v + recip_tau * input[g];
        float spike = (h >= v_threshold) ? 1.0f : 0.0f;
        v = (1.0f - spike) * h;
        spikes[g] = spike;
    }
    membrane[s] = v;
}

__global__ void add_fp32_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void maxpool2d_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
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
            float v = input[((n * H + ih) * W + iw) * C + c];
            if (v > max_val) max_val = v;
        }
    }
    output[idx] = max_val;
}

__global__ void global_avgpool_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
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
        sum += input[(n * hw + i) * C + c];
    }
    output[idx] = sum / (float)hw;
}

__global__ void temporal_mean_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int T, int spatial_elems
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        sum += input[t * spatial_elems + s];
    }
    output[s] = sum / (float)T;
}

__global__ void layout_transpose_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int D1, int D2, int D3, int direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D1 * D2 * D3;
    if (idx >= total) return;
    int d3 = idx % D3;
    int rem = idx / D3;
    int d2 = rem % D2;
    rem = rem / D2;
    int d1 = rem % D1;
    int n = rem / D1;
    if (direction == 0) {
        // (N, D1, D2, D3) → (N, D2, D3, D1)  [NCHW→NHWC: D1=C, D2=H, D3=W]
        output[((n * D2 + d2) * D3 + d3) * D1 + d1] = input[idx];
    } else {
        // (N, D1, D2, D3) → (N, D3, D1, D2)  [NHWC→NCHW: D1=H, D2=W, D3=C]
        output[((n * D3 + d3) * D1 + d1) * D2 + d2] = input[idx];
    }
}

__global__ void scale_tensor_broadcast_fp32_kernel(
    float* data, const float* scale, int total, int heads, int inner
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = (i / inner) % heads;
    data[i] = data[i] * scale[h];
}

__global__ void scale_nhwc_broadcast_fp32_kernel(
    float* data, const float* scale, int total, int C, int hd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i % C;
    int h = col / hd;
    data[i] = data[i] * scale[h];
}

// FP16 GEMM: output = input @ weight^T  (row-major)
// input: (M, K), weight: (N, K), output: (M, N)
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

// Naive Conv2d+BN NHWC: for stem/grouped Conv (not perf-critical)
// Supports groups (g=1 for standard, g>1 for grouped conv).
// input:  (N, H, W, C_in) NHWC FP16
// weight: (KH, KW, C_in/g, C_out) NHWC FP16
// output: (N, OH, OW, C_out) NHWC FP16
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
    int g = co / C_out_per_g;           // which group this output channel belongs to
    int ci_start = g * C_in_per_g;      // input channel range for this group

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
    // BN epilogue
    sum = sum * bn_scale[co] + bn_bias[co];
    output[idx] = __float2half(sum);
}

__global__ void naive_conv2d_bn_fp32_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bn_scale, const float* __restrict__ bn_bias,
    float* __restrict__ output,
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
                sum += input[((n * H + ih) * W + iw) * C_in + ci]
                     * weight[((kh * KW + kw) * C_in_per_g + ci_local) * C_out + co];
            }
        }
    }
    sum = sum * bn_scale[co] + bn_bias[co];
    output[idx] = sum;
}

// Layout Transpose: NHWC↔NCHW
// direction=0: NHWC(N,H,W,C) → NCHW(N,C,H,W)
// direction=1: NCHW(N,C,H,W) → NHWC(N,H,W,C)
__global__ void layout_transpose_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int N, int H, int W, int C, int direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    if (direction == 0) {
        // NHWC → NCHW: input[n,h,w,c] → output[n,c,h,w]
        int c = idx % C;
        int w = (idx / C) % W;
        int h = (idx / C / W) % H;
        int n = idx / (C * W * H);
        output[n*C*H*W + c*H*W + h*W + w] = input[idx];
    } else {
        // NCHW → NHWC: input[n,c,h,w] → output[n,h,w,c]
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / W / H) % C;
        int n = idx / (W * H * C);
        output[n*H*W*C + h*W*C + w*C + c] = input[idx];
    }
}

// ─── Fused attention helper kernels ───

// Element-wise scale: data[i] *= scale
__global__ void scale_fp16_kernel(half* data, int n, half scale_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = __hmul(data[i], scale_val);
}

// Element-wise multiply: out[i] = a[i] * b[i]
__global__ void mul_fp16_kernel(const half* a, const half* b, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

// Broadcast scale tensor: data shape (TB, heads, D1, D2), scale shape (1, heads, 1, 1)
// data[tb, h, d1, d2] *= scale[h]
__global__ void scale_tensor_broadcast_kernel(
    half* data, const half* scale, int total, int heads, int inner
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int h = (i / inner) % heads;
    data[i] = __float2half(__half2float(data[i]) * __half2float(scale[h]));
}

// NHWC scale broadcast: data shape (rows, C) where C = heads*hd, scale shape (heads,)
// data[r, head*hd + d] *= scale[head]
__global__ void scale_nhwc_broadcast_kernel(
    half* data, const half* scale, int total, int C, int hd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i % C;
    int h = col / hd;
    data[i] = __float2half(__half2float(data[i]) * __half2float(scale[h]));
}

// ReduceSum over dim=2 (head_dim): (TB*heads, head_dim, N) → (TB*heads, 1, N)
// For TokenQK attention: sum Q over head_dim
__global__ void reduce_sum_head_dim_kernel(
    const half* input, half* output, int outer, int head_dim, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = outer * N;
    if (idx >= total_out) return;
    int n_idx = idx % N;
    int o_idx = idx / N;  // TB*heads index
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        sum += __half2float(input[(o_idx * head_dim + d) * N + n_idx]);
    }
    output[o_idx * N + n_idx] = __float2half(sum);
}

// ─── BN epilogue (channel-wise scale+bias on NHWC tensor) ───
__global__ void bn_epilogue_fp16_kernel(
    half* __restrict__ data, const float* __restrict__ scale,
    const float* __restrict__ bias, int total, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    float val = __half2float(data[idx]) * scale[c] + bias[c];
    data[idx] = __float2half(val);
}

__global__ void bn_epilogue_fp32_kernel(
    float* __restrict__ data, const float* __restrict__ scale,
    const float* __restrict__ bias, int total, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    data[idx] = data[idx] * scale[c] + bias[c];
}

// ─── Detection model kernels ───

// Nearest-neighbor upsample (NHWC): (N,H,W,C) → (N,OH,OW,C)
__global__ void nearest_upsample_nhwc_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int N, int H, int W, int C, int OH, int OW, int scale_h, int scale_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW * C;
    if (idx >= total) return;
    int c  = idx % C;
    int ow = (idx / C) % OW;
    int oh = (idx / (C * OW)) % OH;
    int n  = idx / (C * OW * OH);
    int ih = oh / scale_h;
    int iw = ow / scale_w;
    output[idx] = input[((n * H + ih) * W + iw) * C + c];
}

__global__ void nearest_upsample_nhwc_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int N, int H, int W, int C, int OH, int OW, int scale_h, int scale_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW * C;
    if (idx >= total) return;
    int c  = idx % C;
    int ow = (idx / C) % OW;
    int oh = (idx / (C * OW)) % OH;
    int n  = idx / (C * OW * OH);
    int ih = oh / scale_h;
    int iw = ow / scale_w;
    output[idx] = input[((n * H + ih) * W + iw) * C + c];
}

// Channel concat (NHWC): (N,H,W,Ca) + (N,H,W,Cb) → (N,H,W,Ca+Cb)
__global__ void concat_nhwc_kernel(
    const half* __restrict__ a, const half* __restrict__ b,
    half* __restrict__ output,
    int N, int H, int W, int Ca, int Cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C_out = Ca + Cb;
    int total = N * H * W * C_out;
    if (idx >= total) return;
    int c   = idx % C_out;
    int hw  = (idx / C_out);
    if (c < Ca)
        output[idx] = a[hw * Ca + c];
    else
        output[idx] = b[hw * Cb + (c - Ca)];
}

__global__ void concat_nhwc_fp32_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output,
    int N, int H, int W, int Ca, int Cb
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C_out = Ca + Cb;
    int total = N * H * W * C_out;
    if (idx >= total) return;
    int c   = idx % C_out;
    int hw  = (idx / C_out);
    if (c < Ca)
        output[idx] = a[hw * Ca + c];
    else
        output[idx] = b[hw * Cb + (c - Ca)];
}

// I-LIF neuron: mem = decay*(mem-spike) + x; spike = round(clamp(mem,0,max_level))
__global__ void ilif_neuron_kernel(
    const half* __restrict__ input, float* __restrict__ membrane,
    half* __restrict__ spikes, int total_elems, int spatial_elems,
    float decay, float max_level
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float v = membrane[s];
    float spike = 0.0f;
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        v = decay * (v - spike) + __half2float(input[g]);
        spike = roundf(fminf(fmaxf(v, 0.0f), max_level));
        spikes[g] = __float2half(spike);
    }
    membrane[s] = v;
}

__global__ void ilif_neuron_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ membrane,
    float* __restrict__ spikes, int total_elems, int spatial_elems,
    float decay, float max_level
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial_elems) return;
    int T = total_elems / spatial_elems;
    float v = membrane[s];
    float spike = 0.0f;
    for (int t = 0; t < T; t++) {
        int g = t * spatial_elems + s;
        v = decay * (v - spike) + input[g];
        spike = roundf(fminf(fmaxf(v, 0.0f), max_level));
        spikes[g] = spike;
    }
    membrane[s] = v;
}

// Softmax along inner dimension: input(outer, inner) → output(outer, inner)
__global__ void softmax_fp16_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int outer, int inner
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer) return;
    float max_val = -1e30f;
    for (int i = 0; i < inner; i++)
        max_val = fmaxf(max_val, __half2float(input[o * inner + i]));
    float sum = 0.0f;
    for (int i = 0; i < inner; i++)
        sum += expf(__half2float(input[o * inner + i]) - max_val);
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < inner; i++)
        output[o * inner + i] = __float2half(
            expf(__half2float(input[o * inner + i]) - max_val) * inv_sum);
}

__global__ void softmax_fp32_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int outer, int inner
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer) return;
    float max_val = -1e30f;
    for (int i = 0; i < inner; i++)
        max_val = fmaxf(max_val, input[o * inner + i]);
    float sum = 0.0f;
    for (int i = 0; i < inner; i++)
        sum += expf(input[o * inner + i] - max_val);
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < inner; i++)
        output[o * inner + i] = expf(input[o * inner + i] - max_val) * inv_sum;
}

// ─── Kernel types ───
enum KernelType {
    KT_TILELANG = 0, KT_IF = 1, KT_LIF = 2, KT_ADD = 3, KT_SKIP = 4,
    KT_MAXPOOL = 5, KT_GLOBAL_AVGPOOL = 6, KT_TEMPORAL_MEAN = 7,
    KT_GEMM = 8, KT_ALIAS = 9, KT_LAYOUT_TRANSPOSE = 10,
    KT_TILELANG_3 = 11,  // 3-arg TileLang (MatMul: A, B, output)
    KT_NAIVE_CONV = 12,  // Naive Conv2d+BN for stem (C_in=3)
    KT_FUSED_ATTN = 13,  // Fused attention (cuBLAS batched GEMM + LIF)
    KT_RESIZE = 14,      // Nearest-neighbor upsample (FPN)
    KT_CONCAT = 15,      // Channel concat (FPN/PANet)
    KT_ILIF = 16,        // Integer LIF neuron (spike_yolo)
    KT_SOFTMAX = 17,     // Softmax (DFL detection head)
    KT_CUDNN_CONV = 18   // cuDNN Conv2d+BN (for validator-REVERT'd large 3×3 convs)
};

// ─── TileLang standalone kernel (loaded via dlopen) ───
struct TileLangKernel {
    void* dl_handle;
    // int init() — one-time setup (dynamic smem)
    typedef int (*InitFn)();
    InitFn init_fn;
    // int call(ptr0, ptr1, ..., ptrN, cudaStream_t) — kernel launch
    // We store it as a generic function pointer; actual calling uses per-node info
    void* call_fn;
    int initialized;
};

// ─── Node descriptor ───
struct NodeDesc {
    KernelType type;
    int tilelang_idx;         // index into tilelang_kernels

    // TileLang call convention: array of void* arg pointers + stream
    // Stored as raw GPU pointers in arg order matching call() signature
    void** tl_args;           // array of GPU pointers (in call() arg order)
    int tl_n_args;            // number of pointer args (excl. stream)

    // For neuron kernels
    half* input_ptr;
    half* output_ptr;
    float* membrane_ptr;
    int total_elems, spatial_elems;
    float v_threshold, recip_tau;

    // For add kernel
    half* add_a_ptr;
    half* add_b_ptr;
    half* add_out_ptr;
    int add_n;

    // For maxpool
    half* pool_in;
    half* pool_out;
    int pool_N, pool_H, pool_W, pool_C, pool_OH, pool_OW;
    int pool_ks, pool_stride, pool_pad;

    // For global avgpool
    half* gavg_in;
    half* gavg_out;
    int gavg_N, gavg_H, gavg_W, gavg_C;

    // For temporal mean
    half* tmean_in;
    half* tmean_out;
    int tmean_T, tmean_spatial;

    // For GEMM: output = input @ weight^T
    half* gemm_in;
    half* gemm_w;
    half* gemm_out;
    int gemm_M, gemm_K, gemm_N;

    // For alias (zero-cost: reshape/transpose — just pointer copy)
    half** alias_src;
    half** alias_dst;

    // For layout transpose (NHWC↔NCHW)
    half* lt_in;
    half* lt_out;
    int lt_N, lt_H, lt_W, lt_C, lt_direction;

    // For naive conv2d+BN (stem)
    half* nc_in;
    half* nc_w;
    float* nc_sc;
    float* nc_bi;
    half* nc_out;
    int nc_N, nc_H, nc_W, nc_Cin, nc_Cout, nc_KH, nc_KW, nc_stride, nc_pad, nc_OH, nc_OW, nc_groups;

    // For fused attention (KT_FUSED_ATTN)
    // variant: 0=spikformer, 1=maxformer, 2=dssa, 3=token_qk
    int fa_variant;
    int fa_tl_gemm1_idx;         // TileLang .so index for GEMM1
    int fa_tl_gemm2_idx;         // TileLang .so index for GEMM2
    half *fa_q, *fa_k, *fa_v, *fa_out;
    half *fa_workspace;          // scratch (attn scores, permuted buffers)
    float *fa_membrane;
    half *fa_scale1_ptr, *fa_scale2_ptr;  // DSSA per-head tensor scales
    int fa_TB, fa_heads, fa_hd, fa_N;
    int fa_H, fa_W;              // spatial dims for NHWC↔NCHW
    int fa_spatial_kv;           // LIF membrane spatial size
    float fa_v_thresh, fa_recip_tau;
    int fa_needs_permute;        // Whether NHWC↔NCHW needed (maxformer/dssa)
    int fa_lif_total, fa_lif_spatial;  // LIF kernel dims
    // Workspace offsets (in half elements)
    int fa_ws_gemm1_out;         // GEMM1 output / GEMM2 input
    int fa_ws_perm_q, fa_ws_perm_k, fa_ws_perm_v;  // permuted Q/K/V

    // For nearest-neighbor upsample (KT_RESIZE)
    void *rs_in, *rs_out;
    int rs_N, rs_H, rs_W, rs_C, rs_OH, rs_OW, rs_scale_h, rs_scale_w;

    // For channel concat (KT_CONCAT)
    void *cat_a, *cat_b, *cat_out;
    int cat_NHW, cat_Ca, cat_Cb;

    // For I-LIF neuron (KT_ILIF)
    float ilif_decay, ilif_max_level;

    // For softmax (KT_SOFTMAX)
    void *sm_in, *sm_out;
    int sm_outer, sm_inner;

    // For cuDNN Conv2d+BN (KT_CUDNN_CONV)
    cudnnTensorDescriptor_t cudnn_in_desc, cudnn_out_desc;
    cudnnFilterDescriptor_t cudnn_filt_desc;
    cudnnConvolutionDescriptor_t cudnn_conv_desc;
    cudnnConvolutionFwdAlgo_t cudnn_algo;
    void *cudnn_workspace;
    size_t cudnn_ws_size;
    int cudnn_initialized;
};

// ─── Executor ───
struct SEngineExecutor {
    int* schedule;
    int schedule_len;
    NodeDesc* nodes;
    int max_node_id;
    TileLangKernel* tl_kernels;
    int n_tl_kernels;
    float** membranes;
    int* membrane_sizes;
    int n_membranes;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    int captured;
    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    int is_fp32;  // global precision flag: 0=fp16, 1=fp32
};

// ─── C API (exposed via ctypes) ───
extern "C" {

SEngineExecutor* sengine_create() {
    auto* e = new SEngineExecutor();
    memset(e, 0, sizeof(SEngineExecutor));
    cudaStreamCreate(&e->stream);
    cublasCreate(&e->cublas);
    e->cudnn = nullptr;  // lazy init — created on first cuDNN conv node
    cublasSetStream(e->cublas, e->stream);
    cublasSetMathMode(e->cublas, CUBLAS_TENSOR_OP_MATH);
    return e;
}

void sengine_set_fp32(SEngineExecutor* e, int fp32) {
    e->is_fp32 = fp32;
}

void sengine_destroy(SEngineExecutor* e) {
    if (!e) return;
    if (e->exec) cudaGraphExecDestroy(e->exec);
    if (e->graph) cudaGraphDestroy(e->graph);
    if (e->cublas) cublasDestroy(e->cublas);
    cudaStreamDestroy(e->stream);
    for (int i = 0; i < e->n_tl_kernels; i++) {
        if (e->tl_kernels[i].dl_handle) dlclose(e->tl_kernels[i].dl_handle);
    }
    free(e->tl_kernels);
    free(e->schedule);
    if (e->nodes) {
        for (int i = 0; i <= e->max_node_id; i++) {
            free(e->nodes[i].tl_args);
        }
        free(e->nodes);
    }
    free(e->membranes);
    free(e->membrane_sizes);
    delete e;
}

int sengine_load_tilelang(SEngineExecutor* e, const char* so_path) {
    void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return -1;
    }
    auto init_fn = (TileLangKernel::InitFn)dlsym(handle, "init");
    void* call_fn = dlsym(handle, "call");
    if (!call_fn) {
        fprintf(stderr, "No call() in %s\n", so_path);
        dlclose(handle);
        return -1;
    }
    // Run init() to set up dynamic shared memory
    if (init_fn) {
        int ret = init_fn();
        if (ret != 0) {
            fprintf(stderr, "init() failed in %s\n", so_path);
            dlclose(handle);
            return -1;
        }
    }
    int idx = e->n_tl_kernels;
    e->n_tl_kernels++;
    e->tl_kernels = (TileLangKernel*)realloc(e->tl_kernels,
                                              e->n_tl_kernels * sizeof(TileLangKernel));
    e->tl_kernels[idx].dl_handle = handle;
    e->tl_kernels[idx].init_fn = init_fn;
    e->tl_kernels[idx].call_fn = call_fn;
    e->tl_kernels[idx].initialized = 1;
    return idx;
}

void sengine_set_schedule(SEngineExecutor* e, int* sched, int len) {
    e->schedule = (int*)malloc(len * sizeof(int));
    memcpy(e->schedule, sched, len * sizeof(int));
    e->schedule_len = len;
}

void sengine_alloc_nodes(SEngineExecutor* e, int max_id) {
    e->max_node_id = max_id;
    e->nodes = (NodeDesc*)calloc(max_id + 1, sizeof(NodeDesc));
}

/// ─── TileLang MatMul node (3 args: A, B, output) ───
void sengine_set_tilelang_node_3(SEngineExecutor* e, int nid, int tl_idx,
                                  void* arg0, void* arg1, void* arg2) {
    auto& n = e->nodes[nid];
    n.type = KT_TILELANG_3;
    n.tilelang_idx = tl_idx;
    n.tl_n_args = 3;
    n.tl_args = (void**)malloc(3 * sizeof(void*));
    n.tl_args[0] = arg0;  // A
    n.tl_args[1] = arg1;  // B
    n.tl_args[2] = arg2;  // output
}

// ─── TileLang Conv+BN node (5 args: data, weight, scale, bias, output) ───
void sengine_set_tilelang_node_5(SEngineExecutor* e, int nid, int tl_idx,
                                  void* arg0, void* arg1, void* arg2, void* arg3, void* arg4) {
    auto& n = e->nodes[nid];
    n.type = KT_TILELANG;
    n.tilelang_idx = tl_idx;
    n.tl_n_args = 5;
    n.tl_args = (void**)malloc(5 * sizeof(void*));
    n.tl_args[0] = arg0;  // data
    n.tl_args[1] = arg1;  // weight
    n.tl_args[2] = arg2;  // bn_scale
    n.tl_args[3] = arg3;  // bn_bias
    n.tl_args[4] = arg4;  // output
}

// ─── TileLang fused Conv+BN+IF node (6 args: data, weight, membrane, scale, bias, spikes) ───
void sengine_set_tilelang_node_6(SEngineExecutor* e, int nid, int tl_idx,
                                  void* arg0, void* arg1, void* arg2,
                                  void* arg3, void* arg4, void* arg5) {
    auto& n = e->nodes[nid];
    n.type = KT_TILELANG;
    n.tilelang_idx = tl_idx;
    n.tl_n_args = 6;
    n.tl_args = (void**)malloc(6 * sizeof(void*));
    n.tl_args[0] = arg0;
    n.tl_args[1] = arg1;
    n.tl_args[2] = arg2;
    n.tl_args[3] = arg3;
    n.tl_args[4] = arg4;
    n.tl_args[5] = arg5;
}

void sengine_set_if_node(SEngineExecutor* e, int nid,
                          half* input, half* output, float* membrane,
                          int total_elems, int spatial_elems, float v_threshold) {
    auto& n = e->nodes[nid];
    n.type = KT_IF;
    n.input_ptr = input;
    n.output_ptr = output;
    n.membrane_ptr = membrane;
    n.total_elems = total_elems;
    n.spatial_elems = spatial_elems;
    n.v_threshold = v_threshold;
}

void sengine_set_lif_node(SEngineExecutor* e, int nid,
                           half* input, half* output, float* membrane,
                           int total, int spatial, float v_thr, float recip_tau) {
    auto& n = e->nodes[nid];
    n.type = KT_LIF;
    n.input_ptr = input;
    n.output_ptr = output;
    n.membrane_ptr = membrane;
    n.total_elems = total;
    n.spatial_elems = spatial;
    n.v_threshold = v_thr;
    n.recip_tau = recip_tau;
}

void sengine_set_add_node(SEngineExecutor* e, int nid,
                           half* a, half* b, half* out, int n) {
    auto& nd = e->nodes[nid];
    nd.type = KT_ADD;
    nd.add_a_ptr = a;
    nd.add_b_ptr = b;
    nd.add_out_ptr = out;
    nd.add_n = n;
}

void sengine_set_maxpool_node(SEngineExecutor* e, int nid,
                               half* input, half* output,
                               int N, int H, int W, int C,
                               int OH, int OW, int ks, int stride, int pad) {
    auto& nd = e->nodes[nid];
    nd.type = KT_MAXPOOL;
    nd.pool_in = input;
    nd.pool_out = output;
    nd.pool_N = N; nd.pool_H = H; nd.pool_W = W; nd.pool_C = C;
    nd.pool_OH = OH; nd.pool_OW = OW;
    nd.pool_ks = ks; nd.pool_stride = stride; nd.pool_pad = pad;
}

void sengine_set_global_avgpool_node(SEngineExecutor* e, int nid,
                                      half* input, half* output,
                                      int N, int H, int W, int C) {
    auto& nd = e->nodes[nid];
    nd.type = KT_GLOBAL_AVGPOOL;
    nd.gavg_in = input;
    nd.gavg_out = output;
    nd.gavg_N = N; nd.gavg_H = H; nd.gavg_W = W; nd.gavg_C = C;
}

void sengine_set_temporal_mean_node(SEngineExecutor* e, int nid,
                                     half* input, half* output,
                                     int T, int spatial_elems) {
    auto& nd = e->nodes[nid];
    nd.type = KT_TEMPORAL_MEAN;
    nd.tmean_in = input;
    nd.tmean_out = output;
    nd.tmean_T = T;
    nd.tmean_spatial = spatial_elems;
}

void sengine_set_gemm_node(SEngineExecutor* e, int nid,
                             half* input, half* weight, half* output,
                             int M, int K, int N) {
    auto& nd = e->nodes[nid];
    nd.type = KT_GEMM;
    nd.gemm_in = input;
    nd.gemm_w = weight;
    nd.gemm_out = output;
    nd.gemm_M = M; nd.gemm_K = K; nd.gemm_N = N;
}

void sengine_set_fused_attn_node(SEngineExecutor* e, int nid,
    int variant, int gemm1_idx, int gemm2_idx,
    half* q, half* k, half* v, half* out,
    half* workspace, float* membrane,
    int TB, int heads, int hd, int N, int H, int W,
    int lif_total, int lif_spatial,
    float v_thresh, float recip_tau,
    int needs_permute,
    half* scale1_ptr, half* scale2_ptr,
    int ws_gemm1_out, int ws_perm_q, int ws_perm_k, int ws_perm_v)
{
    auto& nd = e->nodes[nid];
    nd.type = KT_FUSED_ATTN;
    nd.fa_variant = variant;
    nd.fa_tl_gemm1_idx = gemm1_idx;
    nd.fa_tl_gemm2_idx = gemm2_idx;
    nd.fa_q = q; nd.fa_k = k; nd.fa_v = v; nd.fa_out = out;
    nd.fa_workspace = workspace;
    nd.fa_membrane = membrane;
    nd.fa_scale1_ptr = scale1_ptr; nd.fa_scale2_ptr = scale2_ptr;
    nd.fa_TB = TB; nd.fa_heads = heads; nd.fa_hd = hd; nd.fa_N = N;
    nd.fa_H = H; nd.fa_W = W;
    nd.fa_lif_total = lif_total; nd.fa_lif_spatial = lif_spatial;
    nd.fa_v_thresh = v_thresh; nd.fa_recip_tau = recip_tau;
    nd.fa_needs_permute = needs_permute;
    nd.fa_ws_gemm1_out = ws_gemm1_out;
    nd.fa_ws_perm_q = ws_perm_q;
    nd.fa_ws_perm_k = ws_perm_k;
    nd.fa_ws_perm_v = ws_perm_v;
}

void sengine_set_skip_node(SEngineExecutor* e, int nid) {
    e->nodes[nid].type = KT_SKIP;
}

void sengine_set_alias_node(SEngineExecutor* e, int nid,
                              half* src, half* dst, int n_elems) {
    auto& nd = e->nodes[nid];
    nd.type = KT_ALIAS;
    nd.input_ptr = src;
    nd.output_ptr = dst;
    nd.total_elems = n_elems;
}

void sengine_set_naive_conv_node(SEngineExecutor* e, int nid,
                                 half* input, half* weight,
                                 float* bn_scale, float* bn_bias, half* output,
                                 int N, int H, int W, int Cin, int Cout,
                                 int KH, int KW, int stride, int pad, int OH, int OW,
                                 int groups) {
    auto& nd = e->nodes[nid];
    nd.type = KT_NAIVE_CONV;
    nd.nc_in = input; nd.nc_w = weight;
    nd.nc_sc = bn_scale; nd.nc_bi = bn_bias; nd.nc_out = output;
    nd.nc_N = N; nd.nc_H = H; nd.nc_W = W;
    nd.nc_Cin = Cin; nd.nc_Cout = Cout;
    nd.nc_KH = KH; nd.nc_KW = KW;
    nd.nc_stride = stride; nd.nc_pad = pad;
    nd.nc_OH = OH; nd.nc_OW = OW; nd.nc_groups = groups;
}

void sengine_set_layout_transpose_node(SEngineExecutor* e, int nid,
                                        half* input, half* output,
                                        int N, int H, int W, int C, int direction) {
    auto& nd = e->nodes[nid];
    nd.type = KT_LAYOUT_TRANSPOSE;
    nd.lt_in = input;
    nd.lt_out = output;
    nd.lt_N = N; nd.lt_H = H; nd.lt_W = W; nd.lt_C = C;
    nd.lt_direction = direction;
}

// ─── Detection model node setters ───

void sengine_set_resize_node(SEngineExecutor* e, int nid,
                              void* input, void* output,
                              int N, int H, int W, int C,
                              int OH, int OW, int scale_h, int scale_w) {
    auto& nd = e->nodes[nid];
    nd.type = KT_RESIZE;
    nd.rs_in = input; nd.rs_out = output;
    nd.rs_N = N; nd.rs_H = H; nd.rs_W = W; nd.rs_C = C;
    nd.rs_OH = OH; nd.rs_OW = OW;
    nd.rs_scale_h = scale_h; nd.rs_scale_w = scale_w;
}

void sengine_set_concat_node(SEngineExecutor* e, int nid,
                              void* a, void* b, void* output,
                              int NHW, int Ca, int Cb) {
    auto& nd = e->nodes[nid];
    nd.type = KT_CONCAT;
    nd.cat_a = a; nd.cat_b = b; nd.cat_out = output;
    nd.cat_NHW = NHW; nd.cat_Ca = Ca; nd.cat_Cb = Cb;
}

void sengine_set_ilif_node(SEngineExecutor* e, int nid,
                            void* input, void* output, float* membrane,
                            int total, int spatial, float decay, float max_level) {
    auto& nd = e->nodes[nid];
    nd.type = KT_ILIF;
    nd.input_ptr = (half*)input;
    nd.output_ptr = (half*)output;
    nd.membrane_ptr = membrane;
    nd.total_elems = total;
    nd.spatial_elems = spatial;
    nd.ilif_decay = decay;
    nd.ilif_max_level = max_level;
}

void sengine_set_softmax_node(SEngineExecutor* e, int nid,
                               void* input, void* output,
                               int outer, int inner) {
    auto& nd = e->nodes[nid];
    nd.type = KT_SOFTMAX;
    nd.sm_in = input; nd.sm_out = output;
    nd.sm_outer = outer; nd.sm_inner = inner;
}

// ─── cuDNN Conv2d+BN node (for validator-REVERT'd large convolutions) ───
void sengine_set_cudnn_conv_node(SEngineExecutor* e, int nid,
                                  void* input, void* weight,
                                  float* bn_scale, float* bn_bias, void* output,
                                  int N, int H, int W, int C_in, int C_out,
                                  int KH, int KW, int stride, int pad,
                                  int OH, int OW, int groups) {
    auto& nd = e->nodes[nid];
    nd.type = KT_CUDNN_CONV;
    // Reuse naive_conv fields for pointers and params
    nd.nc_in = (half*)input; nd.nc_w = (half*)weight;
    nd.nc_sc = bn_scale; nd.nc_bi = bn_bias;
    nd.nc_out = (half*)output;
    nd.nc_N = N; nd.nc_H = H; nd.nc_W = W;
    nd.nc_Cin = C_in; nd.nc_Cout = C_out;
    nd.nc_KH = KH; nd.nc_KW = KW;
    nd.nc_stride = stride; nd.nc_pad = pad;
    nd.nc_OH = OH; nd.nc_OW = OW;
    nd.nc_groups = groups > 0 ? groups : 1;

    // Lazy-init cuDNN handle on first use
    if (!e->cudnn) {
        cudnnCreate(&e->cudnn);
        cudnnSetStream(e->cudnn, e->stream);
    }

    // Create cuDNN descriptors
    cudnnDataType_t dt = e->is_fp32 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;

    cudnnCreateTensorDescriptor(&nd.cudnn_in_desc);
    cudnnSetTensor4dDescriptor(nd.cudnn_in_desc, CUDNN_TENSOR_NHWC, dt, N, C_in, H, W);

    cudnnCreateTensorDescriptor(&nd.cudnn_out_desc);
    cudnnSetTensor4dDescriptor(nd.cudnn_out_desc, CUDNN_TENSOR_NHWC, dt, N, C_out, OH, OW);

    cudnnCreateFilterDescriptor(&nd.cudnn_filt_desc);
    cudnnSetFilter4dDescriptor(nd.cudnn_filt_desc, dt, CUDNN_TENSOR_NCHW,
                                C_out, C_in / nd.nc_groups, KH, KW);

    cudnnCreateConvolutionDescriptor(&nd.cudnn_conv_desc);
    cudnnSetConvolution2dDescriptor(nd.cudnn_conv_desc, pad, pad, stride, stride, 1, 1,
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    if (nd.nc_groups > 1)
        cudnnSetConvolutionGroupCount(nd.cudnn_conv_desc, nd.nc_groups);
    cudnnSetConvolutionMathType(nd.cudnn_conv_desc, CUDNN_TENSOR_OP_MATH);

    // Get best algorithm via heuristic (instant, no GPU benchmarking)
    int n_algos = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    cudnnSetStream(e->cudnn, e->stream);
    cudnnGetConvolutionForwardAlgorithm_v7(e->cudnn,
        nd.cudnn_in_desc, nd.cudnn_filt_desc, nd.cudnn_conv_desc, nd.cudnn_out_desc,
        8, &n_algos, perf);
    nd.cudnn_algo = (n_algos > 0) ? perf[0].algo : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Get workspace size
    cudnnGetConvolutionForwardWorkspaceSize(e->cudnn,
        nd.cudnn_in_desc, nd.cudnn_filt_desc, nd.cudnn_conv_desc, nd.cudnn_out_desc,
        nd.cudnn_algo, &nd.cudnn_ws_size);
    if (nd.cudnn_ws_size > 0)
        cudaMalloc(&nd.cudnn_workspace, nd.cudnn_ws_size);
    else
        nd.cudnn_workspace = nullptr;
    nd.cudnn_initialized = 1;
}

void sengine_add_membrane(SEngineExecutor* e, float* ptr, int size) {
    int idx = e->n_membranes++;
    e->membranes = (float**)realloc(e->membranes, e->n_membranes * sizeof(float*));
    e->membrane_sizes = (int*)realloc(e->membrane_sizes, e->n_membranes * sizeof(int));
    e->membranes[idx] = ptr;
    e->membrane_sizes[idx] = size;
}

// ─── Execute schedule (tight C loop, no Python) ───
void sengine_execute(SEngineExecutor* e) {
    cudaStream_t s = e->stream;
    for (int i = 0; i < e->schedule_len; i++) {
        int nid = e->schedule[i];
        if (nid < 0 || nid > e->max_node_id) continue;
        auto& nd = e->nodes[nid];

        switch (nd.type) {
        case KT_TILELANG:
        case KT_TILELANG_3: {
            auto& tl = e->tl_kernels[nd.tilelang_idx];
            // call(arg0, arg1, ..., argN, stream)
            if (nd.tl_n_args == 3) {
                typedef int (*CallFn3)(void*, void*, void*, cudaStream_t);
                auto fn = (CallFn3)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2], s);
            } else if (nd.tl_n_args == 4) {
                typedef int (*CallFn4)(void*, void*, void*, void*, cudaStream_t);
                auto fn = (CallFn4)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2],
                   nd.tl_args[3], s);
            } else if (nd.tl_n_args == 5) {
                typedef int (*CallFn5)(void*, void*, void*, void*, void*, cudaStream_t);
                auto fn = (CallFn5)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2],
                   nd.tl_args[3], nd.tl_args[4], s);
            } else if (nd.tl_n_args == 6) {
                typedef int (*CallFn6)(void*, void*, void*, void*, void*, void*, cudaStream_t);
                auto fn = (CallFn6)tl.call_fn;
                fn(nd.tl_args[0], nd.tl_args[1], nd.tl_args[2],
                   nd.tl_args[3], nd.tl_args[4], nd.tl_args[5], s);
            }
            break;
        }
        case KT_IF: {
            int thr = 256, blk = (nd.spatial_elems + thr - 1) / thr;
            if (e->is_fp32)
                if_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.input_ptr, nd.membrane_ptr, (float*)nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.v_threshold);
            else
                if_neuron_kernel<<<blk, thr, 0, s>>>(
                    nd.input_ptr, nd.membrane_ptr, nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.v_threshold);
            break;
        }
        case KT_LIF: {
            int thr = 256, blk = (nd.spatial_elems + thr - 1) / thr;
            if (e->is_fp32)
                lif_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.input_ptr, nd.membrane_ptr, (float*)nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.v_threshold, nd.recip_tau);
            else
                lif_neuron_kernel<<<blk, thr, 0, s>>>(
                    nd.input_ptr, nd.membrane_ptr, nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.v_threshold, nd.recip_tau);
            break;
        }
        case KT_ADD: {
            int thr = 256, blk = (nd.add_n + thr - 1) / thr;
            if (e->is_fp32)
                add_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.add_a_ptr, (float*)nd.add_b_ptr, (float*)nd.add_out_ptr, nd.add_n);
            else
                add_fp16_kernel<<<blk, thr, 0, s>>>(
                    nd.add_a_ptr, nd.add_b_ptr, nd.add_out_ptr, nd.add_n);
            break;
        }
        case KT_MAXPOOL: {
            int total = nd.pool_N * nd.pool_OH * nd.pool_OW * nd.pool_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                maxpool2d_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.pool_in, (float*)nd.pool_out,
                    nd.pool_N, nd.pool_H, nd.pool_W, nd.pool_C,
                    nd.pool_OH, nd.pool_OW,
                    nd.pool_ks, nd.pool_stride, nd.pool_pad);
            else
                maxpool2d_nhwc_kernel<<<blk, thr, 0, s>>>(
                    nd.pool_in, nd.pool_out,
                    nd.pool_N, nd.pool_H, nd.pool_W, nd.pool_C,
                    nd.pool_OH, nd.pool_OW,
                    nd.pool_ks, nd.pool_stride, nd.pool_pad);
            break;
        }
        case KT_GLOBAL_AVGPOOL: {
            int total = nd.gavg_N * nd.gavg_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                global_avgpool_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.gavg_in, (float*)nd.gavg_out,
                    nd.gavg_N, nd.gavg_H, nd.gavg_W, nd.gavg_C);
            else
                global_avgpool_nhwc_kernel<<<blk, thr, 0, s>>>(
                    nd.gavg_in, nd.gavg_out,
                    nd.gavg_N, nd.gavg_H, nd.gavg_W, nd.gavg_C);
            break;
        }
        case KT_TEMPORAL_MEAN: {
            int thr = 256, blk = (nd.tmean_spatial + thr - 1) / thr;
            if (e->is_fp32)
                temporal_mean_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.tmean_in, (float*)nd.tmean_out, nd.tmean_T, nd.tmean_spatial);
            else
                temporal_mean_kernel<<<blk, thr, 0, s>>>(
                    nd.tmean_in, nd.tmean_out, nd.tmean_T, nd.tmean_spatial);
            break;
        }
        case KT_GEMM: {
            // cuBLAS GEMM: C = A @ B^T (row-major via column-major trick)
            if (e->is_fp32) {
                const float alpha_f = 1.0f, beta_f = 0.0f;
                cublasSgemm(e->cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    nd.gemm_N, nd.gemm_M, nd.gemm_K,
                    &alpha_f,
                    (float*)nd.gemm_w, nd.gemm_K,
                    (float*)nd.gemm_in, nd.gemm_K,
                    &beta_f,
                    (float*)nd.gemm_out, nd.gemm_N);
            } else {
                const half alpha_h = __float2half(1.0f);
                const half beta_h = __float2half(0.0f);
                cublasHgemm(e->cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    nd.gemm_N, nd.gemm_M, nd.gemm_K,
                    &alpha_h,
                    nd.gemm_w, nd.gemm_K,
                    nd.gemm_in, nd.gemm_K,
                    &beta_h,
                    nd.gemm_out, nd.gemm_N);
            }
            break;
        }
        case KT_NAIVE_CONV: {
            int total = nd.nc_N * nd.nc_OH * nd.nc_OW * nd.nc_Cout;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                naive_conv2d_bn_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.nc_in, (float*)nd.nc_w, nd.nc_sc, nd.nc_bi, (float*)nd.nc_out,
                    nd.nc_N, nd.nc_H, nd.nc_W, nd.nc_Cin, nd.nc_Cout,
                    nd.nc_KH, nd.nc_KW, nd.nc_stride, nd.nc_pad, nd.nc_OH, nd.nc_OW,
                    nd.nc_groups);
            else
                naive_conv2d_bn_nhwc_kernel<<<blk, thr, 0, s>>>(
                    nd.nc_in, nd.nc_w, nd.nc_sc, nd.nc_bi, nd.nc_out,
                    nd.nc_N, nd.nc_H, nd.nc_W, nd.nc_Cin, nd.nc_Cout,
                    nd.nc_KH, nd.nc_KW, nd.nc_stride, nd.nc_pad, nd.nc_OH, nd.nc_OW,
                    nd.nc_groups);
            break;
        }
        case KT_LAYOUT_TRANSPOSE: {
            int total = nd.lt_N * nd.lt_H * nd.lt_W * nd.lt_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                layout_transpose_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.lt_in, (float*)nd.lt_out,
                    nd.lt_N, nd.lt_H, nd.lt_W, nd.lt_C, nd.lt_direction);
            else
                layout_transpose_kernel<<<blk, thr, 0, s>>>(
                    nd.lt_in, nd.lt_out, nd.lt_N, nd.lt_H, nd.lt_W, nd.lt_C, nd.lt_direction);
            break;
        }
        case KT_ALIAS: {
            // Just memcpy (for reshape/transpose that need contiguous copy)
            if (nd.input_ptr != nd.output_ptr && nd.total_elems > 0) {
                int elem_size = e->is_fp32 ? sizeof(float) : sizeof(half);
                cudaMemcpyAsync(nd.output_ptr, nd.input_ptr,
                    nd.total_elems * elem_size, cudaMemcpyDeviceToDevice, s);
            }
            break;
        }
        case KT_FUSED_ATTN: {
            // ─── Fused Attention: TileLang batched GEMM .so + native LIF + permute ───
            typedef int (*CallFn3)(void*, void*, void*, cudaStream_t);
            auto& tl1 = e->tl_kernels[nd.fa_tl_gemm1_idx];
            auto& tl2 = e->tl_kernels[nd.fa_tl_gemm2_idx];
            int TB = nd.fa_TB, heads = nd.fa_heads, hd = nd.fa_hd;
            int N = nd.fa_N, H = nd.fa_H, W = nd.fa_W;
            int C = heads * hd;
            // Workspace pointer arithmetic: offsets are in element counts.
            // Use byte arithmetic with correct element size for fp16/fp32.
            int elem_sz = e->is_fp32 ? 4 : 2;
            char* ws_bytes = (char*)nd.fa_workspace;
            half* ws = nd.fa_workspace;
            void* pq = ws_bytes + nd.fa_ws_perm_q * elem_sz;
            void* pk = ws_bytes + nd.fa_ws_perm_k * elem_sz;
            void* pv = ws_bytes + nd.fa_ws_perm_v * elem_sz;
            void* gemm1_out = ws_bytes + nd.fa_ws_gemm1_out * elem_sz;

            if (nd.fa_variant == 0) {
                // ── SpikFormer: TileLang batched GEMM ──
                // 1. Permute Q,K,V: (TB,N,heads,hd) → (TB,heads,N,hd) via layout_transpose
                // 2. GEMM1 (bt): Q@K^T*scale via TileLang .so
                // 3. GEMM2: attn@V via TileLang .so
                // 4. Permute back: (TB,heads,N,hd) → (TB,N,heads,hd)
                // 5. LIF neuron
                int total = TB * N * C;
                int thr = 256, blk = (total + thr - 1) / thr;

                // Permute
                if (e->is_fp32) {
                    layout_transpose_fp32_kernel<<<blk, thr, 0, s>>>((float*)nd.fa_q, (float*)pq, TB, N, heads, hd, 0);
                    layout_transpose_fp32_kernel<<<blk, thr, 0, s>>>((float*)nd.fa_k, (float*)pk, TB, N, heads, hd, 0);
                    layout_transpose_fp32_kernel<<<blk, thr, 0, s>>>((float*)nd.fa_v, (float*)pv, TB, N, heads, hd, 0);
                } else {
                    layout_transpose_kernel<<<blk, thr, 0, s>>>(nd.fa_q, (half*)pq, TB, N, heads, hd, 0);
                    layout_transpose_kernel<<<blk, thr, 0, s>>>(nd.fa_k, (half*)pk, TB, N, heads, hd, 0);
                    layout_transpose_kernel<<<blk, thr, 0, s>>>(nd.fa_v, (half*)pv, TB, N, heads, hd, 0);
                }

                // GEMM1: attn_scores(batch*N, N) = Q(batch*N, hd) @ K(batch*N, hd)^T * scale
                ((CallFn3)tl1.call_fn)(pq, pk, gemm1_out, s);

                // GEMM2: out(batch*N, hd) = attn(batch*N, N) @ V(batch*N, hd)
                void* g2out = pq;  // reuse Q buffer
                ((CallFn3)tl2.call_fn)(gemm1_out, pv, g2out, s);

                // Permute back
                if (e->is_fp32)
                    layout_transpose_fp32_kernel<<<blk, thr, 0, s>>>((float*)g2out, (float*)nd.fa_out, TB, heads, N, hd, 1);
                else
                    layout_transpose_kernel<<<blk, thr, 0, s>>>((half*)g2out, nd.fa_out, TB, heads, N, hd, 1);

                // LIF
                if (nd.fa_lif_spatial > 0 && nd.fa_lif_spatial < nd.fa_lif_total) {
                    blk = (nd.fa_lif_spatial + thr - 1) / thr;
                    if (e->is_fp32)
                        lif_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                            (float*)nd.fa_out, nd.fa_membrane, (float*)nd.fa_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                    else
                        lif_neuron_kernel<<<blk, thr, 0, s>>>(
                            nd.fa_out, nd.fa_membrane, nd.fa_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                }
            }
            else if (nd.fa_variant == 1) {
                // ── MaxFormer ──
                ((CallFn3)tl1.call_fn)(nd.fa_k, nd.fa_v, gemm1_out, s);
                ((CallFn3)tl2.call_fn)(nd.fa_q, gemm1_out, nd.fa_out, s);
                if (nd.fa_lif_spatial > 0 && nd.fa_lif_spatial < nd.fa_lif_total) {
                    int thr = 256, blk = (nd.fa_lif_spatial + thr - 1) / thr;
                    if (e->is_fp32)
                        lif_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                            (float*)nd.fa_out, nd.fa_membrane, (float*)nd.fa_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                    else
                        lif_neuron_kernel<<<blk, thr, 0, s>>>(
                            nd.fa_out, nd.fa_membrane, nd.fa_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                }
            }
            else if (nd.fa_variant == 2) {
                // ── DSSA ──
                ((CallFn3)tl1.call_fn)(nd.fa_q, nd.fa_k, gemm1_out, s);

                // Scale1
                if (nd.fa_scale1_ptr) {
                    int batch = TB * heads;
                    int spatial_kv = nd.fa_ws_perm_q;
                    int spatial_q = nd.fa_N;
                    int total_attn = batch * spatial_kv * spatial_q;
                    int inner = spatial_kv * spatial_q;
                    int thr = 256, blk = (total_attn + thr - 1) / thr;
                    if (e->is_fp32)
                        scale_tensor_broadcast_fp32_kernel<<<blk, thr, 0, s>>>(
                            (float*)gemm1_out, (float*)nd.fa_scale1_ptr, total_attn, heads, inner);
                    else
                        scale_tensor_broadcast_kernel<<<blk, thr, 0, s>>>(
                            (half*)gemm1_out, nd.fa_scale1_ptr, total_attn, heads, inner);
                }

                // LIF on attention scores
                if (nd.fa_lif_spatial > 0 && nd.fa_lif_spatial < nd.fa_lif_total) {
                    int thr = 256, blk = (nd.fa_lif_spatial + thr - 1) / thr;
                    if (e->is_fp32)
                        lif_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                            (float*)gemm1_out, nd.fa_membrane, (float*)gemm1_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                    else
                        lif_neuron_kernel<<<blk, thr, 0, s>>>(
                            (half*)gemm1_out, nd.fa_membrane, (half*)gemm1_out,
                            nd.fa_lif_total, nd.fa_lif_spatial, nd.fa_v_thresh, nd.fa_recip_tau);
                }

                // GEMM2
                ((CallFn3)tl2.call_fn)(nd.fa_q, gemm1_out, nd.fa_out, s);

                // Scale2
                if (nd.fa_scale2_ptr) {
                    int spatial_q = nd.fa_N;
                    int total_out = TB * spatial_q * C;
                    int thr = 256, blk = (total_out + thr - 1) / thr;
                    if (e->is_fp32)
                        scale_nhwc_broadcast_fp32_kernel<<<blk, thr, 0, s>>>(
                            (float*)nd.fa_out, (float*)nd.fa_scale2_ptr, total_out, C, hd);
                    else
                        scale_nhwc_broadcast_kernel<<<blk, thr, 0, s>>>(
                            nd.fa_out, nd.fa_scale2_ptr, total_out, C, hd);
                }
            }
            else if (nd.fa_variant == 3) {
                // TokenQK: not yet in C++, handled by Python runtime.
            }
            break;
        }
        case KT_RESIZE: {
            int total = nd.rs_N * nd.rs_OH * nd.rs_OW * nd.rs_C;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                nearest_upsample_nhwc_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.rs_in, (float*)nd.rs_out,
                    nd.rs_N, nd.rs_H, nd.rs_W, nd.rs_C,
                    nd.rs_OH, nd.rs_OW, nd.rs_scale_h, nd.rs_scale_w);
            else
                nearest_upsample_nhwc_kernel<<<blk, thr, 0, s>>>(
                    (half*)nd.rs_in, (half*)nd.rs_out,
                    nd.rs_N, nd.rs_H, nd.rs_W, nd.rs_C,
                    nd.rs_OH, nd.rs_OW, nd.rs_scale_h, nd.rs_scale_w);
            break;
        }
        case KT_CONCAT: {
            int C_out = nd.cat_Ca + nd.cat_Cb;
            int total = nd.cat_NHW * C_out;
            int thr = 256, blk = (total + thr - 1) / thr;
            if (e->is_fp32)
                concat_nhwc_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.cat_a, (float*)nd.cat_b, (float*)nd.cat_out,
                    1, 1, nd.cat_NHW, nd.cat_Ca, nd.cat_Cb);
            else
                concat_nhwc_kernel<<<blk, thr, 0, s>>>(
                    (half*)nd.cat_a, (half*)nd.cat_b, (half*)nd.cat_out,
                    1, 1, nd.cat_NHW, nd.cat_Ca, nd.cat_Cb);
            break;
        }
        case KT_ILIF: {
            int thr = 256, blk = (nd.spatial_elems + thr - 1) / thr;
            if (e->is_fp32)
                ilif_neuron_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.input_ptr, nd.membrane_ptr, (float*)nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.ilif_decay, nd.ilif_max_level);
            else
                ilif_neuron_kernel<<<blk, thr, 0, s>>>(
                    nd.input_ptr, nd.membrane_ptr, nd.output_ptr,
                    nd.total_elems, nd.spatial_elems, nd.ilif_decay, nd.ilif_max_level);
            break;
        }
        case KT_SOFTMAX: {
            int thr = 256, blk = (nd.sm_outer + thr - 1) / thr;
            if (e->is_fp32)
                softmax_fp32_kernel<<<blk, thr, 0, s>>>(
                    (float*)nd.sm_in, (float*)nd.sm_out, nd.sm_outer, nd.sm_inner);
            else
                softmax_fp16_kernel<<<blk, thr, 0, s>>>(
                    (half*)nd.sm_in, (half*)nd.sm_out, nd.sm_outer, nd.sm_inner);
            break;
        }
        case KT_CUDNN_CONV: {
            // cuDNN convolution (NHWC) then BN epilogue (element-wise scale+bias)
            cudnnSetStream(e->cudnn, s);
            {
                const float alpha = 1.0f, beta = 0.0f;
                cudnnConvolutionForward(e->cudnn, &alpha,
                    nd.cudnn_in_desc, nd.nc_in,
                    nd.cudnn_filt_desc, nd.nc_w,
                    nd.cudnn_conv_desc, nd.cudnn_algo,
                    nd.cudnn_workspace, nd.cudnn_ws_size,
                    &beta, nd.cudnn_out_desc, nd.nc_out);
            }
            // BN epilogue: output = output * bn_scale + bn_bias (in-place, per-channel)
            if (nd.nc_sc && nd.nc_bi) {
                int total = nd.nc_N * nd.nc_OH * nd.nc_OW * nd.nc_Cout;
                int thr = 256, blk = (total + thr - 1) / thr;
                if (e->is_fp32)
                    bn_epilogue_fp32_kernel<<<blk, thr, 0, s>>>(
                        (float*)nd.nc_out, nd.nc_sc, nd.nc_bi, total, nd.nc_Cout);
                else
                    bn_epilogue_fp16_kernel<<<blk, thr, 0, s>>>(
                        nd.nc_out, nd.nc_sc, nd.nc_bi, total, nd.nc_Cout);
            }
            break;
        }
        case KT_SKIP:
            break;
        }
    }
}

void sengine_reset_membranes(SEngineExecutor* e) {
    for (int i = 0; i < e->n_membranes; i++) {
        int thr = 256, blk = (e->membrane_sizes[i] + thr - 1) / thr;
        zero_fp32_kernel<<<blk, thr, 0, e->stream>>>(
            e->membranes[i], e->membrane_sizes[i]);
    }
}

// Debug: run nodes sequentially, sync + check after EACH node, stop on first error
void sengine_execute_sequential_debug(SEngineExecutor* e) {
    cudaStream_t s = e->stream;
    cudaGetLastError(); // clear
    for (int i = 0; i < e->schedule_len; i++) {
        int nid = e->schedule[i];
        if (nid < 0 || nid > e->max_node_id) continue;
        auto& nd = e->nodes[nid];

        // Run this ONE node using the regular dispatch
        int orig_len = e->schedule_len;
        int* orig_sched = e->schedule;
        int single = nid;
        e->schedule = &single;
        e->schedule_len = 1;
        sengine_execute(e);
        e->schedule = orig_sched;
        e->schedule_len = orig_len;

        cudaError_t err = cudaStreamSynchronize(s);
        cudaError_t err2 = cudaGetLastError();
        if (err != cudaSuccess || err2 != cudaSuccess) {
            const char* msg = (err != cudaSuccess) ? cudaGetErrorString(err) : cudaGetErrorString(err2);
            fprintf(stderr, "FIRST FAILURE at schedule[%d]: node %d (type %d, tl_idx=%d, n_args=%d): %s\n",
                    i, nid, nd.type, nd.tilelang_idx, nd.tl_n_args, msg);
            // Print pointer info
            if (nd.tl_args) {
                fprintf(stderr, "  args: ");
                for (int a = 0; a < nd.tl_n_args && a < 8; a++)
                    fprintf(stderr, "[%d]=%p ", a, nd.tl_args[a]);
                fprintf(stderr, "\n");
            }
            return;
        }
    }
    fprintf(stderr, "Sequential debug: all %d nodes passed.\n", e->schedule_len);
}

// Debug: execute with error checking per node
void sengine_execute_checked(SEngineExecutor* e) {
    sengine_execute(e);
    cudaError_t err = cudaStreamSynchronize(e->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after execute: %s\n", cudaGetErrorString(err));
        // Find which node failed by running one at a time
        for (int i = 0; i < e->schedule_len; i++) {
            int nid = e->schedule[i];
            if (nid < 0 || nid > e->max_node_id) continue;
            // Save and restore schedule to run single node
            int orig_len = e->schedule_len;
            int orig_sched = e->schedule[0];
            e->schedule[0] = nid;
            e->schedule_len = 1;
            cudaGetLastError(); // clear
            sengine_execute(e);
            cudaError_t nerr = cudaStreamSynchronize(e->stream);
            if (nerr != cudaSuccess) {
                fprintf(stderr, "  Node %d (type %d) FAILED: %s\n",
                        nid, e->nodes[nid].type, cudaGetErrorString(nerr));
            }
            e->schedule[0] = orig_sched;
            e->schedule_len = orig_len;
        }
    }
}

void sengine_capture_graph(SEngineExecutor* e) {
    // Warm up with error checking
    cudaGetLastError(); // clear any prior errors
    sengine_execute_checked(e);
    for (int i = 1; i < 3; i++) sengine_execute(e);
    cudaStreamSynchronize(e->stream);

    cudaStreamBeginCapture(e->stream, cudaStreamCaptureModeGlobal);
    sengine_execute(e);
    cudaStreamEndCapture(e->stream, &e->graph);
    cudaGraphInstantiate(&e->exec, e->graph, 0);
    e->captured = 1;
    // Clear any deferred CUDA errors from warmup/capture
    cudaGetLastError();
}

void sengine_replay(SEngineExecutor* e) {
    if (e->captured) {
        cudaGraphLaunch(e->exec, e->stream);
    } else {
        sengine_execute(e);
    }
}

void sengine_sync(SEngineExecutor* e) {
    cudaStreamSynchronize(e->stream);
}

float sengine_benchmark(SEngineExecutor* e, int warmup, int n_iters) {
    for (int i = 0; i < warmup; i++) sengine_replay(e);
    cudaStreamSynchronize(e->stream);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, e->stream);
    for (int i = 0; i < n_iters; i++) sengine_replay(e);
    cudaEventRecord(end, e->stream);
    cudaStreamSynchronize(e->stream);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / n_iters;
}

} // extern "C"
