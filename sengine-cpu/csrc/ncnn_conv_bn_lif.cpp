/**
 * ncnn Winograd Conv + fused BN+LIF wrapper.
 *
 * Uses ncnn's optimized Convolution layer (Winograd F(2,3)/F(4,3) with
 * AVX2 SIMD) as Conv backend, then applies BN+LIF on the L2-hot output.
 *
 * ncnn Conv outputs to Mat buffer → data is L1/L2 hot →
 * we immediately apply BN+LIF → spike output.
 *
 * This gives us ncnn-quality Conv (~1ms for VGG-9) + fused LIF epilogue.
 */

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <immintrin.h>

/* ncnn headers */
#include "net.h"
#include "layer.h"
#include "mat.h"

/* ═══ BN+LIF epilogue on ncnn Mat output ═══
 *
 * ncnn Mat layout: channel-major with elempack packing
 *   For elempack=8 (AVX2): data is packed as (H, W, C/8, 8)
 *   For elempack=1: data is (C, H, W) standard NCHW
 *
 * We handle both cases.
 */
static void apply_bn_lif_on_mat(
    const ncnn::Mat& conv_out,  /* ncnn Conv output, one frame */
    const float* bn_scale,
    const float* bn_bias,
    float* membrane,            /* (H*W, C) spatial-major */
    float* spike_out,           /* (H*W, C) spatial-major */
    int C, int H, int W,
    int frame_spatial_offset,   /* offset into membrane for this frame's batch */
    float v_threshold, float decay, float recip_tau)
{
    int HW = H * W;
    int elempack = conv_out.elempack;

    if (elempack == 1) {
        /* Standard NCHW layout */
        for (int c = 0; c < C; c++) {
            const float* chptr = conv_out.channel(c);
            float sc = bn_scale[c];
            float bi = bn_bias[c];
            for (int hw = 0; hw < HW; hw++) {
                float bn_val = chptr[hw] * sc + bi;
                float* mem = &membrane[(frame_spatial_offset + hw) * C + c];
                float h = decay * (*mem) + recip_tau * bn_val;
                float sp = (h >= v_threshold) ? 1.0f : 0.0f;
                *mem = (1.0f - sp) * h;
                spike_out[(frame_spatial_offset + hw) * C + c] = sp;
            }
        }
    } else {
        /* Packed layout (elempack=4 or 8): iterate by spatial position */
        for (int hw = 0; hw < HW; hw++) {
            int y = hw / W, x = hw % W;
            for (int c = 0; c < C; c++) {
                /* Access packed element */
                int c_outer = c / elempack;
                int c_inner = c % elempack;
                float val = conv_out.channel(c_outer).row(y)[x * elempack + c_inner];

                float bn_val = val * bn_scale[c] + bn_bias[c];
                float* mem = &membrane[(frame_spatial_offset + hw) * C + c];
                float h = decay * (*mem) + recip_tau * bn_val;
                float sp = (h >= v_threshold) ? 1.0f : 0.0f;
                *mem = (1.0f - sp) * h;
                spike_out[(frame_spatial_offset + hw) * C + c] = sp;
            }
        }
    }
}


/* ═══ Persistent Conv layer handle (create once, forward many times) ═══ */
struct NcnnConvHandle {
    ncnn::Layer* layer;
    ncnn::Option opt;
    int C_in, C_out, H, W, OH, OW;
};

extern "C"
void* ncnn_conv_create(
    const float* weight, int C_in, int H, int W, int C_out,
    int KH, int KW, int pad, int stride, int n_threads)
{
    auto* h = new NcnnConvHandle();
    h->C_in = C_in; h->C_out = C_out; h->H = H; h->W = W;
    h->OH = (H + 2*pad - KH) / stride + 1;
    h->OW = (W + 2*pad - KW) / stride + 1;

    h->layer = ncnn::create_layer("Convolution");

    ncnn::ParamDict pd;
    pd.set(0, C_out);
    pd.set(1, KH); pd.set(11, KH);
    pd.set(2, 1); pd.set(12, 1);
    pd.set(3, stride); pd.set(13, stride);
    pd.set(4, pad); pd.set(14, pad);
    pd.set(5, 0);  /* no bias */
    pd.set(6, C_out * C_in * KH * KW);
    pd.set(9, 0);  /* no activation */
    h->layer->load_param(pd);

    ncnn::Mat weights[1];
    weights[0] = ncnn::Mat(C_out * C_in * KH * KW);
    memcpy(weights[0], weight, C_out * C_in * KH * KW * sizeof(float));
    h->layer->load_model(ncnn::ModelBinFromMatArray(weights));

    h->opt.num_threads = n_threads;
    h->opt.use_packing_layout = true;
    h->layer->create_pipeline(h->opt);  /* Winograd weight transform happens HERE (once) */

    return h;
}

extern "C"
void ncnn_conv_destroy(void* handle) {
    auto* h = (NcnnConvHandle*)handle;
    h->layer->destroy_pipeline(h->opt);
    delete h->layer;
    delete h;
}

/* ═══ Forward: TB-merged Conv then BN+LIF T-loop ═══ */
extern "C"
void ncnn_conv_bn_lif_forward(
    void* handle,
    const float* input,       /* (TB, C_in, H, W) NCHW */
    const float* bn_scale,
    const float* bn_bias,
    float* membrane,          /* (B*OH*OW, C_out) spatial-major */
    float* spikes,            /* (T*B*OH*OW, C_out) spatial-major */
    int B, int T,
    float v_threshold, float decay, float recip_tau)
{
    auto* h = (NcnnConvHandle*)handle;
    int C_in = h->C_in, C_out = h->C_out, H = h->H, W = h->W;
    int OH = h->OH, OW = h->OW;
    int TB = T * B;
    int spatial_per_frame = OH * OW;

    /* Step 1: TB-merged Conv (all T*B frames in one call, ncnn uses Winograd) */
    ncnn::Mat in_mat(W, H, C_in * TB);  /* pack TB frames as extra channels */
    /* Actually ncnn Conv expects (W, H, C_in) for batch=1. For batch>1 we need
     * to call forward per-frame or use ncnn's batch dimension.
     * ncnn doesn't have native batch support in Conv — it processes N=1.
     * So: call forward TB times, but ALL at once before LIF. */

    /* Conv outputs stored contiguously */
    float* conv_output = (float*)aligned_alloc(64,
        (size_t)TB * C_out * OH * OW * sizeof(float));

    for (int n = 0; n < TB; n++) {
        ncnn::Mat frame_in(W, H, C_in);
        const float* src = input + (size_t)n * C_in * H * W;
        for (int c = 0; c < C_in; c++)
            memcpy(frame_in.channel(c), src + c * H * W, H * W * sizeof(float));

        ncnn::Mat frame_out;
        h->layer->forward(frame_in, frame_out, h->opt);

        /* Copy output to contiguous buffer (channel-first per frame) */
        float* dst = conv_output + (size_t)n * C_out * OH * OW;
        int elempack = frame_out.elempack;
        if (elempack == 1) {
            for (int c = 0; c < C_out; c++)
                memcpy(dst + c * OH * OW, frame_out.channel(c), OH * OW * sizeof(float));
        } else {
            /* Unpack from packed layout */
            for (int c = 0; c < C_out; c++) {
                int c_outer = c / elempack, c_inner = c % elempack;
                for (int hw = 0; hw < OH * OW; hw++) {
                    int y = hw / OW, x = hw % OW;
                    dst[c * OH * OW + hw] = frame_out.channel(c_outer).row(y)[x * elempack + c_inner];
                }
            }
        }
    }

    /* Step 2: BN+LIF T-loop on conv_output (now in L2 cache) */
    memset(membrane, 0, (size_t)B * spatial_per_frame * C_out * sizeof(float));

    for (int t = 0; t < T; t++) {
        for (int b = 0; b < B; b++) {
            int frame_idx = t * B + b;
            const float* frame_out = conv_output + (size_t)frame_idx * C_out * OH * OW;
            float* spike_frame = spikes + (size_t)frame_idx * spatial_per_frame * C_out;
            int spatial_offset = b * spatial_per_frame;

            for (int hw = 0; hw < spatial_per_frame; hw++) {
                for (int c = 0; c < C_out; c++) {
                    float val = frame_out[c * OH * OW + hw];  /* NCHW */
                    float bn_val = val * bn_scale[c] + bn_bias[c];
                    float* mem = &membrane[(spatial_offset + hw) * C_out + c];
                    float hh = decay * (*mem) + recip_tau * bn_val;
                    float sp = (hh >= v_threshold) ? 1.0f : 0.0f;
                    *mem = (1.0f - sp) * hh;
                    spike_frame[hw * C_out + c] = sp;
                }
            }
        }
    }

    free(conv_output);
}
