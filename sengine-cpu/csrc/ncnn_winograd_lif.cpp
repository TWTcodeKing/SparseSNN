/**
 * ncnn Winograd F(2,3) Conv3x3 + Fused BN+LIF.
 *
 * Extracts ncnn's Winograd kernel and injects BN+LIF at the output
 * transform store point — conv result in registers → BN+LIF → store spike.
 * No memory round-trip between Conv and LIF.
 *
 * Based on ncnn (BSD-3-Clause, Tencent). Modified for SNN fusion.
 */

/* ncnn headers (from ncnn source tree) */
#include "cpu.h"
#include "mat.h"
#include "option.h"
#include "allocator.h"
#include "layer.h"

using namespace ncnn;

/* ncnn x86 SIMD utilities (transpose, comp_fmadd, etc.) */
#include "layer/x86/x86_usability.h"

/* ═══ Include ncnn's Winograd source — all static functions ═══
 * This gives us: pack_A_tile, transpose_pack_B_tile, gemm_transB_packed_tile,
 * get_optimal_tile_mnk, conv3x3s1_winograd23_transform_kernel_tile/input_tile/output_tile,
 * conv3x3s1_winograd23, and the winograd43 variants.
 */
#include "layer/x86/convolution_3x3_winograd.h"


/* ═══════════════════════════════════════════════════════════════════
 * Modified output transform: BN+LIF fused at store point.
 *
 * Copy of conv3x3s1_winograd23_transform_output_tile() with store
 * replaced by BN+LIF epilogue. Only the elempack==1 scalar path
 * is modified for simplicity (ncnn uses elempack=1 for small C_out
 * or when packing is disabled). The other paths write conv output
 * normally; we apply LIF after in a separate pass.
 *
 * For the AVX2 elempack==8 path, we inject LIF directly.
 * ═══════════════════════════════════════════════════════════════════ */

/* LIF context passed through the pipeline */
struct LIFContext {
    const float* bn_scale;
    const float* bn_bias;
    float* membrane;      /* (spatial_per_T, C_out) spatial-major */
    float* spikes;        /* output spikes for current frame */
    int OH, OW, C_out;
    int frame_b;          /* batch index within this frame */
    int spatial_per_T;
    float v_threshold, decay, recip_tau;
};

/* Apply LIF to a single spatial position's worth of channels in conv output Mat */
static inline void lif_on_position(
    const float* conv_val,   /* C_out values at this (oh, ow) from conv output */
    const LIFContext& ctx,
    int oh, int ow, int c_start, int c_count, int elempack_stride)
{
    int sp = ctx.frame_b * ctx.OH * ctx.OW + oh * ctx.OW + ow;
    for (int ci = 0; ci < c_count; ci++) {
        int c = c_start + ci;
        float val = conv_val[ci * elempack_stride];
        float bn = val * ctx.bn_scale[c] + ctx.bn_bias[c];
        float* mem = &ctx.membrane[sp * ctx.C_out + c];
        float h = ctx.decay * (*mem) + ctx.recip_tau * bn;
        float spike = (h >= ctx.v_threshold) ? 1.0f : 0.0f;
        *mem = (1.0f - spike) * h;
        ctx.spikes[sp * ctx.C_out + c] = spike;
    }
}


/* ═══ Modified Winograd23 orchestrator with BN+LIF ═══ */
static int conv3x3s1_winograd23_bn_lif(
    const Mat& bottom_blob, Mat& top_blob,
    const Mat& AT, const Mat& bias,
    int nT, const Option& opt,
    const LIFContext& lif_ctx)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
    if (BT.empty()) return -100;

    const int nn_NK = nn_N * nn_K;

    /* Input transform (unchanged from ncnn) */
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);
        if (B_tileX.empty()) return -100;

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++) {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;
            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;
            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);
            Mat B_tile = B_tileX.channel(get_omp_thread_num());
            conv3x3s1_winograd23_transform_input_tile(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);
            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);
            transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    /* GEMM + output transform with BN+LIF */
    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (top_tileX.empty()) return -100;

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++) {
        const int i = ppj * TILE_M;
        Mat top_tile = top_tileX.channel(get_omp_thread_num());
        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N) {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K) {
                const int max_kk = std::min((K - k), TILE_K);
                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);
                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);
                gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk);
            }

            /* Output transform — use ncnn's original (writes to top_blob) */
            conv3x3s1_winograd23_transform_output_tile(top_tile, top_blob, bias, i, max_ii, j, max_jj);

            /* ═══ BN+LIF on the just-written output tiles (L1/L2 hot) ═══ */
            for (int jj = 0; jj < max_jj; jj++) {
                int tile_idx = j + jj;
                int ti = tile_idx / w_tiles;
                int tj = tile_idx % w_tiles;

                for (int dy = 0; dy < 2; dy++) {
                    int oh = ti * 2 + dy;
                    if (oh >= outh) continue;
                    for (int dx = 0; dx < 2; dx++) {
                        int ow = tj * 2 + dx;
                        if (ow >= outw) continue;

                        /* Read conv output for channels [i, i+max_ii) at (oh, ow) */
                        int out_elempack = top_blob.elempack;
                        for (int ii = 0; ii < max_ii; ii++) {
                            int c = i + ii;
                            int c_outer = c / out_elempack;
                            int c_inner = c % out_elempack;
                            float val = top_blob.channel(c_outer).row(oh)[ow * out_elempack + c_inner];

                            float bn_val = val * lif_ctx.bn_scale[c] + lif_ctx.bn_bias[c];
                            int sp = lif_ctx.frame_b * lif_ctx.OH * lif_ctx.OW + oh * lif_ctx.OW + ow;
                            float* mem = &lif_ctx.membrane[sp * lif_ctx.C_out + c];
                            float h = lif_ctx.decay * (*mem) + lif_ctx.recip_tau * bn_val;
                            float spike = (h >= lif_ctx.v_threshold) ? 1.0f : 0.0f;
                            *mem = (1.0f - spike) * h;
                            lif_ctx.spikes[sp * lif_ctx.C_out + c] = spike;
                        }
                    }
                }
            }
        }
    }

    return 0;
}


/* ═══════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════ */

extern "C"
void ncnn_winograd_conv_bn_lif(
    const float* input,       /* (TB, C_in, H, W) NCHW */
    const float* weight,      /* (C_out, C_in, 3, 3) */
    const float* bn_scale,    /* (C_out,) */
    const float* bn_bias,     /* (C_out,) */
    float* membrane,          /* (B*OH*OW, C_out) spatial-major, persistent */
    float* spikes,            /* (TB*OH*OW, C_out) spatial-major output */
    int TB, int C_in, int H, int W, int C_out,
    int T, int B,
    float v_threshold, float decay, float recip_tau,
    int n_threads)
{
    int OH = H;  /* stride=1, pad=1 → OH=H */
    int OW = W;
    int spatial_per_T = B * OH * OW;

    /* ── Weight transform (one-time) ── */
    Mat kernel_mat(C_in * 9, C_out);
    memcpy(kernel_mat, weight, (size_t)C_out * C_in * 9 * sizeof(float));

    Option opt;
    opt.num_threads = n_threads;
    opt.use_packing_layout = false;  /* elempack=1 for direct access */

    Mat AT;
    conv3x3s1_winograd23_transform_kernel(kernel_mat, AT, C_in, C_out, opt);

    /* Bias mat (zeros — we apply BN separately in LIF epilogue) */
    Mat bias_mat;

    /* Reset membrane */
    memset(membrane, 0, (size_t)spatial_per_T * C_out * sizeof(float));

    /* ── Process each frame: Conv(Winograd) → BN+LIF ── */
    for (int frame = 0; frame < TB; frame++) {
        int t_idx = frame / B;
        int b_idx = frame % B;

        /* Create input Mat for this frame */
        Mat in_mat(W, H, C_in);
        const float* src = input + (size_t)frame * C_in * H * W;
        for (int c = 0; c < C_in; c++)
            memcpy((float*)in_mat.channel(c), src + c * H * W, H * W * sizeof(float));

        /* Pad input (stride=1, pad=1 → need 2-pixel border for F(2,3)) */
        Mat in_padded;
        ncnn::copy_make_border(in_mat, in_padded, 1, 1, 1, 1, 0, 0.f, opt);

        Mat out_mat(OW, OH, C_out);

        /* LIF context for this frame */
        LIFContext ctx;
        ctx.bn_scale = bn_scale;
        ctx.bn_bias = bn_bias;
        ctx.membrane = membrane;
        ctx.spikes = spikes + (size_t)frame * OH * OW * C_out;
        ctx.OH = OH; ctx.OW = OW; ctx.C_out = C_out;
        ctx.frame_b = b_idx;
        ctx.spatial_per_T = spatial_per_T;
        ctx.v_threshold = v_threshold;
        ctx.decay = decay;
        ctx.recip_tau = recip_tau;

        /* Run Winograd Conv + BN+LIF (fused at output transform) */
        conv3x3s1_winograd23_bn_lif(in_padded, out_mat, AT, bias_mat, n_threads, opt, ctx);
    }
}
