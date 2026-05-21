/**
 * Winograd F(2,3) Conv3x3 + BN + LIF fused kernel.
 *
 * Algorithm: Winograd F(2,3) produces 2×2 output from 4×4 input tile.
 *   FLOPs: 16 mul per output element (vs 18 for direct 3×3 conv) = 2.25x reduction.
 *
 * Fusion: BN+LIF is injected in the output transform, BEFORE writing to memory.
 *   The conv result lives in YMM registers → BN scale/bias → LIF dynamics → store spike.
 *   Membrane loads/stores are spatial-major (contiguous per position).
 *
 * Based on ncnn's convolution_3x3_winograd.h (MIT License, Tencent).
 * Stripped down to single-precision FP32, AVX2, one function, no framework dependency.
 *
 * Compile: gcc -O3 -march=native -mavx2 -mfma -fPIC -fopenmp -shared -o libwinograd_lif.so winograd_conv_bn_lif.c
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Winograd F(2,3) constants:
 *   Input transform G:  4×3 matrix
 *   Output transform A^T: 2×4 matrix
 *
 *   BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
 *   AT = [[1,1,1,0],[0,1,-1,1]]
 *   G  = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
 */

/* ═══ Weight transform: (C_out, C_in, 3, 3) → (16, C_out, C_in) Winograd domain ═══ */
void winograd23_transform_kernel(
    const float* kernel,       /* (C_out, C_in, 3, 3) */
    float* kernel_tm,          /* (16, C_out, C_in) output */
    int C_in, int C_out)
{
    /* G * g * G^T for each (cout, cin) pair */
    /* G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]] */
    for (int co = 0; co < C_out; co++) {
        for (int ci = 0; ci < C_in; ci++) {
            const float* k = kernel + (co * C_in + ci) * 9;
            float g00=k[0],g01=k[1],g02=k[2];
            float g10=k[3],g11=k[4],g12=k[5];
            float g20=k[6],g21=k[7],g22=k[8];

            /* tmp = G * g (4×3) */
            float t00=g00, t01=g01, t02=g02;
            float t10=0.5f*(g00+g10+g20), t11=0.5f*(g01+g11+g21), t12=0.5f*(g02+g12+g22);
            float t20=0.5f*(g00-g10+g20), t21=0.5f*(g01-g11+g21), t22=0.5f*(g02-g12+g22);
            float t30=g20, t31=g21, t32=g22;

            /* U = tmp * G^T (4×4) */
            float* u = kernel_tm + ci * C_out + co;  /* interleaved layout */
            int stride = C_out * C_in;
            u[0*stride] = t00;
            u[1*stride] = 0.5f*(t00+t01+t02);
            u[2*stride] = 0.5f*(t00-t01+t02);
            u[3*stride] = t02;
            u[4*stride] = t10;
            u[5*stride] = 0.5f*(t10+t11+t12);
            u[6*stride] = 0.5f*(t10-t11+t12);
            u[7*stride] = t12;
            u[8*stride] = t20;
            u[9*stride] = 0.5f*(t20+t21+t22);
            u[10*stride]= 0.5f*(t20-t21+t22);
            u[11*stride]= t22;
            u[12*stride]= t30;
            u[13*stride]= 0.5f*(t30+t31+t32);
            u[14*stride]= 0.5f*(t30-t31+t32);
            u[15*stride]= t32;
        }
    }
}


/* ═══ Fused Winograd F(2,3) Conv + BN + LIF ═══ */
void winograd23_conv_bn_lif(
    const float* input,        /* (N, C_in, H, W) NCHW */
    const float* kernel_tm,    /* (16, C_out, C_in) pre-transformed */
    const float* bn_scale,     /* (C_out,) */
    const float* bn_bias,      /* (C_out,) */
    float* membrane,           /* (spatial, C_out) spatial-major, persistent across T */
    float* spikes,             /* (N*OH*OW, C_out) spatial-major output */
    int N, int C_in, int H, int W, int C_out,
    int T, int B,              /* N = T*B */
    float v_threshold, float decay, float recip_tau)
{
    int OH = H;  /* stride=1, pad=1: OH=H, OW=W */
    int OW = W;
    int tile_h = (OH + 1) / 2;  /* number of 2×2 output tiles vertically */
    int tile_w = (OW + 1) / 2;
    int n_tiles = tile_h * tile_w;
    int spatial_per_T = B * OH * OW;

    /* Allocate workspace for input tiles and GEMM output */
    int tile_count = N * n_tiles;
    float* input_tm = (float*)aligned_alloc(64,
        (size_t)16 * tile_count * C_in * sizeof(float));
    float* output_tm = (float*)aligned_alloc(64,
        (size_t)16 * tile_count * C_out * sizeof(float));

    /* ── Step 1: Input transform BT * d * B ── */
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N * n_tiles; idx++) {
        int n = idx / n_tiles;
        int tile = idx % n_tiles;
        int ti = tile / tile_w;
        int tj = tile % tile_w;
        int ih = ti * 2 - 1;  /* pad=1 */
        int iw = tj * 2 - 1;

        float d[4][4];
        for (int ci = 0; ci < C_in; ci++) {
            const float* src = input + ((long)n * C_in + ci) * H * W;
            /* Load 4×4 input tile with boundary handling */
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) {
                    int y = ih + r, x = iw + c;
                    d[r][c] = (y >= 0 && y < H && x >= 0 && x < W) ? src[y*W+x] : 0.0f;
                }

            /* BT * d: rows */
            float bt[4][4];
            bt[0][0]=d[0][0]-d[2][0]; bt[0][1]=d[0][1]-d[2][1]; bt[0][2]=d[0][2]-d[2][2]; bt[0][3]=d[0][3]-d[2][3];
            bt[1][0]=d[1][0]+d[2][0]; bt[1][1]=d[1][1]+d[2][1]; bt[1][2]=d[1][2]+d[2][2]; bt[1][3]=d[1][3]+d[2][3];
            bt[2][0]=d[2][0]-d[1][0]; bt[2][1]=d[2][1]-d[1][1]; bt[2][2]=d[2][2]-d[1][2]; bt[2][3]=d[2][3]-d[1][3];
            bt[3][0]=d[1][0]-d[3][0]; bt[3][1]=d[1][1]-d[3][1]; bt[3][2]=d[1][2]-d[3][2]; bt[3][3]=d[1][3]-d[3][3];

            /* * B: cols → V = BT * d * B */
            float* dst = input_tm + ci * tile_count + idx;
            int ts = C_in * tile_count;
            dst[0*ts] = bt[0][0]-bt[0][2]; dst[1*ts] = bt[0][1]+bt[0][2];
            dst[2*ts] = bt[0][2]-bt[0][1]; dst[3*ts] = bt[0][1]-bt[0][3];
            dst[4*ts] = bt[1][0]-bt[1][2]; dst[5*ts] = bt[1][1]+bt[1][2];
            dst[6*ts] = bt[1][2]-bt[1][1]; dst[7*ts] = bt[1][1]-bt[1][3];
            dst[8*ts] = bt[2][0]-bt[2][2]; dst[9*ts] = bt[2][1]+bt[2][2];
            dst[10*ts]= bt[2][2]-bt[2][1]; dst[11*ts]= bt[2][1]-bt[2][3];
            dst[12*ts]= bt[3][0]-bt[3][2]; dst[13*ts]= bt[3][1]+bt[3][2];
            dst[14*ts]= bt[3][2]-bt[3][1]; dst[15*ts]= bt[3][1]-bt[3][3];
        }
    }

    /* ── Step 2: Batched GEMM in Winograd domain ── */
    /* For each of 16 frequency points: output_tm[p] = kernel_tm[p] @ input_tm[p]^T */
    /* kernel_tm[p]: (C_in, C_out), input_tm[p]: (C_in, tile_count) */
    /* output_tm[p]: (C_out, tile_count) = kernel_tm[p]^T @ input_tm[p] */
    int gem_stride = C_in * tile_count;
    int out_stride = C_out * tile_count;

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < 16; p++) {
        const float* A = kernel_tm + p * C_out * C_in;  /* (C_in, C_out) */
        const float* Bp = input_tm + p * gem_stride;     /* (C_in, tile_count) */
        float* Cp = output_tm + p * out_stride;           /* (C_out, tile_count) */

        /* C = A^T @ B: (C_out, tile_count) */
        for (int co = 0; co < C_out; co++) {
            for (int t = 0; t < tile_count; t++) {
                float sum = 0;
                for (int ci = 0; ci < C_in; ci++) {
                    sum += A[ci * C_out + co] * Bp[ci * tile_count + t];
                }
                Cp[co * tile_count + t] = sum;
            }
        }
    }

    /* ── Step 3: Output transform AT * M * A + BN + LIF (FUSED) ── */
    /* AT = [[1,1,1,0],[0,1,-1,1]] */
    /* For each tile: transform 4×4 freq → 2×2 spatial, then apply BN+LIF */

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N * n_tiles; idx++) {
        int n = idx / n_tiles;
        int tile = idx % n_tiles;
        int ti = tile / tile_w;
        int tj = tile % tile_w;

        for (int co = 0; co < C_out; co++) {
            /* Gather 4×4 from output_tm */
            float m[4][4];
            for (int p = 0; p < 16; p++) {
                int pr = p / 4, pc = p % 4;
                m[pr][pc] = output_tm[p * out_stride + co * tile_count + idx];
            }

            /* AT * m: rows (4→2) */
            float at[2][4];
            at[0][0]=m[0][0]+m[1][0]+m[2][0]; at[0][1]=m[0][1]+m[1][1]+m[2][1];
            at[0][2]=m[0][2]+m[1][2]+m[2][2]; at[0][3]=m[0][3]+m[1][3]+m[2][3];
            at[1][0]=m[1][0]-m[2][0]+m[3][0]; at[1][1]=m[1][1]-m[2][1]+m[3][1];
            at[1][2]=m[1][2]-m[2][2]+m[3][2]; at[1][3]=m[1][3]-m[2][3]+m[3][3];

            /* * A: cols (4→2) → 2×2 conv output */
            float out[2][2];
            out[0][0] = at[0][0]+at[0][1]+at[0][2];
            out[0][1] = at[0][1]-at[0][2]+at[0][3];
            out[1][0] = at[1][0]+at[1][1]+at[1][2];
            out[1][1] = at[1][1]-at[1][2]+at[1][3];

            /* ═══ FUSED BN + LIF (register-level, no memory round-trip) ═══ */
            float scale = bn_scale[co];
            float bias = bn_bias[co];

            for (int dy = 0; dy < 2; dy++) {
                int oy = ti * 2 + dy;
                if (oy >= OH) continue;
                for (int dx = 0; dx < 2; dx++) {
                    int ox = tj * 2 + dx;
                    if (ox >= OW) continue;

                    float conv_out = out[dy][dx];
                    float bn_val = conv_out * scale + bias;

                    /* Spatial position in the output */
                    int spatial_pos = (n / T) * OH * OW + oy * OW + ox;  /* b*OH*OW + oy*OW + ox */
                    int t_step = n % (N / B);  /* which timestep (0..T-1, via n index) */
                    /* Actually n indexes merged T*B: n = t*B + b */
                    int b_idx = n % B;
                    int t_idx = n / B;
                    spatial_pos = b_idx * OH * OW + oy * OW + ox;

                    /* LIF dynamics */
                    float* mem = &membrane[spatial_pos * C_out + co];
                    float h = decay * (*mem) + recip_tau * bn_val;
                    float spike = (h >= v_threshold) ? 1.0f : 0.0f;
                    *mem = (1.0f - spike) * h;

                    /* Output spike (spatial-major) */
                    int out_pos = (t_idx * B + b_idx) * OH * OW + oy * OW + ox;
                    spikes[out_pos * C_out + co] = spike;
                }
            }
        }
    }

    free(input_tm);
    free(output_tm);
}
