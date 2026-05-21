#include "sgemm_6x8.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>


/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps(dst,src1,src2)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
// sgemm_6x8(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 8] @DRAM,
//     C : f32[6, 8] @DRAM
// )
void sgemm_6x8( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C ) {
for (int_fast32_t i = 0; i < 6; i++) {
  for (int_fast32_t j = 0; j < 8; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * 8 + j] += A[i * K + k] * B[k * 8 + j];
    }
  }
}
}

// sgemm_6x8_avx2(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 8] @DRAM,
//     C : f32[6, 8] @DRAM
// )
void sgemm_6x8_avx2( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C ) {
__m256 C_reg[6];
for (int_fast32_t i0 = 0; i0 < 6; i0++) {
  C_reg[i0] = _mm256_loadu_ps(&C[(i0) * 8]);
}
for (int_fast32_t k = 0; k < K; k++) {
  __m256 B_reg;
  B_reg = _mm256_loadu_ps(&B[(k) * 8]);
  for (int_fast32_t i = 0; i < 6; i++) {
    __m256 A_reg;
    A_reg = _mm256_broadcast_ss(&A[(i) * K + k]);
    C_reg[i] = _mm256_fmadd_ps(A_reg, B_reg, C_reg[i]);
  }
}
for (int_fast32_t i0 = 0; i0 < 6; i0++) {
  _mm256_storeu_ps(&C[(i0) * 8], C_reg[i0]);
}
}

