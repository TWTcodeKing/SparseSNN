
#pragma once
#ifndef SGEMM_6X8_H
#define SGEMM_6X8_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


#ifndef EXO_WIN_1F32
#define EXO_WIN_1F32
struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1F32C
#define EXO_WIN_1F32C
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
#endif
// sgemm_6x8(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 8] @DRAM,
//     C : f32[6, 8] @DRAM
// )
void sgemm_6x8( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C );

// sgemm_6x8_avx2(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 8] @DRAM,
//     C : f32[6, 8] @DRAM
// )
void sgemm_6x8_avx2( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C );



#ifdef __cplusplus
}
#endif
#endif  // SGEMM_6X8_H
