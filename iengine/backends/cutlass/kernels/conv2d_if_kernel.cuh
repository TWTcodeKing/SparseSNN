/*****************************************************************************
 * CUTLASS Conv2d + BN + IF Neuron Fused Kernel
 *
 * Instantiates a CUTLASS implicit GEMM Conv2d with our custom IF neuron
 * epilogue.  FP16 tensor cores for the Conv body, FP32 compute in epilogue.
 *
 * Two versions:
 *   1. __global__ kernel: standard CUTLASS launch from host
 *   2. __device__ block kernel: Rammer-transformed for PTB mega-kernel
 *
 * The __device__ version replaces blockIdx/threadIdx with parameters,
 * allowing it to be called from within a persistent thread block.
 *****************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"

#include "snn_epilogue.h"

namespace snn {
namespace kernels {

/*****************************************************************************
 * Kernel type definitions for RTX 4090 (SM89, Ampere-compatible)
 *****************************************************************************/

// Data types
using ElementInputA  = cutlass::half_t;          // Activation: FP16
using ElementInputB  = cutlass::half_t;          // Weight: FP16
using ElementOutput  = cutlass::half_t;          // Output: FP16 (spike)
using ElementAccum   = float;                     // Accumulation: FP32
using ElementCompute = float;                     // Epilogue compute: FP32

// Layouts: NHWC for tensor core efficiency
using LayoutInputA   = cutlass::layout::TensorNHWC;
using LayoutInputB   = cutlass::layout::TensorNHWC;
using LayoutOutput   = cutlass::layout::TensorNHWC;

// Architecture
using SmArch = cutlass::arch::Sm80;  // SM89 is Ampere-compatible

// MMA operation: FP16 tensor cores → FP32 accumulation
using MmaOp = cutlass::arch::OpClassTensorOp;

/*****************************************************************************
 * Conv2d + IF Neuron kernel type
 *
 * Tile sizes are configurable per-layer for optimal performance.
 * Defaults: 128x128 threadblock, 64x64 warp, 16x8x16 instruction
 *****************************************************************************/

template <
  int ThreadblockM = 128,
  int ThreadblockN = 128,
  int ThreadblockK = 64,
  int WarpM = 64,
  int WarpN = 64,
  int InstructionM = 16,
  int InstructionN = 8,
  int InstructionK = 16,
  int AlignmentA = 8,
  int AlignmentB = 8
>
struct Conv2dIFNeuronKernel {

  // Epilogue: Conv + BN + IF neuron (all in registers)
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationIFNeuron<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // vector width
      ElementAccum,
      ElementCompute
  >;

  // Tile shapes
  using ThreadblockShape = cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;

  // Swizzle: identity (can optimize later)
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Number of pipeline stages
  static int const kStages = 3;

  // Alignment
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;

  // Full kernel type
  using ImplicitGemmKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementInputA, LayoutInputA,
      ElementInputB, LayoutInputB,
      ElementOutput, LayoutOutput,
      ElementAccum,
      MmaOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      kStages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      kAlignmentA,
      kAlignmentB
  >::Kernel;

  // Device-level wrapper
  using DeviceConv = cutlass::conv::device::ImplicitGemmConvolution<ImplicitGemmKernel>;
};


/*****************************************************************************
 * Conv2d + LIF Neuron kernel type (same structure, different epilogue)
 *****************************************************************************/

template <
  int ThreadblockM = 128,
  int ThreadblockN = 128,
  int ThreadblockK = 64,
  int WarpM = 64,
  int WarpN = 64,
  int InstructionM = 16,
  int InstructionN = 8,
  int InstructionK = 16,
  int AlignmentA = 8,
  int AlignmentB = 8
>
struct Conv2dLIFNeuronKernel {

  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationLIFNeuron<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccum,
      ElementCompute
  >;

  using ThreadblockShape = cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>;
  using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>;
  using InstructionShape = cutlass::gemm::GemmShape<InstructionM, InstructionN, InstructionK>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  static int const kStages = 3;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;

  using ImplicitGemmKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementInputA, LayoutInputA,
      ElementInputB, LayoutInputB,
      ElementOutput, LayoutOutput,
      ElementAccum,
      MmaOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      kStages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      kAlignmentA,
      kAlignmentB
  >::Kernel;

  using DeviceConv = cutlass::conv::device::ImplicitGemmConvolution<ImplicitGemmKernel>;
};

}  // namespace kernels
}  // namespace snn
