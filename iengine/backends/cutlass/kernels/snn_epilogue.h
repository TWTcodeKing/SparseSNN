/*****************************************************************************
 * SNN Epilogue Functors for CUTLASS Conv2d
 *
 * Custom epilogue that fuses BN + IF/LIF neuron into the Conv2d kernel.
 * Conv output stays in registers through BN and neuron — never hits DRAM.
 *
 * This is the key contribution: TRT's cuDNN cannot fuse Conv with custom
 * stateful activations (IF/LIF), but CUTLASS's epilogue functor API can.
 *
 * Usage in CUTLASS kernel instantiation:
 *   using EpilogueOp = LinearCombinationIFNeuron<half_t, 8, float, float>;
 *   using Conv2dKernel = DefaultConv2dFprop<..., EpilogueOp, ...>;
 *****************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/scale_type.h"

namespace cutlass {
namespace epilogue {
namespace thread {

/*****************************************************************************
 * IF Neuron Epilogue: Conv output → BN → IF spike + membrane state update
 *
 * Computes (per element, in registers):
 *   bn_out  = alpha * accumulator + beta * source    // Conv+BN (alpha=scale, beta=bias)
 *   h       = state + bn_out                          // membrane potential update
 *   spike   = (h >= v_threshold) ? 1.0 : 0.0         // fire
 *   state'  = (1 - spike) * h + spike * v_reset       // reset
 *
 * Output: spike (binary), Side-effect: updates state buffer
 *****************************************************************************/

template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = float,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationIFNeuron {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kCount = Count;
  static const bool kIsHeavy = true;  // Has side effects (state update)

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;
  using FragmentState = Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;           // BN scale (folded)
    ElementCompute beta;            // BN bias (folded)
    ElementCompute v_threshold;     // IF neuron threshold
    ElementCompute v_reset;         // IF neuron reset value
    ElementCompute *state_ptr;      // Pointer to membrane state buffer
    int state_stride;               // Stride in state buffer (elements per row)

    CUTLASS_HOST_DEVICE
    Params():
      alpha(1), beta(0), v_threshold(1), v_reset(0),
      state_ptr(nullptr), state_stride(0) {}

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha,
      ElementCompute beta,
      ElementCompute v_threshold,
      ElementCompute v_reset,
      ElementCompute *state_ptr = nullptr,
      int state_stride = 0
    ): alpha(alpha), beta(beta), v_threshold(v_threshold), v_reset(v_reset),
       state_ptr(state_ptr), state_stride(state_stride) {}
  };

private:
  Params params_;

public:

  CUTLASS_HOST_DEVICE
  LinearCombinationIFNeuron(Params const &params): params_(params) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// D = IF_neuron(alpha * accumulator + beta * source, state)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source,
    int linear_idx = 0) const {

    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> acc_converter;
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> out_converter;

    FragmentCompute conv_out = acc_converter(accumulator);
    FragmentCompute bn_bias = source_converter(source);

    FragmentOutput result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      // BN: scale * conv_out + bias
      // (alpha = BN_scale, source = BN_bias passed via beta path)
      ElementCompute bn_out = params_.alpha * conv_out[i] + bn_bias[i];

      // IF neuron: h = v + bn_out
      ElementCompute v = ElementCompute(0);
      if (params_.state_ptr) {
        v = params_.state_ptr[linear_idx + i];
      }
      ElementCompute h = v + bn_out;

      // Fire
      ElementCompute spike = (h >= params_.v_threshold)
                             ? ElementCompute(1) : ElementCompute(0);

      // Reset
      ElementCompute v_new = (ElementCompute(1) - spike) * h
                             + spike * params_.v_reset;

      // Write back state
      if (params_.state_ptr) {
        params_.state_ptr[linear_idx + i] = v_new;
      }

      // Output = spike
      result[i] = ElementOutput(spike);
    }

    return result;
  }

  /// D = IF_neuron(alpha * accumulator, state)  — no source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    int linear_idx = 0) const {

    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> acc_converter;
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> out_converter;

    FragmentCompute conv_out = acc_converter(accumulator);
    FragmentOutput result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute bn_out = params_.alpha * conv_out[i] + params_.beta;
      ElementCompute v = params_.state_ptr ? params_.state_ptr[linear_idx + i]
                                           : ElementCompute(0);
      ElementCompute h = v + bn_out;
      ElementCompute spike = (h >= params_.v_threshold)
                             ? ElementCompute(1) : ElementCompute(0);
      ElementCompute v_new = (ElementCompute(1) - spike) * h
                             + spike * params_.v_reset;
      if (params_.state_ptr) {
        params_.state_ptr[linear_idx + i] = v_new;
      }
      result[i] = ElementOutput(spike);
    }
    return result;
  }
};


/*****************************************************************************
 * LIF Neuron Epilogue: Conv output → BN → LIF spike + membrane state update
 *
 * LIF dynamics (per element, in registers):
 *   bn_out  = alpha * accumulator + beta * source
 *   h       = (1 - 1/tau) * state + (1/tau) * bn_out
 *   spike   = (h >= v_threshold) ? 1.0 : 0.0
 *   state'  = (1 - spike) * h + spike * v_reset
 *****************************************************************************/

template <
  typename ElementOutput_,
  int Count,
  typename ElementAccumulator_ = ElementOutput_,
  typename ElementCompute_ = float,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
class LinearCombinationLIFNeuron {
public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  static int const kCount = Count;
  static const bool kIsHeavy = true;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using FragmentCompute = Array<ElementCompute, kCount>;

  struct Params {
    ElementCompute alpha;           // BN scale (folded)
    ElementCompute beta;            // BN bias (folded)
    ElementCompute recip_tau;       // 1/tau for LIF leak
    ElementCompute v_threshold;
    ElementCompute v_reset;
    ElementCompute *state_ptr;
    int state_stride;

    CUTLASS_HOST_DEVICE
    Params():
      alpha(1), beta(0), recip_tau(0.5f), v_threshold(1), v_reset(0),
      state_ptr(nullptr), state_stride(0) {}

    CUTLASS_HOST_DEVICE
    Params(
      ElementCompute alpha, ElementCompute beta,
      ElementCompute recip_tau, ElementCompute v_threshold,
      ElementCompute v_reset,
      ElementCompute *state_ptr = nullptr, int state_stride = 0
    ): alpha(alpha), beta(beta), recip_tau(recip_tau),
       v_threshold(v_threshold), v_reset(v_reset),
       state_ptr(state_ptr), state_stride(state_stride) {}
  };

private:
  Params params_;

public:

  CUTLASS_HOST_DEVICE
  LinearCombinationLIFNeuron(Params const &params): params_(params) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    FragmentOutput const &source,
    int linear_idx = 0) const {

    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> acc_converter;

    FragmentCompute conv_out = acc_converter(accumulator);
    FragmentCompute bn_bias = source_converter(source);

    FragmentOutput result;
    ElementCompute one_sub_recip = ElementCompute(1) - params_.recip_tau;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute bn_out = params_.alpha * conv_out[i] + bn_bias[i];

      ElementCompute v = params_.state_ptr ? params_.state_ptr[linear_idx + i]
                                           : ElementCompute(0);
      // LIF: h = (1 - 1/tau) * v + (1/tau) * bn_out
      ElementCompute h = one_sub_recip * v + params_.recip_tau * bn_out;

      ElementCompute spike = (h >= params_.v_threshold)
                             ? ElementCompute(1) : ElementCompute(0);

      ElementCompute v_new = (ElementCompute(1) - spike) * h
                             + spike * params_.v_reset;

      if (params_.state_ptr) {
        params_.state_ptr[linear_idx + i] = v_new;
      }
      result[i] = ElementOutput(spike);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const &accumulator,
    int linear_idx = 0) const {

    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> acc_converter;
    FragmentCompute conv_out = acc_converter(accumulator);
    FragmentOutput result;
    ElementCompute one_sub_recip = ElementCompute(1) - params_.recip_tau;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      ElementCompute bn_out = params_.alpha * conv_out[i] + params_.beta;
      ElementCompute v = params_.state_ptr ? params_.state_ptr[linear_idx + i]
                                           : ElementCompute(0);
      ElementCompute h = one_sub_recip * v + params_.recip_tau * bn_out;
      ElementCompute spike = (h >= params_.v_threshold)
                             ? ElementCompute(1) : ElementCompute(0);
      ElementCompute v_new = (ElementCompute(1) - spike) * h
                             + spike * params_.v_reset;
      if (params_.state_ptr) {
        params_.state_ptr[linear_idx + i] = v_new;
      }
      result[i] = ElementOutput(spike);
    }
    return result;
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
