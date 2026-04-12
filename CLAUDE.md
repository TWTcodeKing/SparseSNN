# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseSNN is an end-to-end framework for deploying Spiking Neural Networks (SNNs) on hardware accelerators. It addresses two fundamental bottlenecks that arise from SNNs' unique temporal dimension T — both rooted in the same cause: the mismatch between SNN temporal semantics and hardware execution models designed for static ANNs.

### The Unified Bottleneck: SNN Temporal Dimension

SNNs process input over T timesteps, producing binary spike tensors of shape `(T, B, C, H, W)`. This extra temporal dimension T creates two deployment bottlenecks that are **two manifestations of the same root cause**:

**Bottleneck 1: Neuron Temporal Loop.** Each spiking neuron (LIF/IF) maintains membrane potential state across T timesteps: `v[t] = f(v[t-1], x[t])`. Standard inference engines unroll this into T separate element-wise kernel launches (charge, fire, reset) per neuron layer. For a model with L neuron layers, this produces `O(T*L)` kernel launches with no data reuse — the dominant cost at 74% of total inference time (measured on SpikingResformer-S via TRT CUPTI profiling).

**Bottleneck 2: 4D/5D Data Reformat.** Hardware accelerators (GPU tensor cores, NPU, ANE) operate on 4D spatial tensors `(N, C, H, W)` in hardware-preferred layouts (e.g., NHWC). But SNN neurons and spike-driven attention require the temporal dimension explicit as 5D `(T, B, C, H, W)`. Every transition between spatial ops (Conv/GEMM in 4D) and temporal ops (neurons/attention in 5D) forces a data layout rearrangement — measured at 1.2 seconds per inference on TRT (35% of total time). This cost is cross-platform: on edge NPUs (ARM Ethos-U85, Apple ANE) it manifests as accelerator-to-CPU fallback; on FPGA as DMA stalls.

Both bottlenecks vanish for standard ANNs because ANNs have no temporal dimension. They are unique to SNNs and represent a structural gap between SNN computation models and existing hardware/software inference stacks.

### Two-Layer Solution

**Layer 1 (Algorithm): Structured Sparsity Conversion** — Convert dense SNN weights into hardware-accelerated structured sparse formats (2:4, N:M) using activation-aware, calibration-only (training-free) methods. The core insight: SNN binary spike activations exhibit emergent regularity (pattern clustering, row-subset reuse) that guides where weight sparsity can be introduced with minimal accuracy loss. This is hardware-agnostic — each platform's sparsity constraint is a mathematical input to the optimization.

**Layer 2 (System): Temporal Dimension Lowering (TDL)** — A domain-specific compiler pass that systematically lowers the SNN temporal dimension from an explicit tensor axis into implicit operator state, bridging the semantic gap between SNNs' temporal IR `(T, B, C, H, W)` and the spatial IR `(N, C, H, W)` that all hardware compilers expect. TDL does not replace TRT/CoreML/Vela — it transforms SNN graphs into a canonical form these compilers can optimize as efficiently as ANN graphs.

#### The Stateless/Stateful Operator Dichotomy

The key theoretical insight behind TDL: SNN operators cleanly partition into two categories with respect to the temporal dimension T.

**Stateless operators** (Conv2d, BN, Linear, MaxPool, MatMul, etc.) have no temporal dependency — their output at timestep t depends only on input at timestep t. They are **temporally independent**: `f(x[0], x[1], ..., x[T-1]) = [f(x[0]), f(x[1]), ..., f(x[T-1])]`. These operators can process all T timesteps as a single batched 4D operation `f(x_batched)` where the T dimension is absorbed into the batch. This is exactly what `SeqToANNContainer` does in PyTorch — but existing inference compilers do not recognize this pattern, leaving explicit 5D→4D→5D reshape nodes that break format propagation.

**Stateful operators** (LIF/IF spiking neurons) carry membrane potential across timesteps: `v[t] = g(v[t-1], x[t])`. The output at timestep t depends on all previous timesteps through the recurrent state. These operators **cannot** be naively batched across T — the sequential dependency is real. However, the state is local to each spatial position (each neuron evolves independently), enabling a fused kernel where one GPU thread handles all T steps for one spatial neuron.

This dichotomy is fundamental: **stateless operators should see T*B as batch (4D), while stateful operators should see T as a loop bound (kernel attribute)**. No existing framework makes this distinction explicit.

#### Three TDL Transformations

**TDL-1: T-Axis Absorption.** For all stateless operators, systematically rewrite the graph so data stays in 4D `(T*B, C, H, W)` throughout. This is not a naive reshape — TDL must prove each operator is temporally independent and that the absorption preserves semantic equivalence. The T dimension disappears from the tensor shape entirely for these operators.

**TDL-2: Stateful Operator Extraction.** For spiking neurons (the only stateful operators in standard SNNs), extract the temporal recurrence into a fused kernel where T becomes an operator attribute, not a tensor dimension. The kernel receives 4D input `(T*B, C, H, W)`, internally iterates `t=0..T-1` with stride `N = numel/T` to process each timestep, and emits 4D output. The temporal semantics (membrane potential dynamics, spike generation) are fully preserved inside the kernel, invisible to the host compiler. This reduces `O(T*L)` kernel launches to `O(L)` — an 11.8x measured speedup on neuron computation.

**TDL-3: Temporal Attention Decomposition.** Spike-driven self-attention (DSSA, etc.) appears to require explicit T for cross-timestep interaction. TDL proves this is not the case: the attention matmul `Q^T K` operates **independently per frame** (temporally decomposable), and the only truly temporal ops within attention (neuron membrane dynamics, firing rate statistics) are already handled by TDL-2. The firing rate buffers are frozen constants at inference time (no cross-timestep reduction needed). This allows the entire attention block to be flattened to 4D — the most impactful transformation, eliminating 1254ms of measured reformat overhead.

### GPU Profiling Insights (ncu/nsys, RTX 4090)

Profiled SEW-ResNet-34 (B=16, T=4, FP16) and SEW-ResNet-152 (B=32, T=4, FP16) with nsys and ncu. Key findings:

**Neuron kernels (`__myl_*`) are severely memory-bound:**
- SM Busy: 6%, Issue Slots Busy: 5.79% — GPU is 94% idle during neuron execution
- L1 cache hit rate: 0%, L2 hit rate: ~50% — pure streaming, no data reuse
- Stall Long Scoreboard: >100 cycles — warps blocked waiting for DRAM loads
- Arithmetic intensity: ~0.125 FLOP/byte (2,640x below the GPU's ridge point)
- Already on the memory roofline ceiling — hardware-limited, no kernel-level optimization possible

**Conv kernels (`sm80_xmma_fprop_implicit_gemm`) are well-utilized:**
- Use FP16 tensor cores with cuDNN's hand-tuned `_trt` exclusive kernels
- L2 hit rate: 97.83% — working set fits in L2 cache
- Shared memory used for im2col tiles (566 MB traffic per inference)
- At B=16, slightly memory-bound (below compute ceiling on roofline) due to insufficient parallelism

**Reformat overhead is the dominant bottleneck (not neuron compute):**
- Baseline 5D: 63 reformat layers consuming 1.06 ms (SEW-ResNet-34)
- Native 4D: 1 reformat layer consuming 0.09 ms — **92% reduction**
- This single change (TDL-1) gives 1.78x speedup on SEW-ResNet-152

**Conv+IF fusion via cuDNN Backend Graph API: proven feasible but impractical:**
- cuDNN 8.9.7's graph API can fuse Conv+BN+IF pointwise ops (Add, CmpGE, BinarySelect) into one kernel
- Virtual tensors keep conv_out in registers — verified with test_graph.cpp
- **But**: cuDNN's public API uses Cutlass-based kernels, not TRT's hand-tuned `_trt` kernels
- The inferior Conv kernels negate the fusion benefit
- TRT plugins are opaque to Myelin — break cross-layer fusion and format propagation
- **Conclusion**: the correct approach is to express IF neuron as standard ONNX element-wise ops and let TRT/Myelin handle optimization natively

**The remaining open problem: memory-bound neuron dynamics.**
In ANNs, TRT fuses Conv+BN+ReLU into one kernel — ReLU never materializes to DRAM. In SNNs, TRT does not fuse Conv with IF neuron element-wise ops because cuDNN's Conv epilogue only supports fixed activation types (ReLU, tanh, etc.), not custom stateful operations. The IF neuron ops remain as separate Myelin kernels that reload conv_out from DRAM. Solving this — making Conv→IF fusion work with TRT's `_trt` kernels — would bring SNN inference performance to ANN parity. This likely requires either NVIDIA adding IF/LIF as a cuDNN activation type, or a way to inject custom epilogues into TRT's internal Conv kernels.

### Benchmark Results

```
SEW-ResNet-152  B=32  T=4  FP16  RTX 4090  ImageNet 224x224
─────────────────────────────────────────────────────────────────
5D baseline (dense):           70.81 ms    452 img/s    1.00x
5D + 2:4 sparse:               71.30 ms    449 img/s    0.99x
4D native (dense):             39.76 ms    805 img/s    1.78x
4D native + 2:4 sparse:        37.63 ms    850 img/s    1.88x
─────────────────────────────────────────────────────────────────

SEW-ResNet-34  B=16  T=4  FP16  RTX 4090  ImageNet 224x224
─────────────────────────────────────────────────────────────────
5D baseline (dense):            4.99 ms   3206 img/s    1.00x
5D + 2:4 sparse:                4.76 ms   3361 img/s    1.05x
4D native (dense):              3.67 ms   4360 img/s    1.36x
4D native + 2:4 sparse:         3.31 ms   4834 img/s    1.51x
─────────────────────────────────────────────────────────────────
```

## Environment & Commands

Uses `uv` for dependency management. Python virtualenv at `.venv/`. Datasets live at `/data/twt/datasets/`.

### SBC pruning pipeline (the main sparsification method)

The `sparse.snn_sbc` pipeline:
1. **Module identification**: finds each Linear/Conv(+BN) → LIF/IF pair
2. **SMP Hessian collection**: H_SMP = 2·(MX)^T·(MX) where M is the Van Rossum Distance matrix encoding LIF temporal dynamics. `im2col=True` for Conv2d captures cross-spatial correlations.
3. **ExactOBS N:M pruning**: element-wise greedy with per-row H⁻¹ and N:M capacity constraint
4. **BN recalibration**: re-estimates batch norm statistics after pruning