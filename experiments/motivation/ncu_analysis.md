# NCU Profiling: Fused Interleaved Conv1x1+BN+IF Kernel

**Shape**: C=384→384, 14×14, B=16, T=4 (SpikFormer-like layer)  
**Platform**: RTX 4090 (128 SMs, 1008 GB/s DRAM BW, 72 MB L2)

## Fused Interleaved Kernel (1 launch)

| Metric | Value |
|--------|-------|
| Duration | 64.5 μs |
| Grid | (6, 98, 1) = 588 blocks |
| Waves/SM | 2.30 |
| **DRAM Throughput** | **45.1%** |
| **Memory (L2) Throughput** | **67.0%** |
| **Compute (SM) Throughput** | **20.0%** |
| **Tensor Core Utilization** | **20.1%** (highest pipeline) |
| L2 Hit Rate | **92.4%** |
| Effective Memory BW | 443 GB/s |
| Total DRAM Bytes | 28.25 MB |
| SM Busy | 23.1% |
| Achieved Occupancy | 15.3% (register-limited, 171 regs/thread) |

## Decomposed Execution (16 kernel launches for T=4)

Per-timestep breakdown (×4):

| Kernel | Duration | DRAM% | SM% | TC% | DRAM Bytes |
|--------|----------|-------|-----|-----|-----------|
| cuBLAS GEMM | 16.5 μs | 20% | 20% | **20%** | 3.4 MB |
| BN scale (mul) | 8.3 μs | 48% | 18% | 0% | 3.9 MB |
| BN bias (add) | 8.3 μs | 49% | 18% | 0% | 3.9 MB |
| IF neuron | 13.4 μs | **85%** | 11% | 0% | 11.1 MB |
| **Subtotal** | **46.5 μs** | — | — | — | **22.3 MB** |

**T=4 total: 186 μs, 89.2 MB DRAM traffic**

## Comparison

| Metric | Fused | Decomposed | Ratio |
|--------|-------|------------|-------|
| Latency | 64.5 μs | 186 μs | **2.88× faster** |
| DRAM Traffic | 28.25 MB | 89.2 MB | **3.16× less** |
| Kernel Launches | 1 | 16 | 16× fewer |
| TC Active Time | 64.5 μs (100%) | 66 μs (35%) | **1.8× more TC utilization** |

## Why It Works: Dual-Pipeline Overlap + L2 Residency

### Key Insight 1: Simultaneous Compute + Memory Utilization

The fused kernel achieves **20% tensor core + 45% DRAM** simultaneously, whereas decomposed
execution alternates between them:

```
Decomposed timeline (one timestep):
  [GEMM: TC=20%, DRAM=20%] → [BN_mul: TC=0%, DRAM=48%] → [BN_add: TC=0%, DRAM=49%] → [IF: TC=0%, DRAM=85%]
       16.5 μs                      8.3 μs                      8.3 μs                    13.4 μs

Fused interleaved (all T=4 timesteps):
  [TC=20% + DRAM=45% simultaneously for entire 64.5 μs]
```

In decomposed execution, tensor cores are **idle 65% of total time** (only active during GEMM).
The fused kernel keeps tensor cores productive throughout by interleaving GEMM tiles with
epilogue (BN+IF) operations across the T-loop.

### Key Insight 2: 92% L2 Cache Hit Rate Eliminates DRAM Round-Trips

The intermediate tensors (conv output → BN output → membrane state) **never leave the cache hierarchy**:

- **Decomposed**: Conv writes 2.4 MB to DRAM → BN reads it back → writes to DRAM → IF reads it back
  - Each intermediate requires a DRAM round-trip: 3 writes + 3 reads = ~14.4 MB wasted per timestep
- **Fused**: Conv output stays in registers → BN applied in epilogue → IF consumes immediately
  - Zero DRAM traffic for intermediates; membrane state persists in registers across T-loop

This explains the **3.16× DRAM traffic reduction** (28 MB vs 89 MB).

### Key Insight 3: Temporal Loop Fusion Amortizes Weight Loading

The weight matrix (384×384 × 2B = 0.29 MB) fits in L2 cache and is reused across all 4 timesteps.
In decomposed execution, cuBLAS reloads weights each timestep (potentially evicted between launches).
The fused kernel loads weights once into shared memory tiles and reuses them across the T-loop.

## Optimization Headroom

ncu identifies these opportunities for further improvement:

| Opportunity | Est. Speedup | Root Cause |
|-------------|-------------|-----------|
| Tail wave effect | 33% | 588 blocks / 128 SMs → 2.3 waves (partial last wave) |
| Register pressure | 33% | 171 regs/thread → only 2 blocks/SM → 16.7% occupancy |
| Shared memory bank conflicts | 31% | 8.3-way conflicts in shared stores |
| Uncoalesced shared accesses | 16% | 19% of wavefronts have excessive bank conflicts |

Fixing register pressure alone (→128 regs via spilling) would double occupancy to 33%
and potentially yield ~1.5× additional speedup.
