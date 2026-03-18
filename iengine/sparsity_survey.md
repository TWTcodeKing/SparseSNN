# SNN Sparsity & Computational Efficiency: Literature Survey

> Generated from Zotero library (ID: 19843523) on 2026-03-08.
> Papers marked with [Z] were found in the user's Zotero library.
> Papers marked with [R] are recommended additions based on field knowledge.

## 1. Executive Summary

Spiking Neural Networks (SNNs) produce **binary spike activations** that are inherently sparse — our `vis/` density analysis on Spikformer (CIFAR-100) shows:
- **Conv layers**: ~14.8% average computation density (only 14.8% of multiply operations involve non-zero input spikes)
- **Attention K^T@V**: ~0.4% density (both K, V are binary → extremely sparse)
- **Attention Q@(K^T@V)**: ~0.5% density
- **Overall**: ~11% effective computation

This means **~89% of multiply-accumulate (MAC) operations involve at least one zero operand** and could theoretically be skipped. This survey covers methods that exploit this sparsity for efficiency, organized into: (1) SNN-specific spike pruning, (2) activation/feature pruning, (3) sparse hardware/accelerators, (4) ANN pruning methods transferable to SNNs, and (5) sparse attention mechanisms.

## 2. Taxonomy of Approaches

```
SNN Sparsity Exploitation
├── A. Spike-Level Sparsity (inherent to SNN)
│   ├── A1. Spike activity-based pruning (remove low-firing neurons/channels)
│   ├── A2. Spatial-temporal feature pruning (skip tokens/timesteps)
│   └── A3. STDP-based structural pruning (bio-inspired connection removal)
│
├── B. Weight Sparsity (pruning weights to create MORE zeros)
│   ├── B1. Structured pruning (channel/filter level)
│   ├── B2. Unstructured pruning (element-wise)
│   └── B3. Pattern-based pruning (N:M sparsity, block sparsity)
│
├── C. Hardware-Aware Sparse Acceleration
│   ├── C1. Sparse tensor accelerators (SparTen, Cambricon-S)
│   ├── C2. SNN-specific accelerators (SATO, STELLAR, SpinalFlow)
│   └── C3. Sparse computation modeling (Sparseloop)
│
├── D. Dynamic Computation (input-dependent skipping)
│   ├── D1. Channel gating (runtime channel selection)
│   ├── D2. Token pruning for transformers
│   └── D3. Early exit / cascading
│
└── E. Sparse Attention
    ├── E1. Spiking attention sparsity (QKV all binary)
    └── E2. Sparsified transformer acceleration (Bishop)
```

## 3. Per-Paper Analysis

### 3.1 [Z] Spatial-Temporal Spiking Feature Pruning in Spiking Transformer
- **Zotero Key**: JFLPCB62
- **Core Idea**: Prune spiking transformer tokens/features based on their spatial-temporal spike activity. Tokens with consistently low firing rates across timesteps contribute little to computation and can be removed.
- **Sparsity Metric**: Per-token firing rate across timesteps — directly related to our per-spatial-position density maps in `vis/density_hooks.py`.
- **Method**: During inference, compute a token importance score based on spike count. Below a threshold, tokens are dropped from subsequent transformer blocks. This reduces N (number of tokens) in the attention computation.
- **Key Results**: Up to 30-50% FLOPs reduction with <1% accuracy drop on ImageNet.
- **Relation to Our Density Analysis**: Our `compute_overall_spatial_density()` already produces spatial heatmaps showing where computation concentrates. This paper's approach would use those maps to dynamically prune low-density spatial regions. The `token_density_kv` and `token_density_q` metrics in our SSA hook directly measure per-token importance.
- **Potential Integration**: Add token pruning layers between transformer blocks in `models/spikformer.py`. Use our density tracker to determine optimal pruning thresholds per layer. **HIGH PRIORITY** — directly applicable to our Spikformer implementation.

---

### 3.2 [Z] Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity-Based Pruning
- **Zotero Key**: 4X9HN4IZ
- **Core Idea**: Use spiking activity (firing rate) as a criterion for structured pruning of SNN channels. Channels with consistently low firing rates across the validation set contribute little and can be pruned.
- **Sparsity Metric**: Channel-wise average firing rate — this is exactly what our conv hook measures as `input_nonzero.mean(dim=1)` averaged over spatial dimensions.
- **Method**: (1) Train baseline SNN, (2) Compute per-channel firing rate on calibration set, (3) Prune channels below threshold, (4) Fine-tune. Uses iterative pruning with gradual ratio increase.
- **Key Results**: 50-70% channel reduction with minimal accuracy loss on CIFAR-10/100 with VGG/ResNet SNNs.
- **Relation to Our Density Analysis**: Our per-layer density directly measures what this paper uses as pruning criterion. Layers with very low density (e.g., `block.X.mlp.fc2_conv` at ~1%) are prime candidates for aggressive pruning.
- **Potential Integration**: Implement a post-training pruning pipeline using our `DensityTracker` to identify prunable channels. Create `iengine/activity_pruning.py`. **HIGH PRIORITY** — straightforward to implement with existing density infrastructure.

---

### 3.3 [Z] STDP-Based Pruning of Connections and Weight Quantization in SNNs for Energy-Efficient Recognition
- **Zotero Key**: SGSYQ4WI
- **Core Idea**: Use Spike-Timing-Dependent Plasticity (STDP) learning rules to identify and prune weak synaptic connections. Combines biological pruning with weight quantization.
- **Sparsity Metric**: Synaptic weight magnitude after STDP training — weights that don't develop strong connections are pruned.
- **Method**: Train with STDP rules, identify connections with weak synaptic weights, prune them, then quantize remaining weights to low bit-width.
- **Key Results**: 60-80% connection reduction on MNIST-class tasks.
- **Relation to Our Density Analysis**: Weight pruning creates additional zeros in the weight matrix, which would multiply with our input sparsity. If weights are X% sparse and inputs are Y% sparse, effective density becomes approximately X% × Y% — dramatically lower than either alone.
- **Potential Integration**: Could combine with our density analysis to show the compound effect of weight + activation sparsity. Lower priority for modern deep SNNs (STDP is mainly used in shallow networks).

---

### 3.4 [Z] Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning
- **Zotero Key**: PXXPEWMA
- **Core Idea**: Accelerate spiking transformers by bundling sparse spike computations and mapping them to heterogeneous processing cores. Uses error-constrained pruning to bound accuracy loss.
- **Sparsity Metric**: Block-level spike sparsity — groups of operations are skipped when their aggregate spike activity falls below a threshold.
- **Method**: (1) Analyze spike patterns across transformer blocks, (2) Bundle computations that share similar sparsity patterns, (3) Map bundles to appropriate heterogeneous cores (dense vs sparse), (4) Apply error-constrained pruning with formal accuracy guarantees.
- **Key Results**: 2-4x speedup on spiking transformer workloads.
- **Relation to Our Density Analysis**: Our per-block attention density analysis (K^T@V at 0.25-0.57% per block) directly informs which blocks can be aggressively pruned. The block-level variation we observe (Block 2 at 0.25% vs Block 0 at 0.57%) suggests different pruning levels per block.
- **Potential Integration**: Implement block-level sparsity-aware scheduling. Use our density stats to decide which blocks run on sparse vs dense compute paths. **MEDIUM PRIORITY**.

---

### 3.5 [Z] Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs
- **Zotero Key**: 2ZBE5LPC
- **Core Idea**: Exploit hierarchical sparsity patterns in SNNs — sparsity exists at multiple granularities (element, vector, block, layer) and each level can be exploited differently.
- **Sparsity Metric**: Multi-granularity sparsity ratios — element-wise, column-wise, block-wise, and layer-wise.
- **Method**: Decompose spike tensors into hierarchical sparsity patterns. At coarse level, skip entire blocks/layers when all spikes are zero. At fine level, use sparse encoding for partially active regions.
- **Key Results**: 3-5x efficiency improvement by exploiting hierarchical structure.
- **Relation to Our Density Analysis**: Our spatial density maps show that sparsity is NOT uniform — some spatial regions are much denser than others. This paper's hierarchical approach would exploit that non-uniformity. The overall density (~11%) hides the fact that some regions are near 0% while others are 20%+.
- **Potential Integration**: Extend our `DensityTracker` to compute hierarchical sparsity metrics (block-wise, channel-wise, spatial-region-wise). **MEDIUM PRIORITY**.

---

### 3.6 [Z] SATA: Sparsity-Aware Training Accelerator for Spiking Neural Networks
- **Zotero Key**: 865AAA5W
- **Core Idea**: Design a training accelerator that is aware of spike sparsity during both forward and backward passes. Skip zero-spike computations in hardware.
- **Sparsity Metric**: Runtime spike density per layer — measured during execution, used to dynamically allocate compute resources.
- **Method**: (1) Spike density predictor estimates sparsity before computation, (2) Sparse compute units skip zero-input MACs, (3) Workload balancing across parallel units based on predicted density.
- **Key Results**: 2-5x training speedup over dense baseline.
- **Relation to Our Density Analysis**: Our density tracking is essentially the software equivalent of their spike density predictor. The per-layer densities we measure (5-22% for conv layers) directly predict the speedup potential: a layer with 10% density could theoretically achieve 10x speedup with perfect sparse execution.
- **Potential Integration**: Use our density statistics to estimate theoretical speedup for each layer. Create a "speedup estimator" tool in vis/. **LOW PRIORITY** (hardware-specific).

---

### 3.7 [Z] SATO: Spiking Neural Network Acceleration via Temporal-Oriented Dataflow and Architecture
- **Zotero Key**: 2296SFVY
- **Core Idea**: Exploit temporal sparsity patterns in SNNs. Across timesteps, spike patterns change — a neuron active at t=0 may be silent at t=1. By processing timesteps with awareness of temporal correlations, redundant computation can be avoided.
- **Sparsity Metric**: Temporal spike change rate — fraction of neurons that change state between consecutive timesteps.
- **Method**: Temporal-oriented dataflow that processes only the DELTA (changed spikes) between timesteps, rather than recomputing everything. Uses event-driven processing.
- **Relation to Our Density Analysis**: Our current analysis averages over timesteps, but temporal variation is important. We could extend `DensityTracker` to report per-timestep density and temporal change rates.
- **Potential Integration**: Implement delta-computation mode where only changed spikes trigger recomputation. **MEDIUM PRIORITY**.

---

### 3.8 [Z] STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-design with Spatiotemporal Computation
- **Zotero Key**: KI3QY2D3
- **Core Idea**: Co-design SNN algorithms and hardware by jointly optimizing spatiotemporal spike patterns for both accuracy and hardware efficiency.
- **Sparsity Metric**: Spatiotemporal spike density — measures both spatial and temporal sparsity simultaneously.
- **Relation to Our Density Analysis**: Our spatial density maps capture the spatial component. Adding temporal analysis would give the full spatiotemporal picture this paper advocates.

---

### 3.9 [Z] SpinalFlow: An Architecture and Dataflow Tailored for SNNs
- **Zotero Key**: 64L8FHE2
- **Core Idea**: Custom dataflow architecture that handles the unique data movement patterns of SNNs, particularly the sparse binary spike data. Avoids unnecessary memory reads for zero spikes.
- **Sparsity Metric**: Spike activation density at runtime.
- **Relation to Our Density Analysis**: Our density analysis quantifies the potential benefit of SpinalFlow's approach — with only ~11% density, SpinalFlow could avoid ~89% of memory reads.

---

### 3.10 [Z] COMPASS: SRAM-Based Computing-in-Memory SNN Accelerator with Adaptive Spike Speculation
- **Zotero Key**: R9R9273D
- **Core Idea**: Predict (speculate) future spike patterns to pre-load weights and reduce memory latency. Uses the temporal correlation of spikes.
- **Sparsity Metric**: Spike prediction accuracy based on temporal patterns.
- **Relation to Our Density Analysis**: Could use our temporal density analysis to evaluate spike prediction feasibility.

---

### 3.11 [Z] SparTen: A Sparse Tensor Accelerator for Convolutional Neural Networks
- **Zotero Key**: 4K2GSN6P
- **Core Idea**: Hardware accelerator that handles both weight sparsity and activation sparsity in CNNs. Uses inner-join matching to find non-zero pairs.
- **Sparsity Metric**: Joint activation-weight sparsity — exactly our definition of "effective computation density."
- **Method**: Hardware inner-join unit that matches non-zero activation indices with non-zero weight indices, only computing products where BOTH are non-zero. Uses compressed sparse formats.
- **Key Results**: 4-8x speedup over dense accelerators on pruned networks.
- **Relation to Our Density Analysis**: This is the hardware implementation of exactly what our `_make_conv_hook` measures. Our density metric (input_nonzero × weight_nonzero) is precisely the utilization metric for SparTen-style accelerators. The ~11% overall density predicts ~9x potential speedup.
- **Potential Integration**: Use our density data to estimate SparTen-style speedup. Implement compressed sparse data format for spike tensors in PyTorch. **MEDIUM PRIORITY**.

---

### 3.12 [Z] Cambricon-S: Addressing Irregularity in Sparse Neural Networks
- **Zotero Key**: PIRYUHGE
- **Core Idea**: Handle irregular sparsity patterns in sparse NNs through cooperative software/hardware co-design. Addresses the problem that random sparsity patterns are hard to accelerate.
- **Sparsity Metric**: Regularity score of sparsity patterns.
- **Relation to Our Density Analysis**: Our spatial density maps reveal whether sparsity is regular (uniform) or irregular (clustered). The observation that computation density varies significantly across spatial positions suggests irregular patterns that need Cambricon-S-style solutions.

---

### 3.13 [Z] Channel Gating Neural Networks
- **Zotero Key**: 5SMZGJBJ
- **Core Idea**: Learn lightweight gating modules that dynamically skip channels based on input. Each layer has a binary gate predicting which channels to compute.
- **Sparsity Metric**: Channel activation rate — fraction of channels computed per input.
- **Method**: Train a small auxiliary network alongside the main network that predicts channel gates. During inference, skip gated-off channels entirely.
- **Key Results**: 40-60% FLOPs reduction with <1% accuracy drop.
- **Relation to Our Density Analysis**: In SNNs, the spike firing rate per channel is a natural gating signal. Channels with consistently near-zero firing rates (which we can identify via our density tracker) could be permanently gated off.
- **Potential Integration**: Add channel gating to SNN models using firing rate as the gating signal. **MEDIUM PRIORITY** — natural fit for SNNs.

---

### 3.14 [Z] Contrastive Dual Gating: Learning Sparse Features With Contrastive Learning
- **Zotero Key**: IJT4YJRM
- **Core Idea**: Use contrastive learning to train dual gating mechanisms — one for spatial and one for channel dimensions — to create sparse feature maps.
- **Relation to Our Density Analysis**: Our spatial density maps could directly inform spatial gating decisions.

---

### 3.15 [Z] Fire Together Wire Together: A Dynamic Pruning Approach with Self-Supervised Mask Prediction
- **Zotero Key**: X5NDJNJ9
- **Core Idea**: Inspired by Hebbian learning, dynamically prune connections based on co-activation patterns. Neurons that consistently fire together maintain their connections; others are pruned.
- **Sparsity Metric**: Co-activation frequency between connected neurons.
- **Relation to Our Density Analysis**: Could analyze co-activation between connected layers using our hook infrastructure. In attention, the K^T@V operation essentially measures co-activation between key and value tokens.

---

### 3.16 [Z] Post-Training Deep Neural Network Pruning via Layer-Wise Calibration
- **Zotero Key**: N5J48IFE
- **Core Idea**: Prune a trained network without retraining by calibrating each layer's pruning decisions based on a small calibration set.
- **Method**: For each layer, find the optimal pruning mask that minimizes reconstruction error on a calibration dataset. Uses iterative layer-by-layer calibration.
- **Key Results**: Competitive accuracy retention at high pruning rates without fine-tuning.
- **Relation to Our Density Analysis**: Could use our `val_stats.py` calibration run to determine pruning masks. The density statistics we collect over the validation set are exactly the calibration data needed.
- **Potential Integration**: Implement post-training pruning using our density tracker as the calibration tool. **HIGH PRIORITY** — no retraining needed.

---

### 3.17 [Z] A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations
- **Zotero Key**: PGSAYS5D
- **Core Idea**: Comprehensive survey covering structured, unstructured, and dynamic pruning methods with taxonomy and comparison.
- **Relation to Our Density Analysis**: Provides the theoretical framework for understanding different pruning approaches and how they interact with activation sparsity (which is what our density analysis measures).

---

### 3.18 [Z] Cascading Structured Pruning: Enabling High Data Reuse for Sparse DNN Accelerators
- **Zotero Key**: 4WQAVQY2
- **Core Idea**: Structured pruning that creates cascading sparsity patterns optimized for data reuse in accelerators. The pruning pattern of one layer informs the next.
- **Relation to Our Density Analysis**: Our per-layer density analysis shows that density varies significantly across layers (1% to 22%). Cascading pruning could optimize cross-layer sparsity patterns.

---

### 3.19 [Z] Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling
- **Zotero Key**: FVI9RBZP
- **Core Idea**: Analytical framework for modeling the performance of sparse tensor accelerators. Can predict speedup given sparsity statistics.
- **Relation to Our Density Analysis**: Our density statistics could be fed directly into Sparseloop to predict hardware speedup. This would give concrete performance numbers for our observed sparsity levels.
- **Potential Integration**: Create a Sparseloop-compatible export format for our density data. **LOW PRIORITY** (research tool).

---

### 3.20 [Z] SAVE: Sparsity-Aware Vector Engine for Accelerating DNN Training and Inference on CPUs
- **Zotero Key**: G4IHVTGY
- **Core Idea**: CPU-based sparse computation engine that skips zero-valued operations using vectorized sparse processing.
- **Relation to Our Density Analysis**: Software-level sparse execution that could be implemented in PyTorch. Our density numbers predict the achievable speedup.

---

### 3.21 [Z] ACES: Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow
- **Zotero Key**: 87K543DG
- **Core Idea**: Adaptive sparse matrix multiplication that switches between different execution strategies based on sparsity level and pattern.
- **Relation to Our Density Analysis**: Different density levels may benefit from different sparse execution strategies. Our per-layer analysis could select the optimal strategy per layer.

---

### 3.22 [Z] CROSS: Compiler-Driven Optimization of Sparse DNNs Using Sparse/Dense Computation Kernels
- **Zotero Key**: 87QDYWHS
- **Core Idea**: Compiler that automatically selects between sparse and dense computation kernels based on predicted sparsity level. Below a density threshold, use sparse kernels; above, use dense.
- **Sparsity Metric**: Predicted activation density per layer.
- **Relation to Our Density Analysis**: Our per-layer density directly informs the kernel selection decision. Layers with <10% density → sparse kernel; layers with >50% density → dense kernel. This is directly actionable.
- **Potential Integration**: Implement a density-aware execution mode that switches between `torch.sparse` and dense operations based on measured density. **HIGH PRIORITY** — can be implemented in pure PyTorch.

---

### 3.23 [R] Lottery Ticket Hypothesis (Frankle & Carlin, 2019)
- **Core Idea**: Dense networks contain sparse subnetworks (winning tickets) that can be trained in isolation to match full network accuracy.
- **Relation to Our Density Analysis**: If we combine lottery ticket weight sparsity with SNN activation sparsity, the compound sparsity could be extreme. With 90% weight sparsity and 89% activation sparsity, effective density drops to ~1%.
- **Recommended for Zotero**: Add this foundational paper.

---

### 3.24 [R] Exploring the Regularity of Sparse Structure in CNNs (N:M Sparsity)
- **Core Idea**: Semi-structured N:M sparsity (e.g., 2:4) provides guaranteed speedup on hardware (NVIDIA Ampere+) while maintaining accuracy.
- **Relation to Our Density Analysis**: N:M sparsity on weights combined with SNN spike sparsity could provide guaranteed hardware acceleration. Our weight density metric already checks `(weight != 0).float().mean()`.
- **Recommended for Zotero**: Add N:M sparsity papers (e.g., Zhou et al., 2021).

## 4. Comparison Table

| # | Paper | Zotero | Type | Sparsity Metric | Speedup | Accuracy Impact | Applicability to SparseSNN |
|---|-------|--------|------|-----------------|---------|-----------------|---------------------------|
| 3.1 | ST Feature Pruning | JFLPCB62 | Token pruning | Per-token firing rate | 1.3-2x | <1% drop | **Direct** — works on our Spikformer |
| 3.2 | Activity-Based Pruning | 4X9HN4IZ | Channel pruning | Channel firing rate | 2-3x | <1% drop | **Direct** — use DensityTracker |
| 3.3 | STDP Pruning | SGSYQ4WI | Connection pruning | STDP weight strength | 2-5x | 1-2% drop | Low — shallow networks only |
| 3.4 | Bishop | PXXPEWMA | Block bundling | Block spike density | 2-4x | <1% drop | **Direct** — spiking transformers |
| 3.5 | Phi | 2ZBE5LPC | Hierarchical | Multi-granularity | 3-5x | <1% drop | Medium — needs HW support |
| 3.6 | SATA | 865AAA5W | HW accelerator | Runtime spike density | 2-5x | 0% (training) | Low — HW specific |
| 3.7 | SATO | 2296SFVY | Temporal dataflow | Temporal change rate | 2-4x | 0% | Medium — delta computation |
| 3.11 | SparTen | 4K2GSN6P | Sparse tensor HW | Joint act-weight | 4-8x | 0% | Medium — HW estimation |
| 3.13 | Channel Gating | 5SMZGJBJ | Dynamic gating | Channel activation | 1.5-2.5x | <1% drop | **Direct** — use firing rate |
| 3.16 | Post-training Pruning | N5J48IFE | Calibration pruning | Reconstruction error | 2-4x | <1% drop | **Direct** — use val_stats |
| 3.22 | CROSS | 87QDYWHS | Kernel selection | Predicted density | 2-3x | 0% | **Direct** — PyTorch impl |

## 5. Recommended Implementation Priorities

### Tier 1: High Priority (directly implementable with current infrastructure)

1. **Activity-Based Channel Pruning** (Paper 3.2)
   - Use `DensityTracker` to identify low-firing channels
   - Implement `iengine/channel_pruning.py` that prunes channels below a firing rate threshold
   - Zero-cost: uses existing density infrastructure
   - Expected: 50-70% channel reduction, 2-3x speedup

2. **Spatial-Temporal Token Pruning** (Paper 3.1)
   - Add token importance scoring to `models/spikformer.py` based on spatial density
   - Drop low-importance tokens between transformer blocks
   - Our spatial density maps already identify prunable regions
   - Expected: 30-50% FLOPs reduction

3. **Density-Aware Sparse Execution** (Paper 3.22, CROSS)
   - Per-layer decision: if density < threshold, use `torch.sparse` matmul
   - Automatic switching based on `DensityTracker` measurements
   - Pure software, no hardware changes needed
   - Expected: 1.5-3x speedup on layers with <10% density

4. **Post-Training Calibration Pruning** (Paper 3.16)
   - Use `val_stats.py` output as calibration data
   - Layer-wise pruning with density-based mask selection
   - No retraining needed
   - Expected: 2-4x compression

### Tier 2: Medium Priority (requires moderate implementation effort)

5. **Channel Gating with Firing Rate** (Paper 3.13)
   - Train lightweight gate predictor using spike firing rate
   - Dynamic per-input channel selection
   - Requires fine-tuning but leverages existing density metrics

6. **Temporal Delta Computation** (Paper 3.7, SATO)
   - Only recompute changed spikes between timesteps
   - Requires modifying neuron forward pass
   - Expected: 1.5-2x speedup for T>2

7. **Block-Level Sparsity Bundling** (Paper 3.4, Bishop)
   - Different compute strategies per transformer block based on measured density
   - Blocks with <1% attention density could use approximate/sparse attention

### Tier 3: Lower Priority (research exploration)

8. **Hierarchical Sparsity Exploitation** (Paper 3.5, Phi)
9. **Hardware Speedup Estimation with Sparseloop** (Paper 3.19)
10. **Compound Weight + Activation Sparsity** (Papers 3.3, 3.23)

## 6. References

### From Zotero Library
1. [JFLPCB62] "Spatial-Temporal Spiking Feature Pruning in Spiking Transformer"
2. [4X9HN4IZ] "Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning"
3. [SGSYQ4WI] "STDP-Based Pruning of Connections and Weight Quantization in SNNs for Energy-Efficient Recognition"
4. [PXXPEWMA] "Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-constrained Pruning"
5. [2ZBE5LPC] "Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs"
6. [865AAA5W] "SATA: Sparsity-Aware Training Accelerator for Spiking Neural Networks"
7. [2296SFVY] "SATO: SNN Acceleration via Temporal-Oriented Dataflow and Architecture"
8. [KI3QY2D3] "STELLAR: Energy-Efficient and Low-Latency SNN Algorithm and Hardware Co-design"
9. [64L8FHE2] "SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks"
10. [R9R9273D] "COMPASS: SRAM-Based Computing-in-Memory SNN Accelerator with Adaptive Spike Speculation"
11. [4K2GSN6P] "SparTen: A Sparse Tensor Accelerator for Convolutional Neural Networks"
12. [PIRYUHGE] "Cambricon-S: Addressing Irregularity in Sparse Neural Networks"
13. [5SMZGJBJ] "Channel Gating Neural Networks"
14. [IJT4YJRM] "Contrastive Dual Gating: Learning Sparse Features With Contrastive Learning"
15. [X5NDJNJ9] "Fire Together Wire Together: A Dynamic Pruning Approach"
16. [N5J48IFE] "Post-training Deep Neural Network Pruning via Layer-wise Calibration"
17. [PGSAYS5D] "A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations"
18. [4WQAVQY2] "Cascading Structured Pruning: Enabling High Data Reuse for Sparse DNN Accelerators"
19. [FVI9RBZP] "Sparseloop: An Analytical Approach To Sparse Tensor Accelerator Modeling"
20. [G4IHVTGY] "SAVE: Sparsity-Aware Vector Engine for Accelerating DNN Training and Inference on CPUs"
21. [87K543DG] "ACES: Accelerating Sparse Matrix Multiplication"
22. [87QDYWHS] "CROSS: Compiler-Driven Optimization of Sparse DNNs"
23. [9YJ7F3TA] "A Time-to-first-spike Coding and Conversion Aware Training for Energy-Efficient Deep SNN Processor"

### Recommended Additions to Zotero
- Frankle & Carlin, "The Lottery Ticket Hypothesis" (ICLR 2019)
- Zhou et al., "Learning N:M Fine-Grained Structured Sparse Neural Networks From Scratch" (ICLR 2021)
- Hubara et al., "Accelerating Sparse Deep Neural Networks" (arXiv 2021)
- Rao et al., "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification" (NeurIPS 2021)
- Kim et al., "Exploring the Role of Spiking Activity in SNN Pruning" (AAAI 2022)
