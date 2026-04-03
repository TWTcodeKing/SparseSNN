# Structured Sparsity Deployment Platforms

Hardware platforms, software stacks, and deployment pipelines for structured sparse SNN inference.

## Hardware Landscape

### Platforms with 2:4 Structured Sparsity Acceleration

| Platform | Pattern | Conv Support | Precision | Status |
|----------|---------|-------------|-----------|--------|
| **NVIDIA** Ampere+ | 2:4 | GEMM only (Conv via TensorRT internal fusion) | FP16/BF16/INT8/FP8/FP4 | Production |
| **AMD** MI300 | 2:4 | GEMM only | FP16/BF16 | Production (hipSPARSELt) |
| **ARM** Ethos-U85 | 2:4 | **Conv + MatMul natively** | INT8 | Production (edge/IoT) |

### Platforms with Other Structured Sparsity

| Platform | Pattern | Notes |
|----------|---------|-------|
| **Apple** M-series | N:M (m = factor of 16, n/m >= 0.5) | CPU Linear (N:M); ANE skips consecutive zeros |
| **Cerebras** WSE | Any pattern (unstructured) | Natively skips all zeros; cloud-only |
| **FPGA** (AMD/Xilinx) | 2:4, 1:4, 1:3, block, custom | Fully flexible, reconfigurable |
| **Qualcomm** Cloud AI 100 | Unstructured | 2.5x at 80% sparsity |

### Platforms with No Structured Sparsity

| Platform | Status |
|----------|--------|
| Intel AMX/Gaudi | No hardware support |
| Google TPU | SparseCore for embeddings only, MXU is dense |
| Huawei Ascend | Unconfirmed |
| Samsung NPU | Activation sparsity only |

---

## Weight Pruning Constraints

### NVIDIA (TensorRT)

For Conv2d weight `[K, C, R, S]` (NCHW), sparsity is along **C_in per spatial position**:

```python
# Every 4 consecutive input channels must have >= 2 zeros
# Independently for each (output_channel, kernel_row, kernel_col)
for k in range(K):
    for r in range(R):
        for s in range(S):
            for c in range(0, C, 4):
                assert count_nonzero(weights[k, c:c+4, r, s]) <= 2
```

For Linear weight `[out, in]`, every 4 consecutive elements along the reduction axis (in_features) must have >= 2 zeros.

**Important**: Our OBC pruning (im2col-flattened `C_in*Kh*Kw` grouping) is NOT TensorRT-compatible for 3x3 convolutions. 31.3% of groups fail. Need TRT-specific pruning along C_in per spatial position.

### ARM Ethos-U85

Same 2:4 constraint as NVIDIA: every 4 input channels must have >= 2 zeros, per output channel and spatial position.

### Apple (Core ML)

N:M structured sparsity where:
- `m` must be a factor of 16 (e.g., 4, 8, 16)
- `n/m >= 0.5` (at least 50% zeros)
- Valid patterns: 2:4, 3:4, 4:8, 7:8, 8:16, 14:16, etc.

Applied along the reduction dimension for Linear layers on CPU.

---

## Per-Platform Software Stack

### NVIDIA GPU

| Layer | Tool | Purpose |
|-------|------|---------|
| Pruning | `sparse/OBC.py` | Produce 2:4 sparse weights |
| Export | `torch.onnx.export()` | PyTorch to ONNX |
| Compilation | **TensorRT** | ONNX to optimized engine with sparse acceleration |
| Runtime | TensorRT Runtime / Triton Server | Execute the engine |
| Alternative | `torch.sparse.to_sparse_semi_structured` | PyTorch-native, Linear layers only |

**Key API calls:**
```python
import tensorrt as trt

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)  # enable 2:4 sparse TC

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("model.onnx")

engine = builder.build_serialized_network(network, config)
```

**Verification:**
```bash
polygraphy inspect sparsity model.onnx
```

---

### Apple Neural Engine

| Layer | Tool | Purpose |
|-------|------|---------|
| Pruning | `sparse/OBC.py` with N:M (m = factor of 16) | Apple-compatible sparse weights |
| Conversion | **coremltools** | PyTorch to Core ML |
| Sparsity | Pre-pruned weights preserved in export | ANE detects zeros automatically |
| Runtime | Core ML Framework | Inference on ANE/CPU/GPU |
| Profiling | Xcode Instruments (Core ML template) | Measure ANE execution |

**Key API calls:**
```python
import coremltools as ct
import torch

# Export traced model
traced = torch.jit.trace(model, example_input)
ml_model = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_precision=ct.precision.FLOAT16,
)
ml_model.save("model.mlpackage")

# Inference
model = ct.models.MLModel("model.mlpackage")
prediction = model.predict({"input": input_data})
```

**Hardware access:**
| Provider | Cost | Notes |
|----------|------|-------|
| Scaleway | ~EUR 0.10/hr | Cheapest, M1 Mac Mini |
| AWS EC2 | ~$0.65/hr | `mac2-m2.metal` instance |
| MacStadium | ~$60/month | Dedicated Mac Mini M2/M4 |

---

### ARM Ethos-U85

| Layer | Tool | Purpose |
|-------|------|---------|
| Pruning | `sparse/OBC.py` (2:4 along C_in per spatial position) | Same constraint as TensorRT |
| Export path A | PyTorch -> ONNX -> TFLite (`onnx2tf`) -> **Vela** | Most common |
| Export path B | PyTorch -> **ExecuTorch** with ARM delegate | Newer, more direct |
| Compilation | **Vela compiler** (`ethos-u-vela`) | TFLite/TOSA to NPU command stream |
| Runtime | TensorFlow Lite Micro (TFLM) or ExecuTorch | Run on Cortex-M + Ethos-U85 |
| Profiling | Vela output / Arm Virtual Hardware | Cycle-accurate estimation |

**Key CLI:**
```bash
# Compile for Ethos-U85 (256 MAC config)
vela model.tflite \
    --accelerator-config ethos-u85-256 \
    --optimise Performance \
    --output-dir output/

# Vela auto-detects 2:4 sparsity and selects sparse tactics
```

**Export pipeline (PyTorch -> TFLite):**
```bash
# Step 1: PyTorch -> ONNX
python -c "
import torch
model = ...  # load pruned model
torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=13)
"

# Step 2: ONNX -> TFLite (via onnx2tf)
pip install onnx2tf
onnx2tf -i model.onnx -o output/ -oiqt  # with INT8 quantization

# Step 3: TFLite -> Vela optimized
vela output/model_integer_quant.tflite --accelerator-config ethos-u85-256
```

**Hardware access:**
| Method | Cost | Real Silicon? |
|--------|------|---------------|
| Arm Virtual Hardware (AWS AMI) | Free tier / ~$0.10/hr | No (cycle-accurate simulator) |
| Alif Ensemble E7 dev kit | ~$150 one-time | Yes (dual Ethos-U85 NPUs) |

---

### FPGA (AMD/Xilinx)

| Layer | Tool | Purpose |
|-------|------|---------|
| Pruning | `sparse/OBC.py` with custom (n,m) | Any pattern — tune for accuracy |
| Path A: Vitis AI | **Vitis AI** (`vai_q_pytorch` + `vai_c_xir`) | Quantize + compile for DPU overlay |
| Path B: Custom HLS | **Vitis HLS** | Write custom sparse accelerator IP |
| Runtime | PYNQ / Vitis AI Runtime (VART) | Execute on FPGA fabric |

**Vitis AI pipeline:**
```bash
# Quantize
vai_q_pytorch quantize --quant_mode calib --model model.pth --input_fn input_fn

# Compile for target DPU
vai_c_xir -x model.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
```

**Custom HLS for maximum flexibility:**
- Implement sparse GEMM with zero-skipping in C++/HLS
- Read compressed weights (value + index format)
- Any N:M or block pattern
- Full control over datapath and memory

---

## Learning Priority

| Priority | Tool | Time to Learn | Why First |
|----------|------|---------------|-----------|
| 1st | **TensorRT** Python API | 1-2 days | Fastest path to real 2:4 sparse conv acceleration |
| 2nd | **coremltools** | 1 day | Pure Python, simple API, rent a Mac |
| 3rd | **Vela compiler** | 2-3 days | Simple CLI but PyTorch->TFLite pipeline has friction |
| 4th | **Vitis AI** / custom HLS | Already familiar | Existing FPGA expertise |

---

## Minimum Viable Cross-Platform Demo

Same OBC-pruned model deployed on all 4 platforms:

```
                    ┌─────────────┐
                    │  Dense SNN   │
                    │   Model      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   OBC.py     │
                    │  (pruning)   │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐──────────────┐
            │              │              │              │
     ┌──────▼──────┐ ┌────▼────┐  ┌──────▼──────┐ ┌────▼────┐
     │  TensorRT   │ │CoreML   │  │   Vela      │ │Vitis AI │
     │  (NVIDIA)   │ │(Apple)  │  │  (ARM)      │ │ (FPGA)  │
     │  2:4 FP16   │ │N:M FP16 │  │  2:4 INT8   │ │Custom   │
     └─────────────┘ └─────────┘  └─────────────┘ └─────────┘
```

**Key constraint per platform:**

| Platform | Pruning constraint | Export format |
|----------|-------------------|---------------|
| NVIDIA | 2:4 along C_in per (Kh,Kw) | ONNX |
| Apple | N:M (m factor of 16) along reduction dim | Core ML (.mlpackage) |
| ARM | 2:4 along C_in per (Kh,Kw) — same as NVIDIA | TFLite |
| FPGA | Custom (n,m) — optimize for accuracy | XMODEL or custom |

All produced by the same `sparse/OBC.py` optimizer with different `(n, m)` constraint parameters.
