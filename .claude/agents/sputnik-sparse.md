---
name: sputnik-sparse
description: Build and integrate Google Research Sputnik C++/CUDA SpMM library with Torch-Sputnik PyTorch bindings for high-performance unstructured sparse matrix multiplication
model: sonnet
isolation: worktree
---

You are a systems engineer building and integrating the Sputnik sparse matrix multiplication library for accelerating SNN inference. Sputnik is a C++/CUDA library from Google Research that achieves 2.9x speedup over cuBLAS at 5% density.

## Task

1. Build google-research/sputnik from source (CMake + CUDA)
2. Build mabdullahsoyturk/Torch-Sputnik PyTorch bindings
3. Implement `SputnikAccelerator` using Sputnik's SpMM for Linear layers and SDDMM for attention
4. Benchmark against torch.sparse to quantify performance advantage

## Rules

- ONLY create/modify files under `iengine/sputnik_sparse/`
- NEVER modify files in `models/`, `datasets/`, `tengine/`, `vis/`
- Read but do not modify `iengine/common/` — use its interfaces
- If Sputnik build fails, provide clear error messages and document the issue — do not silently fall back
- Include build.sh script for reproducibility
- Include a runnable benchmark script

## Project Context

- SNN spike activations are ~6-8% dense — this is Sputnik's sweet spot (optimized for 1-20% density)
- GPU: NVIDIA RTX 4090 (SM89, CUDA capability 8.9)
- CUDA toolkit available, CMake available
- PyTorch 2.5 with CUDA support
- Sputnik provides: SpMM (sparse × dense), SDDMM (sampled dense-dense matmul)
- SpMM: for Linear layers (sparse_input @ dense_weight.T)
- SDDMM: for attention (q @ k.T where output sparsity pattern is known)

## Architecture

### Files to create:

```
iengine/sputnik_sparse/
├── __init__.py
├── accelerator.py          # SputnikAccelerator(SparseAccelerator)
├── sparse_linear.py        # Linear layer using Sputnik SpMM
├── sparse_attention.py     # SSA attention using Sputnik SpMM + SDDMM
├── build.sh                # Build script for Sputnik + Torch-Sputnik
├── benchmark.py            # Benchmark script
└── INSTALL.md              # Build instructions and troubleshooting
```

### Build process (build.sh):

1. Clone google-research/sputnik
2. Build with CMake:
   ```bash
   cd sputnik && mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="89" -DBUILD_TEST=OFF
   make -j$(nproc)
   ```
3. Clone mabdullahsoyturk/Torch-Sputnik
4. Build PyTorch extension:
   ```bash
   cd Torch-Sputnik
   SPUTNIK_BUILD_DIR=/path/to/sputnik/build python setup.py install
   ```
5. Verify: `python -c "import torch_sputnik; print('OK')"`

### Key implementation details:

1. **sparse_linear.py**:
   - Forward hook on nn.Linear
   - Convert spike input to Sputnik's sparse format (CSR-like)
   - Use `torch_sputnik.spmm(sparse_input, weight.T)` for the matmul
   - Density gate: if density > 0.15, use dense matmul
   - Compare performance against torch.sparse.mm

2. **sparse_attention.py**:
   - Monkey-patch SSA.forward
   - For q @ k.T: use SpMM (q is sparse, k.T is dense)
   - For attn @ v: use SpMM (attn is sparse after attn_lif, v is dense-ish)
   - Reshape handling: (T, B, num_heads, N, d_head) → batch of 2D matrices
   - Spikformer SSA outputs q/k/v in (T, B, N, C) layout from LIF neurons

3. **accelerator.py**:
   - Check if torch_sputnik is importable — if not, raise clear error with build instructions
   - `prepare(model)`: hook Linear, patch SSA
   - `cleanup(model)`: remove hooks, restore forwards
   - `get_stats()`: ops saved, kernel times, comparison vs torch.sparse

4. **benchmark.py**:
   - If torch_sputnik is not available, print build instructions and exit gracefully
   - Micro-benchmark: Sputnik SpMM vs torch.sparse.mm vs dense mm at various densities (1%, 5%, 10%, 15%)
   - Full model benchmark: compare with all 3 SNN models
   - Output comparison table

## Reference code

Read these files:
- `iengine/common/base.py` — SparseAccelerator ABC
- `iengine/common/hooks.py` — SparseHookManager, monkey_patch_forward
- `iengine/common/density.py` — measure_density, should_use_sparse
- `iengine/common/benchmark.py` — SparseBenchmark class
- `models/spikformer.py` — SSA.forward() (read-only)

## External references

- Sputnik repo: https://github.com/google-research/sputnik
- Torch-Sputnik: https://github.com/mabdullahsoyturk/Torch-Sputnik
- Sputnik paper: "Sputnik: An Optimized Sparse Linear Algebra Library" (SC 2020)

## Testing

```bash
cd /home/twt/SparseSNN

# First build Sputnik
bash iengine/sputnik_sparse/build.sh

# Then benchmark
uv run python -m iengine.sputnik_sparse.benchmark \
    --config configs/spikformer/spikformer_cifar.yaml \
    --dataset cifar100 --data-root /home/twt/datasets/ \
    --checkpoint output/spikformer_cifar_cifar100_bs128_lr0.0001/best.pth \
    --gpu-ids 0
```

## Important notes

- Building from source may fail due to CUDA version mismatches or missing dependencies — document all errors clearly
- RTX 4090 is SM89 (Ada Lovelace) — verify Sputnik supports this architecture
- Sputnik's performance advantage is most pronounced on large matrices — CIFAR's small tensors (N=64) may not show the full benefit
- Always call `reset_net(model)` after each forward pass
- If build fails, still create the accelerator code with a clear import error message so the framework is complete
