# Sputnik Sparse Backend — Build Instructions

## Overview

The Sputnik backend uses [google-research/sputnik](https://github.com/google-research/sputnik) CUDA kernels for sparse matrix multiplication (SpMM) and sampled dense-dense matrix multiplication (SDDMM), accessed via [Torch-Sputnik](https://github.com/mabdullahsoyturk/Torch-Sputnik) PyTorch bindings.

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090 / SM89)
- **CUDA Toolkit**: 11.x or 12.x (must match your PyTorch CUDA version)
- **CMake**: >= 3.17
- **Python**: 3.8+ with PyTorch installed
- **C++ compiler**: GCC 9+ or compatible
- **git**

Check your CUDA version:
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

These should match (at least the major version).

## Quick Build

```bash
bash iengine/sputnik_sparse/build.sh
```

The script will clone, build, and install everything under `iengine/sputnik_sparse/third_party/`.

### Custom install prefix

```bash
bash iengine/sputnik_sparse/build.sh /path/to/install
```

### Custom CUDA architecture

```bash
CUDA_ARCH=80 bash iengine/sputnik_sparse/build.sh   # A100 (SM80)
CUDA_ARCH=86 bash iengine/sputnik_sparse/build.sh   # RTX 3090 (SM86)
CUDA_ARCH=89 bash iengine/sputnik_sparse/build.sh   # RTX 4090 (SM89, default)
CUDA_ARCH=90 bash iengine/sputnik_sparse/build.sh   # H100 (SM90)
```

## Manual Build

### Step 1: Build Sputnik

```bash
git clone https://github.com/google-research/sputnik.git
cd sputnik
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHS="89" \
    -DBUILD_TEST=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j$(nproc)
```

### Step 2: Build Torch-Sputnik

```bash
git clone https://github.com/mabdullahsoyturk/Torch-Sputnik.git
cd Torch-Sputnik
export SPUTNIK_BUILD_DIR=/path/to/sputnik/build
export TORCH_CUDA_ARCH_LIST="8.9"
pip install -e .
```

### Step 3: Verify

```bash
python -c "import torch_sputnik; print('OK')"
```

## Troubleshooting

### Sputnik CMake fails: "CUDA_ARCHS not recognized"

Older versions of Sputnik may not support newer GPU architectures. Try:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="80" -DBUILD_TEST=OFF
```
or edit `CMakeLists.txt` to add your architecture to the list.

### Sputnik CMake fails: "Could NOT find CUDA" (CUDA 12.x)

CUDA 12.x deprecated `FindCUDA.cmake`. Sputnik's CMakeLists.txt may need patching:
```cmake
# Replace: find_package(CUDA REQUIRED)
# With:
include(FindCUDAToolkit)
find_package(CUDAToolkit REQUIRED)
```

You may also need to set:
```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
```

### Sputnik build fails: "unsupported GPU architecture 'compute_89'"

Your CUDA toolkit doesn't support SM89. Either:
1. Upgrade CUDA toolkit to 11.8+ (for SM89) or 12.0+ (for SM90)
2. Build for a supported architecture: `CUDA_ARCHS="80"`

### Torch-Sputnik setup.py fails: "sputnik/sputnik.h not found"

Make sure `SPUTNIK_BUILD_DIR` points to the Sputnik **build** directory (not source):
```bash
export SPUTNIK_BUILD_DIR=/path/to/sputnik/build
```

The build directory should contain `libsputnik.a` or `libsputnik.so`.

### Torch-Sputnik import fails: "undefined symbol"

CUDA version mismatch between Sputnik build and PyTorch. Rebuild both with the same CUDA:
```bash
nvcc --version                                          # System CUDA
python -c "import torch; print(torch.version.cuda)"    # PyTorch CUDA
```

### Runtime error: "CUDA error: no kernel image is available"

The Sputnik library was built for a different GPU architecture than you're running on. Rebuild with the correct `CUDA_ARCHS` for your GPU.

### Performance is slower than dense

This can happen when:
1. **Density is too high** (>15%): The sparse conversion overhead exceeds savings. Adjust `density_threshold`.
2. **Matrices are too small**: Sparse kernels have fixed overhead. Adjust `min_elements`.
3. **Batch size is small**: Sputnik performs best with larger matrices. Try increasing batch size.
4. **Memory-bound workload**: Small matrices may be memory-bound rather than compute-bound.

## Using Without Sputnik

All Python code handles missing `torch_sputnik` gracefully:
- `SputnikAccelerator` prints a warning and passes through without modification
- `benchmark.py` runs torch.sparse and dense benchmarks even without Sputnik
- Import errors provide clear build instructions

## Architecture Notes

- Sputnik SpMM operates on CSR-format sparse matrices
- Sputnik requires `row_indices` sorted by nnz per row (descending) for load balancing
- For SNN binary spike tensors (~6-8% density), SpMM can provide 2-5x speedup on large matrices
- The attention SpMM benefit comes primarily from sparse Q in Q@K.T (Q is a spike tensor after q_lif)
