#!/usr/bin/env bash
# Build script for Sputnik + Torch-Sputnik from source.
#
# Tested environment:
#   - CUDA 12.8, GCC 11, PyTorch 2.5 (CUDA 12.1)
#   - GPU: RTX 4090 (SM89 / Ada Lovelace)
#
# Dependencies built automatically if missing:
#   - gflags + glog (installed to ~/.local)
#
# Usage:
#   bash iengine/sputnik_sparse/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${SCRIPT_DIR}/third_party"
LOCAL_PREFIX="$HOME/.local"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_ARCH="${CUDA_ARCH:-89}"
JOBS="${JOBS:-$(nproc)}"

echo "============================================"
echo "  Sputnik + Torch-Sputnik Build Script"
echo "============================================"
echo "  Install prefix: ${PREFIX}"
echo "  CUDA home:      ${CUDA_HOME}"
echo "  CUDA arch:      SM${CUDA_ARCH}"
echo "============================================"

# ── Step 0: Build gflags + glog if not present ──────────────────────────────

if [ ! -f "${LOCAL_PREFIX}/lib/libglog.so" ]; then
    echo ""
    echo "[0a] Building gflags from source..."
    cd /tmp
    [ ! -d gflags ] && git clone --depth 1 --branch v2.2.2 https://github.com/gflags/gflags.git
    cd gflags && mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_TESTING=OFF 2>&1 | tail -3
    make -j"$JOBS" 2>&1 | tail -2
    make install 2>&1 | tail -2

    echo "[0b] Building glog from source..."
    cd /tmp
    [ ! -d glog ] && git clone --depth 1 --branch v0.6.0 https://github.com/google/glog.git
    cd glog && mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
        -DCMAKE_PREFIX_PATH="$LOCAL_PREFIX" \
        -Dgflags_DIR="$LOCAL_PREFIX/lib/cmake/gflags" 2>&1 | tail -3
    make -j"$JOBS" 2>&1 | tail -2
    make install 2>&1 | tail -2
    echo "  gflags + glog installed to ${LOCAL_PREFIX}"
else
    echo "[0] glog already installed at ${LOCAL_PREFIX}/lib/libglog.so"
fi

# ── Step 1: Clone and build Sputnik ──────────────────────────────────────────

SPUTNIK_DIR="${PREFIX}/sputnik"
SPUTNIK_BUILD_DIR="${SPUTNIK_DIR}/build"

if [ ! -d "${SPUTNIK_DIR}" ]; then
    echo ""
    echo "[1/4] Cloning google-research/sputnik..."
    mkdir -p "${PREFIX}"
    git clone https://github.com/google-research/sputnik.git "${SPUTNIK_DIR}"
else
    echo "[1/4] Sputnik source already present"
fi

echo ""
echo "[2/4] Building Sputnik..."
mkdir -p "${SPUTNIK_BUILD_DIR}"
cd "${SPUTNIK_BUILD_DIR}"

if [ ! -f "${SPUTNIK_BUILD_DIR}/sputnik/libsputnik.so" ]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ARCHS="${CUDA_ARCH}" \
        -DBUILD_TEST=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DGLOG_ROOT_DIR="${LOCAL_PREFIX}" \
        -DCMAKE_PREFIX_PATH="${LOCAL_PREFIX}" \
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
        -DCMAKE_CUDA_FLAGS="-I${LOCAL_PREFIX}/include" \
        -DCMAKE_CXX_FLAGS="-I${LOCAL_PREFIX}/include" \
        2>&1 | tail -5
    make -j"$JOBS" 2>&1 | tail -5
else
    echo "  libsputnik.so already built, skipping"
fi

# Stage lib + headers
mkdir -p "${SPUTNIK_BUILD_DIR}/lib" "${SPUTNIK_BUILD_DIR}/include"
cp "${SPUTNIK_BUILD_DIR}/sputnik/libsputnik.so" "${SPUTNIK_BUILD_DIR}/lib/"
rm -rf "${SPUTNIK_BUILD_DIR}/include/sputnik"
cp -r "${SPUTNIK_DIR}/sputnik" "${SPUTNIK_BUILD_DIR}/include/sputnik"
echo "  Sputnik built successfully"

# ── Step 3: Clone and build Torch-Sputnik ────────────────────────────────────

TORCH_SPUTNIK_DIR="${PREFIX}/Torch-Sputnik"

if [ ! -d "${TORCH_SPUTNIK_DIR}" ]; then
    echo ""
    echo "[3/4] Cloning mabdullahsoyturk/Torch-Sputnik..."
    git clone https://github.com/mabdullahsoyturk/Torch-Sputnik.git "${TORCH_SPUTNIK_DIR}"
else
    echo "[3/4] Torch-Sputnik source already present"
fi

# Fix error_check.h: replace glog CHECK_EQ with TORCH_CHECK
if grep -q "CHECK_EQ" "${TORCH_SPUTNIK_DIR}/include/error_check.h"; then
    echo "  Patching error_check.h to use TORCH_CHECK..."
    cat > "${TORCH_SPUTNIK_DIR}/include/error_check.h" << 'HEREDOC'
#pragma once
#include <cusparse.h>
#include <torch/torch.h>

#define CUDA_CALL(code) \
  do { cudaError_t status = code; \
    TORCH_CHECK(status == cudaSuccess, "CUDA Error: ", cudaGetErrorString(status)); \
  } while (0)

#define CUSPARSE_CALL(code) \
  do { cusparseStatus_t status = code; \
    TORCH_CHECK(status == CUSPARSE_STATUS_SUCCESS, "CuSparse Error: ", cusparseGetErrorString(status)); \
  } while (0)

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
HEREDOC
fi

echo ""
echo "[4/4] Building Torch-Sputnik..."
cd "${TORCH_SPUTNIK_DIR}"
rm -rf build dist *.egg-info

ARCH_STR="arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"

CUDA_HOME="${CUDA_HOME}" \
PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
TORCH_CUDA_ARCH_LIST="${CUDA_ARCH:0:1}.${CUDA_ARCH:1:1}" \
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11 \
LD_LIBRARY_PATH="${SPUTNIK_BUILD_DIR}/lib:${LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
python -c "
import os, sys
sys.argv = ['setup.py', 'install']
os.chdir('${TORCH_SPUTNIK_DIR}')
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
sb = '${SPUTNIK_BUILD_DIR}'
ss = '${SPUTNIK_DIR}'
p = '${LOCAL_PREFIX}'
setup(name='torch_sputnik',
  ext_modules=[CUDAExtension('torch_sputnik',
    ['src/sputnik.cpp','src/spmm_cuda.cu','src/left_replicated_spmm.cu',
     'src/sddmm_cuda.cu','src/softmax_cuda.cu','src/transpose_cuda.cu'],
    include_dirs=['./include',sb+'/include',ss,p+'/include'],
    libraries=['sputnik','cusparse'],
    library_dirs=[sb+'/lib',p+'/lib'],
    extra_compile_args={'cxx':['-O2','-I'+sb+'/include','-I'+ss,'-I'+p+'/include'],
      'nvcc':['-O3','-gencode','${ARCH_STR}','-ccbin','/usr/bin/g++-11',
              '-I'+sb+'/include','-I'+ss,'-I'+p+'/include']},
    extra_link_args=['-L'+sb+'/lib','-L'+p+'/lib',
                     '-Wl,-rpath,'+sb+'/lib','-Wl,-rpath,'+p+'/lib'])],
  cmdclass={'build_ext':BuildExtension})
" 2>&1 | tail -10

echo ""
echo "============================================"
echo "  Build complete!"
echo "============================================"

# ── Verify ────────────────────────────────────────────────────────────────────

TORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')")
LD_LIBRARY_PATH="${TORCH_LIB}:${SPUTNIK_BUILD_DIR}/lib:${LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
python -c "
import torch_sputnik
print('  torch_sputnik imported OK')
for fn in ['spmm', 'sddmm']:
    print(f'  {fn}: {\"available\" if hasattr(torch_sputnik, fn) else \"NOT FOUND\"}')" 2>&1

echo ""
echo "Before running benchmarks, set:"
echo "  export LD_LIBRARY_PATH=${SPUTNIK_BUILD_DIR}/lib:${LOCAL_PREFIX}/lib:\$LD_LIBRARY_PATH"
