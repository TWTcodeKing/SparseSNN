#!/bin/bash
# sengine-cpu TVM environment setup
#
# TVM must be installed separately from the main SparseSNN venv because
# tilelang bundles its own TVM (0.23.dev0) without LLVM codegen support.
# We need standalone Apache TVM with LLVM for CPU kernel compilation.
#
# This script creates a dedicated uv venv and installs TVM with LLVM.
# The compiled .so kernels are standalone — TVM is NOT needed at runtime.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.tvm-env"

echo "=== sengine-cpu TVM Environment Setup ==="

# 1. Create venv
if [ ! -d "$ENV_DIR" ]; then
    echo "[1/4] Creating uv venv at $ENV_DIR ..."
    uv venv "$ENV_DIR"
else
    echo "[1/4] Venv already exists at $ENV_DIR"
fi

# 2. Install TVM
echo "[2/4] Installing Apache TVM (with LLVM support) ..."
# Option A: Try tlcpack nightly (includes LLVM)
"$ENV_DIR/bin/pip" install --pre apache-tvm 2>/dev/null \
    || "$ENV_DIR/bin/pip" install tlcpack-nightly 2>/dev/null \
    || {
        echo ""
        echo "ERROR: Could not install TVM from PyPI."
        echo "You may need to build from source with -DUSE_LLVM=ON."
        echo "See: https://tvm.apache.org/docs/install/from_source.html"
        echo ""
        echo "Quick guide:"
        echo "  git clone --recursive https://github.com/apache/tvm.git /tmp/tvm-build"
        echo "  cd /tmp/tvm-build && mkdir build && cd build"
        echo "  cmake .. -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release"
        echo "  make -j\$(nproc)"
        echo "  cd ../python && $ENV_DIR/bin/pip install -e ."
        exit 1
    }

# 3. Install other dependencies (numpy, onnx for kernel compilation)
echo "[3/4] Installing dependencies ..."
"$ENV_DIR/bin/pip" install numpy onnx

# 4. Verify
echo "[4/4] Verifying installation ..."
"$ENV_DIR/bin/python" -c "
import tvm
print(f'TVM version: {tvm.__version__}')
llvm_ok = tvm.runtime.enabled('llvm')
print(f'LLVM enabled: {llvm_ok}')
if not llvm_ok:
    print('ERROR: TVM installed but LLVM codegen not available.')
    print('Rebuild TVM from source with -DUSE_LLVM=ON')
    exit(1)

# Test target creation
target = tvm.target.Target('llvm -mcpu=core-avx2')
print(f'x86 AVX2 target: OK')

print()
print('=== TVM environment ready ===')
print(f'Python: {tvm.__file__}')
print(f'To use: source {\"$ENV_DIR\"}/bin/activate')
"

echo ""
echo "Done. Activate with:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "Supported targets:"
echo "  x86 AVX2:    llvm -mcpu=core-avx2"
echo "  x86 AVX-512: llvm -mcpu=skylake-avx512"
echo "  ARM v8:      llvm -mtriple=aarch64-linux-gnu -mcpu=neoverse-n1"
