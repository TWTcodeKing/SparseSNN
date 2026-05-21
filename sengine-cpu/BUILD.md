# sengine-cpu: Build & Deploy Guide

## Quick Start (any x86_64 Linux)

```bash
# 1. Clone MLAS (one-time)
cd /tmp && git clone https://github.com/openvinotoolkit/mlas.git ov_mlas
cd ov_mlas && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLAS_BUILD_TESTS=OFF && make -j$(nproc)

# 2. Build sengine-cpu MLAS kernel
cd <project_root>/sengine-cpu/csrc
g++ -O3 -march=native -mavx2 -mfma -fPIC -fopenmp -shared -std=c++17 \
    -I/tmp/ov_mlas/inc -I/tmp/ov_mlas/lib \
    -o libmlas_conv_bn_lif.so mlas_conv_bn_lif.cpp \
    -Wl,--whole-archive /tmp/ov_mlas/build/libmlas.a -Wl,--no-whole-archive \
    -ldl -lm -lpthread -fopenmp

# 3. Install Python deps
pip install numpy onnxruntime openvino ncnn torch  # or uv pip install

# 4. Run benchmark
python scripts/bench_snn_vgg9_cpu.py --threads 1,4,8
```

## Platform-specific notes

### Intel Xeon (AVX-512)
- MLAS auto-detects AVX-512 via `-march=native`
- No code changes needed; same build commands work
- May need `numactl` for NUMA pinning: `apt install numactl`

### AMD EPYC (AVX2/FMA3)
- Current development platform, works out of the box

### ARM Cortex-A7 / Raspberry Pi (NEON)
```bash
# Cross-compile MLAS for ARM (on x86 host)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=armv7l

# Or native build on Pi:
cd /tmp && git clone https://github.com/openvinotoolkit/mlas.git ov_mlas
cd ov_mlas && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Build sengine-cpu kernel (ARM)
g++ -O3 -march=native -mfpu=neon -fPIC -fopenmp -shared -std=c++17 \
    -I/tmp/ov_mlas/inc -I/tmp/ov_mlas/lib \
    -o libmlas_conv_bn_lif.so mlas_conv_bn_lif.cpp \
    -Wl,--whole-archive /tmp/ov_mlas/build/libmlas.a -Wl,--no-whole-archive \
    -ldl -lm -lpthread -fopenmp
```
- BN+LIF epilogue has NEON path (vld1q_f32/vst1q_f32, 4-wide)
- ncnn ARM Winograd is highly optimized; expect closer competition on ARM

## Files to copy to target machine

```
sengine-cpu/csrc/mlas_conv_bn_lif.cpp    # Source (rebuild on target)
scripts/bench_snn_vgg9_cpu.py            # Benchmark script
```

The benchmark script auto-exports ONNX/ncnn models on first run.
No pre-exported models needed.
