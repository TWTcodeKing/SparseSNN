# sengine_cpu: build and usage

`sengine_cpu` is the CPU port of the sengine inference engine: the same
ONNX -> IR -> optimization -> BA-MTTS schedule pipeline, executed by a small
C runtime (`csrc/`) with native fused Conv+BN+IF/LIF kernels. FP32 throughout,
NHWC activations. It has no dependency on `sengine/` or on a GPU.

## 1. Build the C runtime

```bash
cd sengine_cpu/csrc
make NO_BLAS=1        # -> libsengine_cpu.so (needs gcc + OpenMP only)
```

`NO_BLAS=1` is the recommended build: the native conv backend (`native_conv.c`)
resolves `cblas_sgemm` at runtime from the OpenBLAS that ships inside the
`scipy` wheel (`runtime/cpp_executor.py` preloads it with `RTLD_GLOBAL`), so no
system BLAS is required. If the symbol is not found it falls back to a naive
OpenMP GEMM and prints a warning.

Plain `make` (without `NO_BLAS=1`) links the system OpenBLAS instead
(`-lopenblas`, `-DHAVE_OPENBLAS`); use it only when `libopenblas` is installed.

The library is loaded from `sengine_cpu/csrc/libsengine_cpu.so` (or next to
`runtime/cpp_executor.py`). Rebuild after any change under `csrc/`.

## 2. Usage

Build from a plugin-mode ONNX export (the same `*_plugin.onnx` the GPU engine
uses, produced by `python -m sengine.scripts.export_onnx ...`):

```python
import numpy as np
import sengine_cpu

engine = sengine_cpu.build("sengine/exports/snn_vgg9_urbansound8k_plugin.onnx",
                           T=4, batch_size=1, n_threads=8)
x = np.random.rand(1, 1, 64, 176).astype(np.float32)   # NCHW (B, C, H, W)
logits = engine.infer(x)                                # (B, num_classes) float32
ms = engine.benchmark()

engine.save("model.sengine-cpu")
engine = sengine_cpu.load("model.sengine-cpu", n_threads=8)
```

`infer(x)` takes one NCHW frame per batch element, tiles it T times into the
input buffer, runs the schedule in C (no Python in the hot loop) and returns
the time-averaged logits.

End-to-end correctness against PyTorch, including full test-set accuracy:

```bash
python scripts/verify_workloads.py --engine sengine_cpu --model snn_vgg9 --dataset urbansound8k \
    --T 4 --checkpoint output/snn_vgg9_urbansound8k_bs32_lr0.0005/best.pth --eval-acc --threads 32
```

Environment variables:

| Variable | Effect |
|---|---|
| `SENGINE_CPU_BACKEND=native` | force the native C conv kernels (default when no TVM interpreter is found) |
| `SENGINE_CPU_BACKEND=tvm` | force the TVM backend; errors out if no TVM interpreter is configured |
| `SENGINE_CPU_TVM_PYTHON` | path to a python with Apache TVM importable (see below) |

## 3. Optional TVM backend

The TVM backend compiles Conv1x1+BN+IF/LIF and Add+LIF kernels with TVM TE
(`kernels/conv_bn_if.py`, `kernels/add_lif.py`) into standalone `.so` files
under `.sengine_cpu.cache/`. Only those two kernel families exist; 3x3
convolutions always use the native backend. It is not needed for any of the
reported CPU results.

TVM must live in its own interpreter because tilelang (used by the GPU
engine) bundles a TVM whose `tvm_ffi` conflicts with Apache TVM.
`build/tvm_compiler.py` talks to that interpreter only through a subprocess.

```bash
bash sengine_cpu/env_setup.sh          # creates sengine_cpu/.tvm-env with TVM + LLVM
# or point at an existing interpreter:
export SENGINE_CPU_TVM_PYTHON=/path/to/tvm-venv/bin/python
SENGINE_CPU_BACKEND=tvm python -c "import sengine_cpu; sengine_cpu.build('model_plugin.onnx', T=4)"
```

Resolution order is `$SENGINE_CPU_TVM_PYTHON`, then `sengine_cpu/.tvm-env/bin/python`
(`tvm_env.py`). When neither exists the engine silently uses the native backend.

## Appendix: standalone MLAS kernel (benchmark only)

`csrc/mlas_conv_bn_lif.cpp` is a single fused Conv+BN+LIF kernel on top of
MLAS (TB-merged GEMM, fused BN + LIF epilogue). It is used only by
`scripts/bench_snn_vgg9_cpu.py` (SNN-VGG-9 on NTU-Fi HumanID vs ONNX Runtime /
OpenVINO / ncnn), not by the engine above.

```bash
# 1. MLAS (one-time)
MLAS=$HOME/ov_mlas
git clone https://github.com/openvinotoolkit/mlas.git $MLAS
cd $MLAS && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLAS_BUILD_TESTS=OFF && make -j$(nproc)

# 2. Kernel (x86-64, AVX2/FMA; AVX-512 is picked up by -march=native)
cd <repo>/sengine_cpu/csrc
g++ -O3 -march=native -mavx2 -mfma -fPIC -fopenmp -shared -std=c++17 \
    -I$MLAS/inc -I$MLAS/lib \
    -o libmlas_conv_bn_lif.so mlas_conv_bn_lif.cpp \
    -Wl,--whole-archive $MLAS/build/libmlas.a -Wl,--no-whole-archive \
    -ldl -lm -lpthread -fopenmp

# 3. Baselines and benchmark
uv pip install onnxruntime openvino ncnn
python scripts/bench_snn_vgg9_cpu.py --threads 1,4,8 [--frameworks sengine,ort] [--numa 0]
```

On ARM (NEON) replace `-mavx2 -mfma` with `-mfpu=neon`; the BN+LIF epilogue has
a 4-wide NEON path. `numactl` (`--numa`) is optional for socket pinning.
The benchmark exports its ONNX / ncnn models on first run into
`.cache/snn_vgg9_cpu_bench/`.
