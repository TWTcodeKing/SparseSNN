# sengine C++ executor.
#   make            -> sengine_exec (standalone binary) + libsengine_exec.so (ctypes library)
#   make lib        -> libsengine_exec.so only
# Defaults target the RTX 4090 host (CUDA 12.8, sm_89). Override per GPU, e.g. Jetson AGX Orin:
#   make lib ARCH="-gencode arch=compute_87,code=sm_87" CUDA_HOME=/usr/local/cuda
# A100: ARCH="-gencode arch=compute_80,code=sm_80". The library also needs cuDNN
# (CUDNN_LIB defaults to the system cuDNN in /usr/lib/x86_64-linux-gnu; on the Orin
# JetPack ships it under /usr/lib/aarch64-linux-gnu).
CUDA_HOME ?= /usr/local/cuda-12.8
NVCC = $(CUDA_HOME)/bin/nvcc
ARCH ?= -gencode arch=compute_89,code=sm_89
# Link against this toolkit's runtime explicitly: without -L the host linker may
# resolve -lcudart/-lcublas from another CUDA install registered in ldconfig
# (e.g. /usr/local/cuda-11.8), and a cudart-11 library loaded next to torch's
# cudart-12 aborts TileLang ("libcudart symbols not found globally").
CUDA_LIB ?= -L$(CUDA_HOME)/lib64
CUDNN_LIB ?= -L/usr/lib/x86_64-linux-gnu -lcudnn

.PHONY: all lib clean

all: sengine_exec libsengine_exec.so

lib: libsengine_exec.so

sengine_exec: sengine_exec.cu kernels.cuh
	$(NVCC) -O3 --use_fast_math $(ARCH) -std=c++17 \
		-o $@ sengine_exec.cu \
		$(CUDA_LIB) -lcudart -lcublas -ldl

libsengine_exec.so: cpp_executor.cu
	$(NVCC) -O3 --use_fast_math $(ARCH) -shared -Xcompiler -fPIC \
		-o $@ cpp_executor.cu \
		$(CUDA_LIB) -lcudart -ldl -lcublas $(CUDNN_LIB)

clean:
	rm -f sengine_exec libsengine_exec.so
