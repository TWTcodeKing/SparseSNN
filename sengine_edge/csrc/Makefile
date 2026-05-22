CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
ARCH = -gencode arch=compute_87,code=sm_87

.PHONY: all clean

all: sengine_exec

sengine_exec: sengine_exec.cu kernels.cuh
	$(NVCC) -O3 --use_fast_math $(ARCH) -std=c++17 \
		-o $@ sengine_exec.cu \
		-lcudart -lcublas -ldl

clean:
	rm -f sengine_exec
