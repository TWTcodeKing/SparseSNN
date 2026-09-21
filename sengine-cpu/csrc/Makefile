# sengine-cpu: C runtime for SNN inference on CPU
#
# Build:  make              (with OpenBLAS)
#         make NO_BLAS=1    (naive GEMM fallback)
#
# Output: libsengine_cpu.so (loaded via Python ctypes)

CC      = gcc
CFLAGS  = -O3 -march=native -fPIC -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -shared -ldl -lm -fopenmp

TARGET  = libsengine_cpu.so
SRCS    = cpu_executor.c native_kernels.c native_conv.c

# ─── Optional OpenBLAS linkage ───
# Set NO_BLAS=1 to disable (uses naive triple-loop GEMM)
ifndef NO_BLAS
    CFLAGS  += -DHAVE_OPENBLAS
    LDFLAGS += -lopenblas
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) cpu_executor.h native_kernels.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
