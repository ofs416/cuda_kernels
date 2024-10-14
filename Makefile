CC = gcc
NVCC = nvcc
CFLAGS = -O3
NVCCFLAGS = -O3
LDFLAGS = -lcublas

SOURCES = benchmark.cu gpu_functions.cu cpu_functions.c
TARGET = benchmark

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

conv_tests: conv_tests.cu
	$(NVCC) $(NVCCFLAGS) -o conv_tests conv_tests.cu conv_kernels.cu cpu_functions.c $(LDFLAGS)

run_gemm: $(TARGET)
	./$(TARGET)

run_conv: conv_tests
	./conv_tests

clean:
	rm -f $(TARGET) conv_tests

.PHONY: all run run_conv_tests clean