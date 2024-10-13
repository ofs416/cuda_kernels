CC = gcc
NVCC = nvcc
CFLAGS = -O3
NVCCFLAGS = -O3
LDFLAGS = -lcublas

SOURCES = benchmark.cu gpu_functions.cu cpu_functions.c
TARGET = benchmark

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

cnn_tests: cnn_tests.cu
	$(NVCC) $(NVCCFLAGS) -o cnn_tests cnn_tests.cu cnn_kernels.cu cpu_functions.c $(LDFLAGS)

run_gemm: $(TARGET)
	./$(TARGET)

run_cnn: cnn_tests
	./cnn_tests

clean:
	rm -f $(TARGET) cnn_tests

.PHONY: all run run_cnn_tests clean