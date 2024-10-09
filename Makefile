# Makefile

CC = gcc
NVCC = nvcc
CFLAGS = -O3
NVCCFLAGS = -O3
LDFLAGS = -lcublas

SOURCES = benchmark.cu gpu_functions.cu cpu_functions.c
TARGET = benchmark

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean