# Makefile

NVCC = nvcc
CFLAGS = -o
TARGET = benchmark
SRC = benchmark.cu gpu_functions.cu cpu_functions.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean