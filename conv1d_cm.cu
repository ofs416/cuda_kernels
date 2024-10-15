#include <cuda_runtime.h>
#include <stdio.h>

extern "C" { 
#include "cpu_functions.h"
}

#define BLOCK_SIZE 256

// Constant memory
__constant__ float filter_cm[5];

__global__ void conv_1dhz_cm(float *input, float *output, int width,
                                                     int height, int f_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    const int total_size = width * height;
    const int padding = f_size / 2;  // Assuming odd kernel size

    for (int i = tid; i < total_size; i += stride) {
        int row = i / width;
        int col = i % width;
        
        float sum = 0.0f;
        for (int j = 0; j < f_size; j++) {
            int input_col = col + j - padding;
            if (input_col >= 0 && input_col < width) {
                sum += input[row * width + input_col] * filter_cm[j];
            }
            // Implicit zero-padding: we don't add anything for out-of-bounds inputs
        }
        output[row * width + col] = sum;
    }
}


void check_cuda_error(cudaError_t error, const char *function_name) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function_name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


int main() {
    // Image and kernel parameters
    const unsigned int width = 4096;
    const unsigned int height = 4096;
    const unsigned int kernel_size = 5;
    const unsigned int image_size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(image_size);
    float *h_output = (float*)malloc(image_size);
    float *h_filter = (float*)malloc(kernel_size * sizeof(float));

    // Initialize input image and kernel (example initialization)
    initMatrix(h_input, height, width);
    initMatrix(h_filter, kernel_size, 1);

    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, image_size), "cudaMalloc d_input");
    check_cuda_error(cudaMalloc(&d_output, image_size), "cudaMalloc d_output");

    // Copy input data to device
    check_cuda_error(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice), "cudaMemcpy H2D input");
    check_cuda_error(cudaMemcpyToSymbol(filter_cm, h_filter, kernel_size * sizeof(float)), "cudaMemcpyToSymbol kernel");

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        conv_1dhz_cm<<<gridDim, blockDim>>>(d_input, d_output, width, height, kernel_size);
        // Wait for kernel to finish
        check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    

    // Create events
    cudaEvent_t start, stop;
    check_cuda_error(cudaEventCreate(&start), "create event start");
    check_cuda_error(cudaEventCreate(&stop), "create event stop");
    // Benchmark
    float elapsed_time;
    float repeats = 100.0f;
    long long flops = 2LL * width * height * kernel_size;
    check_cuda_error(cudaEventRecord(start), "start event recording");
    for (int i = 0; i < repeats; i++) {
        conv_1dhz_cm<<<gridDim, blockDim>>>(d_input, d_output, width, height, kernel_size);
    }
    check_cuda_error(cudaEventRecord(stop), "stop event recording");
    check_cuda_error(cudaEventSynchronize(start), "cudaDeviceSynchronize");
    check_cuda_error(cudaEventSynchronize(stop), "cudaDeviceSynchronize");
    check_cuda_error(cudaEventElapsedTime(&elapsed_time, start, stop), "elapsed time");
    printf(
        "Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );
    

    // Free memory
    check_cuda_error(cudaFree(d_input), "cudaFree d_input");
    check_cuda_error(cudaFree(d_output), "cudaFree d_output");
    check_cuda_error(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda_error(cudaEventDestroy(stop), "cudaEventDestroy stop");
    free(h_input);
    free(h_output);
    free(h_filter);
  

    return 0;
}