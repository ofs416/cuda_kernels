#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

extern "C" { 
#include "cpu_functions.h"
}

#define BLOCK_SIZE 16
#define MAX_FILTER_SIZE 63
#define SMEM_SIZE (BLOCK_SIZE + MAX_FILTER_SIZE - 1)

__constant__ float filter_cm[MAX_FILTER_SIZE];

__global__ void conv_1dhz_smem(float *input, float *output, int width, 
                              int height, int filter_size) {
    extern __shared__ float shared_mem[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    const int radius = filter_size / 2;
    
    if (row >= height) return;

    float* row_shared = &shared_mem[ty * SMEM_SIZE];
    const int block_start = blockIdx.x * BLOCK_SIZE - radius;
    for (int i = tx; i < (BLOCK_SIZE + filter_size - 1); i += BLOCK_SIZE) {
        int global_idx = block_start + i;
        if (global_idx >= 0 && global_idx < width) {
            row_shared[i] = input[row * width + global_idx];
        } else {
            row_shared[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Compute convolution only for valid output positions
    if (col < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < filter_size; i++) {
            sum += row_shared[tx + i] * filter_cm[i];
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
    const unsigned int filter_size = 63;
    const unsigned int image_size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(image_size);
    float *h_output = (float*)malloc(image_size);
    float *h_output_gpu = (float*)malloc(image_size);
    float *h_output_cpu = (float*)malloc(image_size);
    float *h_filter = (float*)malloc(filter_size * sizeof(float));

    // Initialize input image and kernel (example initialization)
    initMatrix(h_input, height, width);
    initMatrix(h_filter, filter_size, 1);

    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, image_size), "cudaMalloc d_input");
    check_cuda_error(cudaMalloc(&d_output, image_size), "cudaMalloc d_output");

    // Copy input data to device
    check_cuda_error(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice), "cudaMemcpy H2D input");
    check_cuda_error(cudaMemcpyToSymbol(filter_cm, h_filter, filter_size * sizeof(float)), "cudaMemcpyToSymbol kernel");

    // Launch configuration
    size_t shared_mem_size = BLOCK_SIZE * SMEM_SIZE * sizeof(float);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
            (height + blockDim.y - 1) / blockDim.y);


    // Check computation
    // Compute convolution on CPU
    conv_1dhz_cpu(h_input, h_output_cpu, width, height, h_filter, filter_size);
    // compute convolution with custom kernel
    conv_1dhz_smem<<<gridDim, blockDim, shared_mem_size>>>(d_input, d_output, width, height, filter_size);
    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError(), "Kernel launch");
    // Wait for kernel to finish
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_output_gpu, d_output, image_size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H output");
    // Compare CPU and GPU results
    compare_results(h_output_cpu, h_output_gpu, width, height, 1e-5f);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        conv_1dhz_smem<<<gridDim, blockDim, shared_mem_size>>>(d_input, d_output, width, height, filter_size);
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
    long long flops = 2LL * width * height * filter_size;
    check_cuda_error(cudaEventRecord(start), "start event recording");
    for (int i = 0; i < repeats; i++) {
        conv_1dhz_smem<<<gridDim, blockDim, shared_mem_size>>>(d_input, d_output, width, height, filter_size);
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