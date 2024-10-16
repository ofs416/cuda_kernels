#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

extern "C" { 
#include "cpu_functions.h"
}

#define BLOCK_SIZE 16
#define PADDING (BLOCK_SIZE + 64)

// Constant memory
__constant__ float filter_cm[64];
__global__ void conv_1dhz_smem(float *input, float *output, int width,
                              int height, int f_size) {
    __shared__ float shared_mem[BLOCK_SIZE*PADDING];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int radius = f_size / 2;
    
    // Load main block data
    if (row < height) {
        if (col < width) {
            shared_mem[ty * PADDING + tx + radius] = input[row * width + col];
        }
        
        // Load left halo
        if (tx < radius) {
            int input_col = col - radius;
            if (input_col >= 0) {
                shared_mem[ty * PADDING + tx] = input[row * width + input_col];
            } else {
                shared_mem[ty * PADDING + tx] = 0.0f;
            }
        }
        
        // Load right halo
        if (tx < radius) {
            int input_col = col + BLOCK_SIZE;
            if (input_col < width) {
                shared_mem[ty * PADDING + tx + BLOCK_SIZE + radius] = 
                    input[row * width + input_col];
            } else {
                shared_mem[ty * PADDING + tx + BLOCK_SIZE + radius] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (row < height && col < width) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < f_size; j++) {
            sum += shared_mem[ty * PADDING + tx + j] * filter_cm[j];
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
    const unsigned int filter_size = 31;
    const unsigned int image_size = width * height * sizeof(float);
    size_t shared_mem_size = (BLOCK_SIZE + 2 * (filter_size / 2)) * BLOCK_SIZE * sizeof(float);

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

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    if (compare_results(h_output_cpu, h_output_gpu, width, height, 1e-5f)) {
        printf("CPU and GPU results match!\n");
    } else {
        printf("CPU and GPU results do not match.\n");
    }

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