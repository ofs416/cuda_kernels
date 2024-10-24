#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

extern "C" { 
#include "cpu_functions.h"
}

#define BLOCK_SIZE 256  // Increased from 16 since we're now 1D
#define MAX_FILTER_SIZE 63
#define SMEM_SIZE (BLOCK_SIZE + MAX_FILTER_SIZE - 1)

#define EL_PER_THR 8  // Each thread processes 4 elements

__constant__ float filter_cm[MAX_FILTER_SIZE];

__global__ void conv_1d_tile_ar(float *input, float *output, int width, 
                                        int height, int filter_size, bool transpose) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int base_col = blockIdx.x * (BLOCK_SIZE * EL_PER_THR) + tid;
    const int radius = filter_size / 2;
    
    int trans_width = transpose ? height : width;
    int trans_height = transpose ? width : height;
    int input_row = transpose ? base_col : row;
    
    if (input_row >= trans_height) return;

    // Load data into shared memory
    const int block_start = blockIdx.x * (BLOCK_SIZE * EL_PER_THR) - radius;
    for (int i = tid; i < (BLOCK_SIZE * EL_PER_THR + filter_size - 1); i += BLOCK_SIZE) {
        int global_idx = block_start + i;
        if (global_idx >= 0 && global_idx < trans_width) {
            shared_mem[i] = input[input_row * trans_width + global_idx];
        } else {
            shared_mem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Process multiple elements per thread
    #pragma unroll
    for (int offset = 0; offset < EL_PER_THR; offset++) {
        int input_col = base_col + (offset * BLOCK_SIZE);
        
        if (input_col < trans_width) {
            float sum = 0.0f;
            
            // Compute convolution for this element
            #pragma unroll
            for (int i = 0; i < filter_size; i++) {
                sum += shared_mem[tid + (offset * BLOCK_SIZE) + i] * filter_cm[i];
            }
            
            // Store result
            if (transpose) {
                output[input_col * trans_height + input_row] = sum;
            } else {
                output[input_row * trans_width + input_col] = sum;
            }
        }
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

    // Launch configuration
    size_t sharedMemSize = (BLOCK_SIZE * EL_PER_THR + MAX_FILTER_SIZE - 1) * sizeof(float);
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim((width + (BLOCK_SIZE * EL_PER_THR) - 1) / (BLOCK_SIZE * EL_PER_THR), 
                 height, 1);

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

    // Check computation
    conv_1dhz_cpu(h_input, h_output_cpu, width, height, h_filter, filter_size);
    conv_1d_tile_ar<<<gridDim, blockDim, sharedMemSize >>>(d_input, d_output, width, height, filter_size, false);
    check_cuda_error(cudaGetLastError(), "Kernel launch");
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda_error(cudaMemcpy(h_output_gpu, d_output, image_size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H output");
    compare_results(h_output_cpu, h_output_gpu, width, height, 1e-5f);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
         conv_1d_tile_ar<<<gridDim, blockDim, sharedMemSize >>>(d_input, d_output, width, height, filter_size, false);
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
         conv_1d_tile_ar<<<gridDim, blockDim, sharedMemSize >>>(d_input, d_output, width, height, filter_size, false);
    }
    check_cuda_error(cudaEventRecord(stop), "stop event recording");
    check_cuda_error(cudaEventSynchronize(start), "cudaDeviceSynchronize");
    check_cuda_error(cudaEventSynchronize(stop), "cudaDeviceSynchronize");
    check_cuda_error(cudaEventElapsedTime(&elapsed_time, start, stop), "elapsed time");
    printf(
        "Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-6) / elapsed_time
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