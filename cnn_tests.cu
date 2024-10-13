#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" { 
#include "cpu_functions.h"
}
#include "cnn_kernels.cuh"

#define N 1024  
#define K 5  
#define M 1024 
#define BLOCK_SIZE 32

int main() {
    float *h_A, *h_B;
    float *d_A, *d_B, *d_C;
    int size_A = M * N * sizeof(float);
    int size_B = K * K * sizeof(float);
    int size_C = (M + 1 - K) * (N + 1 - K) * sizeof(float);

    // Allocate host memory (for cpu benchmarks)
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, K);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_time;
    float repeats = 50.0f;
    long long flops = 2LL * (M + 1 - K) * (N + 1 - K) * K * K;

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M - K) / BLOCK_SIZE, (M - K) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        conv_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
    }

    // Implementation 1
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        conv_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(1) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}