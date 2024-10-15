#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>


extern "C" { 
#include "cpu_functions.h"
}
#include "conv_kernels.cuh"

#define N 2048  
#define K 5  
#define M 2048 
#define BLOCK_SIZE 16

__constant__ float window_cm[K*K];

int main() {
    int size_A = M * N * sizeof(float);
    int size_B = K * K * sizeof(float);
    int size_C = (M + 1 - K) * (N + 1 - K) * sizeof(float);

    // Allocate host memory (for cpu benchmarks)
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_window = (float*)malloc(size_B);

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, K);
    initMatrix(h_window, K, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(window_cm, h_window, size_B);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_time;
    float repeats = 100.0f;
    long long flops = 2LL * (M + 1 - K) * (N + 1 - K) * K * K;

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N - K + BLOCK_SIZE) / BLOCK_SIZE, (M - K + BLOCK_SIZE) / BLOCK_SIZE);
    dim3 blockDim1D(BLOCK_SIZE * BLOCK_SIZE);
    
    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        conv_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        conv_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        conv_cm<<<gridDim, blockDim>>>(d_A, d_C, N, K, M);
        cudaDeviceSynchronize();
        conv_shared<<<gridDim, blockDim>>>(d_A, d_C, N, K, M);
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
        "(naive) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );

    // Implementation 2
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        conv_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(gmc) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );

    // Implementation 3
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        conv_cm<<<gridDim, blockDim>>>(d_A, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(cm) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );

    // Implementation 4
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        conv_cm<<<gridDim, blockDim>>>(d_A, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(shared) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time
    );
    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }


    // Free memory
    free(h_A);
    free(h_B);
    free(h_window);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(window_cm);
}


