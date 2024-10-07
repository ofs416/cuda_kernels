#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" { 
#include "cpu_functions.h"
}
#include "gpu_functions.h"

#define N 2048  // Number of rows in A and C
#define K 1024   // Number of columns in A and rows in B
#define M 2048  // Number of columns in B and C
#define BLOCK_SIZE 32

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory (for cpu benchmarks)
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        matrixMultiplicationGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        gemm_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    // printf("Benchmarking matrixMultiplicationCPU\n");
    // double cpuTotalTime = 0.0;
    // for (int i = 0; i < 20; i++) {
    //    double startTime = getTime();
    //    matrixMultiplicationCPU(h_A, h_B, h_C_cpu, N, K, M);
    //    double endTime = getTime();
    //    cpuTotalTime += endTime - startTime;
    //}
    // double cpuAvgTime = cpuTotalTime / 20.0;

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark GPU implementation 1
    printf("Benchmarking imp 1\n");
    float totalMilliseconds = 0;
    for (int i = 0; i < 100; i++) {
        // Record start event
        cudaEventRecord(start, 0);
        // Launch kernel
        matrixMultiplicationGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        // Record stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }
    float avgMilliseconds1 = totalMilliseconds / 100.0f;

    // Benchmark implementation 2
    printf("Benchmarking imp 2\n");
    totalMilliseconds = 0;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start, 0);
        gemm_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }
    float avgMilliseconds2 = totalMilliseconds / 100.0f;

        // Benchmark implementation 2
    printf("Benchmarking imp 3\n");
    totalMilliseconds = 0;
    for (int i = 0; i < 100; i++) {
        cudaEventRecord(start, 0);
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
    }
    float avgMilliseconds3 = totalMilliseconds / 100.0f;

    // Print results
    printf("Average time for 1: %f ms\n", avgMilliseconds1);
    printf("Average time for 2: %f ms\n", avgMilliseconds2);
    printf("Average time for 3: %f ms\n", avgMilliseconds3);
    printf("speed up from 2: %f\n", avgMilliseconds1/avgMilliseconds2);
    printf("speed up from 3: %f\n", avgMilliseconds1/avgMilliseconds3);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}