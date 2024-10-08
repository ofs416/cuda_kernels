#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

extern "C" { 
#include "cpu_functions.h"
}
#include "gpu_functions.cuh"

#define N 2048  // Number of rows in A and C
#define K 2048   // Number of columns in A and rows in B
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
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim4((N + 64 - 1) / 64, (M + 64 - 1) / 64);
    dim3 blockDim4((64 * 64) / 8);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_1DBlockTiling<<<gridDim4, blockDim4>>>(d_A, d_B, d_C, N, K, M);
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

    float elapsed_time;
    float repeats = 100.0f;
    long long flops = 2LL * M * N * K;

    // Benchmark GPU implementation 1
    printf("Benchmarking imp 1\n");
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(1) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time);

    // Benchmark implementation 2
    printf("Benchmarking imp 2\n");
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_gmc<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(2) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time);

    // Benchmark implementation 2
    printf("Benchmarking imp 3\n");
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(3) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time);

    // Benchmark implementation 4
    printf("Benchmarking imp 4\n");
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_1DBlockTiling<<<gridDim4, blockDim4>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(4) Avg time: %f ms, performance: %f GFLOP\n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time);

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
