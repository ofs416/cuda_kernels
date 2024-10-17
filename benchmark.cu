#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" { 
#include "cpu_functions.h"
}
#include "gpu_functions.cuh"

#define N 4096  
#define K 4096  
#define M 4096  
#define BLOCK_SIZE 32

#define BM_1D 64
#define BN_1D 64
#define BK_1D 8
#define TM_1D 8

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

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

    cublasHandle_t handle;
    cublasCreate(&handle);
    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_time;
    float repeats = 50.0f;
    long long flops = 2LL * M * N * K;

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim1D(BLOCK_SIZE * BLOCK_SIZE);

    // Grid and block dimensions for 1D block tiling
    dim3 blockDim1DTiling((BM_1D * BN_1D) / TM_1D); 
    dim3 gridDim1DTiling((N + BN_1D - 1) / BN_1D, (M + BM_1D - 1) / BM_1D);

    // Grid and block dimensions for 2D block tiling and vectorized
    dim3 blockDim2DTiling((BM * BN) / (TM * TN));  // Each thread handles TM x TN elements
    dim3 gridDim2DTiling((N + BN - 1) / BN, (M + BM - 1) / BM);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_32F,
        N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_gmc<<<gridDim, blockDim1D>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_1DBlockTiling<<<gridDim1DTiling, blockDim1DTiling>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_2DBlockTiling<<<gridDim2DTiling, blockDim2DTiling>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        gemm_vectorised<<<gridDim2DTiling, blockDim2DTiling>>>(d_A, d_B, d_C, N, K, M);
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

    // Benchmark CuBlas
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_32F,
               N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    double CuBlas_GFLOP = (repeats * flops * 1e-9) / elapsed_time;
    printf(
        "(CuBlas) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        CuBlas_GFLOP, 
        100 * CuBlas_GFLOP / CuBlas_GFLOP);

    // Benchmark GPU implementation 1
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(1) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);

    // Benchmark implementation 2
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_gmc<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(2) Avg time: %f ms, performance: %f GFLOP, %f%% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);

    // Benchmark implementation 2
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_smem<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(3) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);

    // Benchmark implementation 4
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_1DBlockTiling<<<gridDim1DTiling, blockDim1DTiling>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(4) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);

    // Benchmark implementation 5
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_2DBlockTiling<<<gridDim2DTiling, blockDim2DTiling>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(5) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);

    // Benchmark implementation 6
    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++) {
        gemm_vectorised<<<gridDim2DTiling, blockDim2DTiling>>>(d_A, d_B, d_C, N, K, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf(
        "(6) Avg time: %f ms, performance: %f GFLOP, %f %% \n", 
        elapsed_time / repeats, 
        (repeats * flops * 1e-9) / elapsed_time,
        (100 * (repeats * flops * 1e-9) / elapsed_time) / CuBlas_GFLOP);


    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
