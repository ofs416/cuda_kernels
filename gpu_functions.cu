#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#include "cpu_functions.h"

#define N 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define M 256  // Number of columns in B and C
#define BLOCK_SIZE 32

// CUDA kernel for matrix addition
__global__ void matrixAdditionGPU (float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        C[n * row + col] = A[n * row + col] + B[n * row + col];
    }
}

// CUDA kernel for matrix multiplication (N x K) @ (K x M)
__global__ void matrixMultiplicationGPU (float *A, float *B, float *C, int n, int k, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * m + col];
        }
        C[m * row + col] = sum;
    }
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
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
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 5; i++) {
        matrixMultiplicationCPU(h_A, h_B, h_C_cpu, N, K, M);
        matrixMultiplicationGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU\n");
    double cpuTotalTime = 0.0;
    for (int i = 0; i < 20; i++) {
        double startTime = getTime();
        matrixMultiplicationCPU(h_A, h_B, h_C_cpu, N, K, M);
        double endTime = getTime();
        cpuTotalTime += endTime - startTime;
    }
    double cpuAvgTime = cpuTotalTime / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU\n");
    double gpuTotalTime = 0.0;
    for (int i = 0; i < 20; i++) {
        double startTime = getTime();
        matrixMultiplicationGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M);
        cudaDeviceSynchronize();
        double endTime = getTime();
        gpuTotalTime += endTime - startTime;
    }
    double gpuAvgTime = gpuTotalTime / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpuAvgTime * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpuAvgTime * 1e6f));
    printf("Speedup: %fx\n", cpuAvgTime / gpuAvgTime);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
