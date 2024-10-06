#include "gpu_functions.h"

#define BLOCK_SIZE 32

// CUDA kernel for matrix addition
__global__ void matrixAdditionGPU (float *A, float *B, float *C, int n) {
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        C[n * row + col] = A[n * row + col] + B[n * row + col];
    }
}

// Naive CUDA kernel for matrix multiplication (N x K) @ (K x M)
__global__ void matrixMultiplicationGPU (float *A, float *B, float *C, int n, int k, int m) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[n * row + col] = sum;
    }
}

// Global Memory Coalescing CUDA kernel for matrix multiplication (N x K) @ (K x M)
// Increases performance by grouping memory accesses of threads that are in the same warp
// and executed as one
// Each warp contains 32 threads and memory accesses can be 32B, 64B and 128B
// To take advantage of 128B single access, the floats should be conseecutive in memory
// and aligned in access (but the accesses don’t have to be consecutive within-warp)
__global__ void gemm_gmc (float *A, float *B, float *C, int n, int k, int m) {
    const uint row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[n * row + col] = sum;
    }
}
