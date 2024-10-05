#include "gpu_functions.h"

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

