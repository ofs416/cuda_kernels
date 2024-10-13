// Implementation of CNN kernel following based on the optimisations seen in gpu_functions.cu

#include <cuda_runtime.h>
#include "cnn_kernels.cuh"

#define BLOCK_SIZE 32

// Naive CUDA kernel for CNN 
// Kernel (matrix B) size of k (heed attention to context of the work kernel)
// Input matrix B with size m X n
// Output size of m + 1 - k x n + 1 - k
__global__ void cnn (float *A, float *B, float *C, uint n, uint k, uint m) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = - (int)(k / 2); i < (int)(k / 2); i++) {
            for (int j = - (int)(k / 2) ; j < (int)(k / 2); j++) {
                 sum += A[k * i + j] * B[n * (row + i) + col + j];
            }
        }
        C[n * row + col] = sum;
    }
}