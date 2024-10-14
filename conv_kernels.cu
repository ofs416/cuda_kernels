// Implementation of CNN kernel following based on the optimisations seen in gpu_functions.cu

#include <cuda_runtime.h>
#include "cONV_kernels.cuh"

#define BLOCK_SIZE 32

// Naive CUDA kernel for convolution 
// Window (matrix B) size of k (heed attention to context of the work kernel)
// Input matrix A with size m X n
// Output size of m + 1 - k x n + 1 - k
__global__ void conv_naive (float *A, float *B, float *C, uint n, uint k, uint m) {
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < (m + 1 - k) && col < (n + 1 - k)) {
        float sum = 0.0f;
        for (int i = 0 ; i < k ; i++) {
            for (int j = 0 ; j < k ; j++) {
                sum += 
                    A[(int)(k / 2) * (n + 1 + row) + (j - (int)(k / 2)) + n * (i - (int)(k / 2))]
                    * B[k * i + j];
            }
        }
        C[(n + 1 - k) * row + col] = sum;
    }
}

// Global Memory Coalescing
__global__ void conv_gmc (float *A, float *B, float *C, uint n, uint k, uint m) {
    const uint row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row < (m + 1 - k) && col < (n + 1 - k)) {
        float sum = 0.0f;
        for (int i = 0 ; i < k ; i++) {
            for (int j = 0 ; j < k ; j++) {
                sum += 
                    A[(int)(k / 2) * (n + 1 + row) + (j - (int)(k / 2)) + n * (i - (int)(k / 2))]
                    * B[k * i + j];
            }
        }
        C[(n + 1 - k) * row + col] = sum;
    }
}


// Shared Memory 
__global__ void gemm_smem(float *A, float *B, float *C, int n, int k, int m) {
    __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_shared[k * k];

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * k;                                   
    C += cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE; 

    for (int i = 0 ; i < k ; i++) {
        for (int j = 0 ; j < k ; j++) {
            B_shared[k * i + j] = B[k * i + j];
        }
    }

    float sum = 0.0f;
    for (int blkIdx = 0; blkIdx < k; blkIdx += BLOCK_SIZE) {
        A_shared[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * k + threadCol];
    
        __syncthreads();
        A += BLOCK_SIZE;

        for (int i = 0 ; i < k ; i++) {
            for (int j = 0 ; j < k ; j++) {
                sum += 
                    A[(int)(k / 2) * (n + 1 + row) + (j - (int)(k / 2)) + n * (i - (int)(k / 2))]
                    * B[k * i + j];
            }
        }
        __syncthreads();
    }
    C[threadRow * n + threadCol] = tmp;
}