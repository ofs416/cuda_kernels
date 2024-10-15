// Implementation of CNN kernel following based on some of the optimisations seen in gpu_functions.cu
// Note these won't accurately compute the convolution, instead they are used to indicate compute performance

#include <cuda_runtime.h>
#include "conv_kernels.cuh"

#define BLOCK_SIZE 16

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

// Constant memory
__constant__ float window_cm[BLOCK_SIZE * BLOCK_SIZE];
__global__ void conv_cm(float *A, float *C, uint n, uint k, uint m) {
    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;
    const uint row = blockIdx.x * BLOCK_SIZE + (threadRow);
    const uint col = blockIdx.y * BLOCK_SIZE + (threadCol);

    if (row < (m + 1 - k) && col < (n + 1 - k)) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                sum += 
                    A[(int)(k / 2) * (n + 1 + row) + (j - (int)(k / 2)) + n * (i - (int)(k / 2))]
                    * window_cm[k * i + j];
            }
        }
        C[(n + 1 - k) * row + col] = sum;
    }
}


// Shared memory
__global__ void conv_shared(float *A, float *C, uint n, uint k, uint m) {
    // Shared memory for the input matrix
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];

    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;
    const uint row = blockIdx.x * BLOCK_SIZE + threadRow;
    const uint col = blockIdx.y * BLOCK_SIZE + threadCol;

    // Load elements into shared memory within bounds
    if (row < m && col < n) {
        A_shared[threadRow][threadCol] = A[row * n + col];
    } else {
        A_shared[threadRow][threadCol] = 0.0f; // Pad with zero if outside bounds
    }

    // Synchronize to ensure all threads have loaded the data into shared memory
    __syncthreads();

    // Perform convolution if within output bounds
    if (row < (m + 1 - k) && col < (n + 1 - k)) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                // Ensure the accesses are within shared memory bounds
                int sharedRow = threadRow + i - k / 2;
                int sharedCol = threadCol + j - k / 2;

                if (sharedRow >= 0 && sharedRow < BLOCK_SIZE && sharedCol >= 0 && sharedCol < BLOCK_SIZE) {
                    sum += A_shared[sharedRow][sharedCol] * window_cm[k * i + j];
                }
            }
        }
        C[(n + 1 - k) * row + col] = sum;
    }

    // Synchronize before finishing
    __syncthreads();
}

