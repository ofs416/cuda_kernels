// Implementation of CNN kernel following based on the optimisations seen in gpu_functions.cu

#include <cuda_runtime.h>
#include "cnn_kernels.cuh"

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
