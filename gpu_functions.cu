#include <cuda_runtime.h>
#include "gpu_functions.cuh"

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
__global__ void gemm (float *A, float *B, float *C, int n, int k, int m) {
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

// Shared Memory CUDA kernel for matrix multiplication (N x K) @ (K x M)
// Shared memory is has a size of O(KB) and bandwidth of >10,000GB/s
// Global memory has a bandwidth of <1,000GB/s 
// Each thread block is responsible for computing a sub-matrix of C
// Each thread block loads a tile of A and B into shared memory
__global__ void gemm_smem(float *A, float *B, float *C, int n, int k, int m) {
    __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * k;                  
    B += cCol * BLOCK_SIZE;                       
    C += cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE; 

    float tmp = 0.0f;
    for (int blkIdx = 0; blkIdx < k; blkIdx += BLOCK_SIZE) {
        A_shared[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * k + threadCol];
        B_shared[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * n + threadCol];

        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * n;

        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            tmp += 
            A_shared[threadRow * BLOCK_SIZE + dotIdx] * B_shared[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * n + threadCol] = tmp;
}

// SMEM per block is roughly 9KB (much less than the potential 48KB) 
// Although this is not the limiting factor here, instead the function above
// suffers from MIO stalls as it waits for SMEM accesses to return
// This can be mitigated by using 1D blocktiling to calc multiple results per thread
// Now each thread is responsible for computing a column of the block in C
__global__ void gemm_1DBlockTiling (float *A, float *B, float *C, int n, int k, int m) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    float threadResults[TM] = {0.0};
    for (uint blkIdx = 0; blkIdx < k; blkIdx += BK) {
        A_shared[innerRowA * BK + innerColA] = A[innerRowA * k + innerColA];
        B_shared[innerRowB * BN + innerColB] = B[innerRowB * n + innerColB];
        __syncthreads();

        A += BK;
        B += BK * n;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = B_shared[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    A_shared[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * n + threadCol] = threadResults[resIdx];
    }
}