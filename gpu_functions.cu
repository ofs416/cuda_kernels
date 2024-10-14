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

// Naive CUDA kernel for matrix multiplication (M x K) @ (K x N)
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

// Global Memory Coalescing CUDA kernel for matrix multiplication (M x K) @ (K x N)
// Increases performance by grouping memory accesses of threads that are in the same warp
// and executed as one
// Each warp contains 32 threads and memory accesses can be 32B, 64B and 128B
// To take advantage of 128B single access, the floats should be consecutive in memory
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

// Shared Memory CUDA kernel for matrix multiplication (M x K) @ (K x N)
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
template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void gemm_1DBlockTiling (float *A, float *B, float *C, uint n, uint k, uint m) {

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

        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                threadResults[resIdx] += A_shared[(threadRow * TM + resIdx) * BK + dotIdx] 
                    * B_shared[dotIdx * BN + threadCol];
            }
        }
        __syncthreads();
    }
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * n + threadCol] = threadResults[resIdx];
    }
}

// This still suffers from MIO stall
// Further mitigated via tiling in 2D, not just 1D
template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_2DBlockTiling (float *A, float *B, float *C, int n, int k, int m) {

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint blkIdx = 0; blkIdx < k; blkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            A_shared[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * k + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            B_shared[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * n + innerColB];
        }
        __syncthreads();

        A += BK; 
        B += BK * n; 

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = A_shared[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = B_shared[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN] =
                threadResults[resIdxM * TN + resIdxN];
        }
    }
}

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_vectorised (float *A, float *B, float *C, int n, int k, int m) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    // Load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint blkIdx = 0; blkIdx < k; blkIdx += BK) {
        float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * k + innerColA * 4])[0];
        A_shared[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        A_shared[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        A_shared[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        A_shared[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&B_shared[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * n + innerColB * 4])[0];
        __syncthreads();

        A += BK;  
        B += BK * n; 

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = A_shared[dotIdx * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = B_shared[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN])[0];
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN])[0] =
            tmp;
        }
    }
}

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void gemm_warptiling (float *A, float *B, float *C, int n, int k, int m) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    // Load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint blkIdx = 0; blkIdx < k; blkIdx += BK) {
        float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * k + innerColA * 4])[0];
        A_shared[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        A_shared[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        A_shared[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        A_shared[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4 *>(&B_shared[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * n + innerColB * 4])[0];
        __syncthreads();

        A += BK;  
        B += BK * n; 

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
            regM[wSubRowIdx * TM + i] =
                A_shared[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                    threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
            regN[wSubColIdx * TN + i] =
                B_shared[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                    threadColInWarp * TN + i];
            }
        }
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                    regM[wSubRowIdx * TM + resIdxM] *
                    regN[wSubColIdx * TN + resIdxN];
                }
            }
            }
        }
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN])[0];
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];
            reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN])[0] =
            tmp;
        }
    }
}