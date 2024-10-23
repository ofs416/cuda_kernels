#ifndef GPU_FUNCTIONS_CUH_
#define GPU_FUNCTIONS_CUH_

__global__ void matrixAdditionGPU(float *A, float *B, float *C, int n);
__global__ void gemm(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_gmc(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_smem(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_1DBlockTiling(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_2DBlockTiling(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_vectorised(float *A, float *B, float *C, int n, int k, int m);

#endif