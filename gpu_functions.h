#ifndef GPU_FUNCTIONS_H_
#define GPU_FUNCTIONS_H_

__global__ void matrixAdditionGPU(float *A, float *B, float *C, int n);
__global__ void matrixMultiplicationGPU(float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_gmc (float *A, float *B, float *C, int n, int k, int m);
__global__ void gemm_smem (float *A, float *B, float *C, int n, int k, int m);

#endif // GPU_FUNCTIONS_H_