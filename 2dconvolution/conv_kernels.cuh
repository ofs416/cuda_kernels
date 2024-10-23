#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H

__global__ void conv_naive (float *A, float *B, float *C, uint n, uint k, uint m);
__global__ void conv_gmc (float *A, float *B, float *C, uint n, uint k, uint m);
__global__ void conv_cm (float *A, float *C, uint n, uint k, uint m);
__global__ void conv_shared (float *A, float *C, uint n, uint k, uint m);

#endif 