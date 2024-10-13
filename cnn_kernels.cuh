#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H

__global__ void conv_naive (float *A, float *B, float *C, uint n, uint k, uint m);

#endif // CNN_KERNELS_H