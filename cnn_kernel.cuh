#ifndef CNN_FUNCTIONS_H_
#define CNN_KERNELS_H

__global__ void cnn (float *A, float *B, float *C, uint n, uint k, uint m);

#endndef CNN_FUNCTIONS_H_