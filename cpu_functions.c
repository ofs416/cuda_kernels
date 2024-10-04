#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Timing
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Matrix initialisation
void initMatrix(float *a, int row, int col) {
    for (int i = 0; i < row * col; i++){
        a[i] = (float)rand() / RAND_MAX;
    }
}

// Note the next two function use pointers to existing variables.
// Furthermore, the matrices are stored in 1D arrays that are column-major.
void matrixAdditionCPU (float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      C[i + n * j] = A[i + n * j] + B[i + n * j];
    }
  }
}

// (N x K) @ (K x M)
void matrixMultiplicationCPU (float *A, float *B, float *C, int n, int k, int m) {
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      float sum = 0.0f;
      for (int l = 0; l < k; l++){
        sum += A[i * k + l] * B[l * m + j];
      }
      C[i * m + j] = sum;
    }
  }
}

