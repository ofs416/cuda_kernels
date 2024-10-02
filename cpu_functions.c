include <stdio.h>

void vectorAdditionCPU (float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++){
    c[i] = a[i] + b[i];
  }
}

void matrixAdditionCPU (float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      C[i + n * j] = A[i + n * j] + B[i + n * j];
    }
  }
}

void matrixMultiplicationCPU (float *A, float *B, float *C, int n, int m, int k) {
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
