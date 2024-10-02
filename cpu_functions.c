include <stdio.h>

void vectorAdditionCPU (float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++){
    c[i] = a[i] + b[i];
  }
}

void matrixAdditionCPU (float *a, float *b, float *c, int n, int m) {
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      c[i][j] = a[i][j] + b[i][j];
    }
  }
}
