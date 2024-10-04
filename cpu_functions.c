#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024
#define K 512
#define M 2048

// Timing
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Matrix initialisation
void MatrixInit(float *a, int row, int col) {
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

void main() {
    // Initialise pointers to the matrices
    float *A, *B, *C;

    // Calculate mem size
    int sizeA = N * K * sizeof(float);
    int sizeB = K * M * sizeof(float);
    int sizeC = N * M * sizeof(float);

    // Allocate memory
    A = (float*)malloc(sizeA);
    B = (float*)malloc(sizeB);
    C = (float*)malloc(sizeC);

    // Initialise the matrices
    srand(time(NULL));
    MatrixInit(A, N, K);
    MatrixInit(B, K, M);

    // Warm-up
    printf("Warming up\n");
    for (int i = 0; i < 5; i++){
        matrixMultiplicationCPU(A, B, C, N, K, M);
    }

    // Benchmark
    printf("Benchmarking\n");
    double totalTime = 0.0;
    for (int i = 0; i < 20; i++){
        double startTime = getTime();
        matrixMultiplicationCPU(A, B, C, N, K, M);
        double endTime = getTime();
        totalTime += endTime - startTime;
    }
    double avgTime = totalTime / 20.0;
    printf("CPU average time: %f ms\n", (avgTime * 1e6f));

    // Free memory
    free(A);
    free(B);
    free(C);
}
