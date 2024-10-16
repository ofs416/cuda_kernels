#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
  for (int i = 0; i < m; i++){
    for (int j = 0; j < n; j++){
      float sum = 0.0f;
      for (int l = 0; l < k; l++){
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

void conv_1dhz_cpu(const float* input, float* output, int width, int height, const float* filter, int filter_size) {
    int padding = filter_size / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int i = 0; i < filter_size; ++i) {
                int input_x = x + i - padding;
                if (input_x >= 0 && input_x < width) {
                    sum += input[y * width + input_x] * filter[i];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

void compare_results(float* cpu_output, float* gpu_output, 
                   int width, int height, float tolerance) {
    int mismatches = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            float diff = fabs(cpu_output[idx] - gpu_output[idx]);
            if (diff > tolerance) {
                mismatches += 1;
            }
        }
    }
    printf("%i mismathces \n", mismatches);
}

