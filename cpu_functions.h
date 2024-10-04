#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

void initMatrix(float *a, int row, int col);
void matrixMultiplicationCPU(float *A, float *B, float *C, int n, int k, int m);
double getTime();

#endif // CPU_FUNCTIONS_H
