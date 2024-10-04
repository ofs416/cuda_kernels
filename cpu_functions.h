#ifndef CPU_FUNCTIONS_H_
#define CPU_FUNCTIONS_H_

void initMatrix(float *a, int row, int col);
void matrixMultiplicationCPU(float *A, float *B, float *C, int n, int k, int m);
double getTime();

#endif // CPU_FUNCTIONS_H_
