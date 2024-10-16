#ifndef CPU_FUNCTIONS_H_
#define CPU_FUNCTIONS_H_

void initMatrix(float *a, int row, int col);
void matrixMultiplicationCPU(float *A, float *B, float *C, int n, int k, int m);
double getTime();
void conv_1dhz_cpu(const float* input, float* output, int width, 
int height, const float* filter, int filter_size);
int compare_results(const float* cpu_output, const float* 
gpu_output, int width, int height, float tolerance);


#endif // CPU_FUNCTIONS_H_
