#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Constant memory
__constant__ float filter_cm[BLOCK_SIZE];

__global__ void conv_1dhz_cm(float *input, float *output, uint width,
                                                     uint height, uint f_size) {
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint stride = blockDim.x * gridDim.x;
    
    const uint total_size = width * height;
    const uint padding = f_size / 2;  // Assuming odd kernel size

    for (uint i = tid; i < total_size; i += stride) {
        uint row = i / width;
        uint col = i % width;
        
        float sum = 0.0f;
        for (int j = 0; j < f_size; j++) {
            int input_col = col + j - padding;
            if (input_col >= 0 && input_col < width) {
                sum += input[row * width + input_col] * filter_cm[j];
            }
            // Implicit zero-padding: we don't add anything for out-of-bounds inputs
        }
        output[row * width + col] = sum;
    }
}


void check_cuda_error(cudaError_t error, const char *function_name) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function_name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


int main() {
    // Image and kernel parameters
    const unsigned int width = 1024;
    const unsigned int height = 1024;
    const unsigned int kernel_size = 5;
    const unsigned int image_size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(image_size);
    float *h_output = (float*)malloc(image_size);
    float *h_filter = (float*)malloc(kernel_size * sizeof(float));

    // Initialize input image and kernel (example initialization)
    for (unsigned int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values between 0 and 1
    }
    for (unsigned int i = 0; i < kernel_size; ++i) {
        h_filter[i] = 1.0f / kernel_size;  // Simple averaging kernel
    }

    // Allocate device memory
    float *d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, image_size), "cudaMalloc d_input");
    check_cuda_error(cudaMalloc(&d_output, image_size), "cudaMalloc d_output");

    // Copy input data to device
    check_cuda_error(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice), "cudaMemcpy H2D input");
    check_cuda_error(cudaMemcpyToSymbol(filter_cm, h_filter, kernel_size * sizeof(float)), "cudaMemcpyToSymbol kernel");

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    conv_1dhz_cm<<<gridDim, blockDim>>>(d_input, d_output, width, height, kernel_size);

    // Check for kernel launch errors
    check_cuda_error(cudaGetLastError(), "Kernel launch");

    // Wait for kernel to finish
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Free memory
    check_cuda_error(cudaFree(d_input), "cudaFree d_input");
    check_cuda_error(cudaFree(d_output), "cudaFree d_output");
    free(h_input);
    free(h_output);
    free(h_filter);

    return 0;
}