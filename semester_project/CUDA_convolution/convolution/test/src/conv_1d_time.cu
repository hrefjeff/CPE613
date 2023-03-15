/*
    Testing for 1D Time Domain Convolution
    To compile: nvcc test.cu -o test.o -g -G
    To debug: cuda-gdb test.o
    Useful debug tools:
        set cuda coalescing off
        break main
        break 28
        run
        continue
        info cuda threads
        print result
*/

#include <iostream>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

#define N 8
#define K 4

__global__ void convolve_1d(float *input, float *kernel, float *output) {
    int idxInput = threadIdx.x + blockIdx.x * blockDim.x;
    if (idxInput > N + K - 1) return;

    float result = 0.0;
    for (int idxFilter = 0; idxFilter < K; idxFilter++) {
        if((idxInput - idxFilter) < 0 || (idxInput - idxFilter) >= N)
            result += 0;
        else
            result += (float)(kernel[idxFilter] * input[idxInput - idxFilter]);
    }
    output[idxInput] = result;
}

int main() {
    float *h_input = new float[N];
    float *h_kernel = new float[K];
    float *h_output = new float[N - K + 1];
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_kernel, K * sizeof(float));
    cudaMalloc((void **)&d_output, (N + K - 1) * sizeof(float));

    h_input[0] = 0.0;
    h_input[1] = -1.0;
    h_input[2] = -1.2;
    h_input[3] = 2.0;
    h_input[4] = 1.4;
    h_input[5] = 1.4;
    h_input[6] = 0.6;
    h_input[7] = 0.0;

    h_kernel[0] = 1.0;
    h_kernel[1] = -0.5;
    h_kernel[2] = -0.25;
    h_kernel[3] = -0.1;

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K * sizeof(float), cudaMemcpyHostToDevice);
    convolve_1d<<<numOfBlocks, numOfThreads>>>(d_input, d_kernel, d_output);
    cudaMemcpy(h_output, d_output, (N + K - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    for (int i = 0; i < N + K - 1; i++) {
        printf ("%20.16e\n", h_output[i]);
    }
    std::cout << std::endl;


    delete[] h_input;
    delete[] h_kernel;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
