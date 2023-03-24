/* Include my stuff */
#include <convolution.h>
#include <Timer.hpp>

/* Include C++ stuff */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#define N 8
#define K 4

int main() {
    float *h_input = new float[N];
    float *h_filter = new float[K];
    float *h_output = new float[N - K + 1];
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_filter, K * sizeof(float));
    cudaMalloc((void **)&d_output, (N + K - 1) * sizeof(float));

    h_input[0] = 0.0;
    h_input[1] = -1.0;
    h_input[2] = -1.2;
    h_input[3] = 2.0;
    h_input[4] = 1.4;
    h_input[5] = 1.4;
    h_input[6] = 0.6;
    h_input[7] = 0.0;

    h_filter[0] = 1.0;
    h_filter[1] = -0.5;
    h_filter[2] = -0.25;
    h_filter[3] = -0.1;

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, K * sizeof(float), cudaMemcpyHostToDevice);
    convolve_1d(d_input, d_filter, d_output, N, K);
    cudaMemcpy(h_output, d_output, (N + K - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    for (int i = 0; i < N + K - 1; i++) {
        printf ("%20.16e\n", h_output[i]);
    }
    std::cout << std::endl;


    delete[] h_input;
    delete[] h_filter;
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    return 0;
}
