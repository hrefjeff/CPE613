#include <convolution.h>

__global__ void convolve_1d_time_kernel (
    float *input, 
    float *kernel,
    float *output,
    int N,
    int K
) {
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

void convolve_1d (
    float* input,
    float* filter,
    float* output,
    int N,
    int K
){

    int numOfThreads = 32;
    int numOfBlocks = ((N + K - 1) + numOfThreads - 1) / numOfThreads;

    convolve_1d_time_kernel<<<numOfBlocks, numOfThreads>>> 
    (
        input,
        filter,
        output,
        N,
        K
    ); 
  
    checkCudaErrors(cudaGetLastError());
}