#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__device__ float doSomethingCool(float input)
{

    return input * input;
}

__global__ void preVolta_kernel(float* input, float* output, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    if (tid < 16) sleep(5000);

    // Loop over the input data, processing one element per thread
    for (int i = tid; i < n; i += stride)
    {
        // Process the element and store the result
        float result = doSomethingCool(input[i]);
        output[i] = result;
    }
}

__global__ void postVolta_kernel(float* input, float* output, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    if (tid < 16) sleep(5000);

    // Loop over the input data, processing one element per thread
    for (int i = tid; i < n; i += stride)
    {
        // Process the element and store the result
        float result = doSomethingCool(input[i]);
        output[i] = result;
    }
}

void oneDimensionalArrayProcessor (
    float* d_input,
    float* d_output,
    int N
){
    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    // Perform CUDA computations on deviceMatrix, Launch Kernel
    postVolta_kernel<<<
        numOfBlocks,
        numOfThreads
    >>> (
        d_input,
        d_output,
        N
    );

    checkCudaErrors(
        cudaGetLastError()
    );
}


int main() {

    int N = 1024;

    // Allocate memory in host RAM
    vector<float> inputarr(N, 1.0);
    vector<float> outputarr(N, 1.0);

    // initialize the matrix1 and matrix2 to some arbitrary values
    for (int i=0; i<N; i++){
        inputarr[N] = rand() % 1000;
    }

    // Allocate memory space on the device
    float* dev_input = nullptr;
    float* dev_output = nullptr;
    size_t byteSize = N * sizeof(float);
    checkCudaErrors(cudaMalloc(&dev_input, byteSize));
    checkCudaErrors(cudaMalloc(&dev_output, byteSize));

    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (
            dev_input,
            inputarr.data(),
            byteSize,
            cudaMemcpyHostToDevice
        )
    );
    checkCudaErrors(
        cudaMemcpy (
            dev_output,
            outputarr.data(),
            byteSize,
            cudaMemcpyHostToDevice
        )
    );

    oneDimensionalArrayProcessor(dev_input, dev_output, N);

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (
            outputarr.data(),
            dev_output,
            byteSize,
            cudaMemcpyDeviceToHost
        )
    );

    // Print C
    for (int i=0; i<N; i++){
        cout << outputarr[N] << " ";
    }

    cout << endl;

    // Free memory
    cudaFree(dev_input);
    cudaFree(dev_output);

    printf("Made it to the end!\n");

    return 0;
}