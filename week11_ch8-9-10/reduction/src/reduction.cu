#include <reduction.h>

// 0
float hostReduction (thrust::host_vector<float> input) {
    float sum = 0;
    for(int i = 0; i < input.size(); ++i)
        sum += input[i];
    return sum;
}

// 1
__global__ void sequentialReduction_kernel (
    float* input,
    float* output,
    int N
){
    for(int i = 0; i < N; ++i)
        *output += input[i];
}

// 2
__global__ void atomicsReduction_kernel (float* input, float* sum, int N){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        atomicAdd(sum, input[i]);
    }
}

// 2
__global__ void parallelAtomicsReduction_kernel (
    float* input,
    float* sum,
    int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        atomicAdd(sum, input[i]);
    }
}

// 3
__global__ void segmentedReduction_kernel (float* input, float* output) {}

// 4
__global__ void segmentedCoalescingReduction_kernel (float* input, float* output) {}

// 5
__global__ void sharedMemoryReduction_kernel (float* input, float* output) {}

// 6
__global__ void coarsenedSharedMemoryReduction_kernel (float* input, float* output) {}

void sequentialReduction(float* input, float* output, int N){

    sequentialReduction_kernel<<<1, 1>>> (input, output, N); 
  
    checkCudaErrors(cudaGetLastError());
}

void reduction(float* input, float* output, int N){

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    parallelAtomicsReduction_kernel<<<numOfBlocks, numOfThreads>>> 
    (
        input,
        output,
        N
    ); 
  
    checkCudaErrors(cudaGetLastError());
}