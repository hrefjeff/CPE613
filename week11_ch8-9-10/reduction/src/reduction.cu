#include <reduction.h>
#define BLOCK_DIM 32
#define COARSE_FACTOR 1

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
__global__ void segmentedReduction_kernel (float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i+BLOCK_DIM];


    for(unsigned int stride = blockDim.x/2; stride >=1; stride /=2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t+stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

// 4 (TODO)
__global__ void segmentedCoalescingReduction_kernel (float* input, float* output) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    for(unsigned int stride = BLOCK_DIM; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
}

// 5 (TODO)
__global__ void sharedMemoryReduction_kernel (float* input, float* output) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Load data to shared memory
    __shared__ float input_s[BLOCK_DIM];
    input_s[threadIdx.x] = input[i] + input[i + BLOCK_DIM];
    __syncthreads();

    // Reduction tree in shared memory
    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

}

// 6 (TODO)
__global__ void coarsenedSharedMemoryReduction_kernel (
    float* input,
    float* output
) {
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Load data to shared memory
    __shared__ float input_s[BLOCK_DIM];
    float threadSum = 0.0f;
    for(unsigned int c = 0; c < COARSE_FACTOR*2; ++c) {
        threadSum += input[i + c*BLOCK_DIM];
    }
    input_s[threadIdx.x] = threadSum;

    __syncthreads();

    // Reduction tree in shared memory
    for(unsigned int stride = BLOCK_DIM/2; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

}

void sequentialReduction(float* input, float* output, int N){

    sequentialReduction_kernel<<<1, 1>>> (input, output, N); 
  
    checkCudaErrors(cudaGetLastError());
}

void reduction(float* input, float* output, int N){

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    segmentedReduction_kernel<<<numOfBlocks, numOfThreads>>> 
    (
        input,
        output
    ); 
  
    checkCudaErrors(cudaGetLastError());
}