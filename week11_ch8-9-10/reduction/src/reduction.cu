#include <reduction.h>

__global__ void matrixMultiplication_kernel(
    float* dev_A,
    float* dev_B,
    float* dev_C,
    int N
) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    // First check that we are in the bounds of the matrix
    if ((ROW < N) && (COL < N)) {
        // iterate over row, and down column
        for (int k = 0; k < N; ++k) {
            tmpSum += dev_A[ROW * N + k] * dev_B[k * N + COL];
        }
        dev_C[ROW * N + COL] = tmpSum;
    }
}

void reduction (
  float* dev_A,
  float* dev_B,
  float* dev_C,
  int N
) {

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 blockSize(numOfThreads, numOfThreads);
    dim3 gridSize(numOfBlocks, numOfBlocks);

    matrixMultiplication_kernel<<<gridSize, blockSize>>> (dev_A, dev_B, dev_C, N); 
  
    checkCudaErrors(cudaGetLastError());
}

void simpleReduction () {}