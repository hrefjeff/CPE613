#include <matrixmult.h>

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

void matrixMultiplication (
  float* dev_A,
  float* dev_B,
  float* dev_C,
  int N
) {

    int blockWidth = 32;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 blockSize(blockWidth, blockWidth);
    dim3 gridSize(
        (N + blockWidth - 1) / blockWidth,
        (N + blockWidth - 1) / blockWidth
    );

    matrixMultiplication_kernel<<<gridSize, blockSize>>> (dev_A, dev_B, dev_C, N); 
  
    checkCudaErrors(cudaGetLastError());
}