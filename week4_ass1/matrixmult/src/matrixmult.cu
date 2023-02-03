#include <matrixmult.h>

__global__ void matrixMultiplication_kernel(
    float* dev_A,
    float* dev_B,
    float* dev_C,
    int N
) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += dev_A[ROW * N + i] * dev_B[i * N + COL];
        }
    }
    dev_C[ROW * N + COL] = tmpSum;
}

void matrixMultiplication (
  float* dev_A,
  float* dev_B,
  float* dev_C,
  int N
) {

  int blockSize;    // The launch configurator returned block size 
  int minGridSize;  // The minimum grid size needed to achieve the 
                    // maximum occupancy for a full device launch 
  int gridSize;     // The actual grid size needed, based on input size 

  checkCudaErrors (
    cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize, 
      matrixMultiplication_kernel,
      0,
      0
    )
  ); 
  
  // Round up according to array size 
  gridSize = (N + blockSize - 1) / blockSize;

  matrixMultiplication_kernel<<<gridSize, blockSize>>> (dev_A, dev_B, dev_C, N); 
  
  checkCudaErrors(cudaGetLastError());

}

