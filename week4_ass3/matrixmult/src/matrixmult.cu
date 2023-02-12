#include <matrixmult.h>
#define TILE_DIM 32

__global__ void tiledMatrixMultiplication_kernel (
    float* A,
    float* B,
    float* C,
    int N
) {
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(unsigned int tile = 0; tile < N/TILE_DIM; ++tile) {
        // Load tile to shared memory
        A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
        __syncthreads();
        // Compute with tile
        for(unsigned int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row*N + col] = sum;
}

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

    int numOfThreads = 32;
    int numOfBlocks = (N + numOfThreads - 1) / numOfThreads;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 blockSize(numOfThreads, numOfThreads);
    dim3 gridSize(numOfBlocks, numOfBlocks);

    matrixMultiplication_kernel<<<gridSize, blockSize>>> (dev_A, dev_B, dev_C, N); 
  
    checkCudaErrors(cudaGetLastError());
}