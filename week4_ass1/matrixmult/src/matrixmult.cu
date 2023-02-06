#include <matrixmult.h>
#define TILE_WIDTH 32

__global__ void tiledMatrixMultiplication_kernel (
    float* M,
    float* N,
    float* P,
    int Width
) {
    // Create arrays that are local to each block
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {

        // Colaborative loading of M and N tiles into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    P[Row*Width + Col] = Pvalue;
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

    int blockWidth = 32;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 blockSize(blockWidth, blockWidth);
    dim3 gridSize(
        (N + blockWidth - 1) / blockWidth,
        (N + blockWidth - 1) / blockWidth
    );

    tiledMatrixMultiplication_kernel<<<gridSize, blockSize>>> (dev_A, dev_B, dev_C, N); 
  
    checkCudaErrors(cudaGetLastError());
}