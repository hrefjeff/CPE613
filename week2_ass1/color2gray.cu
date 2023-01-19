#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ 
void colorToGrayScaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row*width + col;

        // One can think of the RGB image having CHANNEL
        // times more columns than the gray scale image
        int rgbOffset = grayOffset*3; // 3 channels
        unsigned char r = Pin[rgbOffset    ];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g+ 0.07f*b;
    }
}

int main() {

    // Set our problem size
    const int WIDTH = 810;
    const int HEIGHT = 456;
    int *hostMatrix;
    int *deviceMatrix;
    // TODO: Find out what Pin and Pout are
    
    // Allocate memory on the host
    cudaMallocHost(&hostMatrix, row * col * sizeof(int));

    // Set our block size and threads per thread block
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 dimBlocks(threads, threads); // TODO: Read book for how to calculate
    dim3 dimGrid(blocks, blocks);

    // Fill the host matrix with data
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            hostMatrix[i * col + j] = /* TODO: GET VALUES FROM RGB MATRIX */;
        }
    }

    // Allocate memory on the device
    cudaMalloc(&deviceMatrix, row * col * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceMatrix, hostMatrix, row * col * sizeof(int), cudaMemcpyHostToDevice);

    // Perform CUDA computations on deviceMatrix
    // Launch our kernel
    matrixMult<<<dimGrid, dimBlocks>>>(a, b, c, HEIGHT, WIDTH);

    // Free memory
    cudaFree(deviceMatrix);
    cudaFreeHost(hostMatrix);

    return 0;
}
