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
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);

    init_matrix(a, N);
    init_matrix(b, N);

    // Set our block size and threads per thread block
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // Set up kernel launch parameters, so we can create 3d grid
    dim3 dimBlocks(threads, threads); // same as THREADS.x = 16
    dim3 dimGrid(blocks, blocks);

    // Launch our kernel
    matrixMult<<<dimGrid, dimBlocks>>>(a, b, c, N);
    cudaDeviceSynchronize(); // cudaMemcpy

    // Verify result by computing on CPU and comparing it to GPU
    verify_result(a, b, c, N);

    cout << "Program completed successfully." << endl;

    return 0;
}
