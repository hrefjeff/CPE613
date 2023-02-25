#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace cv;
using namespace std;

// 2x2 convolutional mask
#define FILTER_RADIUS 3
// 2*r+1
#define FILTER_SIZE 7
// Allocate mask in constant memory
__constant__ int FILTER[7 * 7];

__global__
void conv_2D_basic_kernel (
    float* N, float* F, float* P, int r, int width, int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int filter_radius = 2 * r + 1;
    
    float accumulator = 0.0;

    // Iterate through the filter rows
    for (int fRow = 0; fRow < filter_radius; fRow++) {
        // Iterate through the filter columns
        for (int fCol = 0; fCol < filter_radius; fCol++) {
            int inputRow = outRow - r + fRow;
            int inputCol = outCol - r + fCol;
            // If we're within the height of the input matrix
            if (inputRow >= 0 && inputRow < height) {
                // If we're within the width of the input matrix
                if (inputCol >= 0 && inputCol < width) {
                    accumulator += F[fRow*FILTER_SIZE + fCol] * N[inputRow*width + inputCol];
                }
            }
        }
    }
    P[outRow*width+outCol] = accumulator;
}

__global__
void conv_2D_const_mem_kernel (
    float* inputMatrix,
    float* outputMatrix,
    int r,
    int width,
    int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    float tmp = 0.0f;
    
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height) {
                if (inCol >= 0 && inCol < width) {
                    tmp += FILTER[fRow*width+fCol] * inputMatrix[inRow*width+ inCol];
                }
            }
        }
    }
    outputMatrix[outRow*width+outCol] = tmp;
}

void convolution (
    float* inputMatrix,
    float* filterMatrix,
    float* outputMatrix,
    int radius,
    int numRows,
    int numCols
) {
    int blockWidth = 32;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 blockSize(blockWidth, blockWidth);
    dim3 gridSize(
        (numCols + blockWidth - 1) / blockWidth,
        (numRows + blockWidth - 1) / blockWidth
    );

    // Perform CUDA computations on deviceMatrix, Launch Kernel
    conv_2D_basic_kernel<<<
        gridSize,
        blockSize
    >>> (
        inputMatrix,
        filterMatrix,
        outputMatrix,
        radius,
        numCols,
        numRows
    );
    // conv_2D_const_mem_kernel<<<
    //     gridSize,
    //     blockSize
    // >>> (
    //     inputMatrix,
    //     outputMatrix,
    //     radius,
    //     numCols,
    //     numRows
    // );

    checkCudaErrors(
        cudaGetLastError()
    );
}

int main() {

    // Initialize image we want to work with
    Mat img = imread("thethreeamigos.jpeg", IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int total = img.total(); // total === rows*cols

    // Allocate memory in host RAM
    // Convert it to a 1D array
    Mat flat = img.reshape(1, total*img.channels());
    vector<float> h_matrix(flat.data, flat.data + flat.total());
    vector<float> h_output(flat.data, flat.data + flat.total());

    // Create convolution filter
    // vector<float> h_filter = {
    //     0.11, 0.11, 0.11, 0.11, 0.11,
    //     0.11, 0.11, 0.11, 0.11, 0.11,
    //     0.11, 0.11, 0.11, 0.11, 0.11,
    //     0.11, 0.11, 0.11, 0.11, 0.11,
    //     0.11, 0.11, 0.11, 0.11, 0.11
    // };

    vector<float> h_filter = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    };

    // // Allocate memory space on the device
    float* d_matrix = nullptr;
    float* d_filter = nullptr;
    float* d_output = nullptr;
    size_t matrixByteSize = total * sizeof(float);
    size_t filterByteSize = FILTER_SIZE * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_matrix, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_filter, filterByteSize));
    checkCudaErrors(cudaMalloc(&d_output, matrixByteSize));
    
    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_matrix, h_matrix.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_filter, h_filter.data(), filterByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_output, h_output.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    // checkCudaErrors(
    //     cudaMemcpyToSymbol(FILTER, h_filter.data(), filterByteSize)
    // );

    convolution (
        d_matrix,
        d_filter,
        d_output,
        FILTER_RADIUS,
        img.cols,
        img.rows
    );

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (h_output.data(), d_output, matrixByteSize, cudaMemcpyDeviceToHost)
    );

    // Reconstruct the image from 1d array
    Mat restored = Mat(rows, cols, img.type(), h_output.data());

    // Write img to gray.jpg
    imwrite("grayboiz.jpg", restored);

    // Free memory
    cudaFree(d_filter);
    cudaFree(d_matrix);
    cudaFree(d_output);

    printf("Made it to the end!\n");

    return 0;
}
