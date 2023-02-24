#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace cv;
using namespace std;

// 2x2 convolutional mask
#define FILTER_RADIUS 1
// 2*r+1
#define FILTER_SIZE 3
// Allocate mask in constant memory
__constant__ int filter[4 * 4];

__global__
void conv_2D_basic_kernel (
    float* N, float* F, float* P, int r, int width, int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    
    float Pvalue = 0.0f;
    int radius = 2 * r + 1;
    
    for (int fRow = 0; fRow < radius; fRow++) {
        for (int fCol = 0; fCol < radius; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += filter[fRow*FILTER_SIZE+fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow*width+outCol] = Pvalue;
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
                    tmp += filter[fRow*width+fCol] * inputMatrix[inRow*width+ inCol];
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

    int radius = 4;

    // Initialize image we want to work with
    Mat img = imread("thethreeamigos.jpeg", IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int total = img.total(); // total === rows*cols

    cout << rows << " " << cols << endl; 

    // Allocate memory in host RAM
    // Convert it to a 1D array
    Mat flat = img.reshape(1, total*img.channels());
    vector<float> h_inputMatrix(flat.data, flat.data + flat.total());
    vector<float> h_outputMatrix(flat.data, flat.data + flat.total());

    // Create convolution filter
    vector<float> h_filter = {
        0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11
    };

    // // Allocate memory space on the device
    float* d_inputMatrix = nullptr;
    float* d_filter = nullptr;
    float* d_outputMatrix = nullptr;
    size_t matrixByteSize = total * sizeof(float);
    size_t filterByteSize = h_filter.size() * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_inputMatrix, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_filter, filterByteSize));
    checkCudaErrors(cudaMalloc(&d_outputMatrix, matrixByteSize));
    
    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_inputMatrix, h_inputMatrix.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_filter, h_filter.data(), filterByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_outputMatrix, h_outputMatrix.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpyToSymbol(filter, h_filter.data(), filterByteSize)
    );

    convolution (
        d_inputMatrix,
        d_filter,
        d_outputMatrix,
        radius,
        img.cols,
        img.rows
    );

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (h_outputMatrix.data(), d_outputMatrix, matrixByteSize, cudaMemcpyDeviceToHost)
    );

    // Reconstruct the image from 1d array
    Mat restored = Mat(rows, cols, img.type(), h_outputMatrix.data());

    // Write img to gray.jpg
    imwrite("grayboiz.jpg", restored);

    // Free memory
    cudaFree(d_filter);
    cudaFree(d_inputMatrix);
    cudaFree(d_outputMatrix);

    printf("Made it to the end!\n");

    return 0;
}
