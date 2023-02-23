#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace cv;
using namespace std;

__global__ 
void rgb2gray_kernel (
    unsigned char* red_d,
    unsigned char* green_d,
    unsigned char* blue_d,
    unsigned char* gray_d,
    int numCols,
    int numRows
) {

    for (
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        rowIdx < numRows;
        rowIdx += blockDim.y * gridDim.y
    ) {
        for (
            int colIdx = threadIdx.x + blockIdx.x * blockDim.x;
            colIdx < numCols;
            colIdx += blockDim.x * gridDim.x
        ) {
            int offset = rowIdx * numCols + colIdx;
            gray_d[offset] = (unsigned char)(
                (float)red_d[offset] * 3.0 / 10.0 +
                (float)green_d[offset] * 6.0 / 10.0 +
                (float)blue_d[offset] * 1.0 / 10.0
            );
        }
    }

}

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
                Pvalue += F[fRow*width+fCol] * N[inRow*width + inCol];
            }
        }
    }
    P[outRow*width+outCol] = Pvalue;
}

void convolution (
    float* inputMatrix,
    float* convMatrix,
    float* outputMatrix,
    int radius,
    int numRows,
    int numCols
) {
    int blockWidth = 16;

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
        convMatrix,
        outputMatrix,
        radius,
        numCols,
        numRows
    );

    checkCudaErrors(
        cudaGetLastError()
    );
}

int main() {

    int radius = 3;

    // Initialize image we want to work with
    Mat img = imread("thethreeamigos.jpeg", IMREAD_COLOR);
    int rows = img.rows;
    int cols = img.cols;
    int total = img.total(); // total === rows*cols

    // Allocate memory in host RAM
    // Convert it to a 1D array
    Mat flat = img.reshape(1, total*img.channels());
    vector<float> h_inputMatrix(flat.data, flat.data + flat.total());
    vector<float> h_outputMatrix(flat.data, flat.data + flat.total());

    // Create convolution filter
    vector<float> h_convMatrix = {
        0.11, 0.11, 0.11,
        0.11, 0.11, 0.11,
        0.11, 0.11, 0.11
    };

    // // Allocate memory space on the device
    float* d_inputMatrix = nullptr;
    float* d_convMatrix = nullptr;
    float* d_outputMatrix = nullptr;
    size_t matrixByteSize = total * sizeof(float);
    size_t convByteSize = h_convMatrix.size() * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_inputMatrix, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_convMatrix, convByteSize));
    checkCudaErrors(cudaMalloc(&d_outputMatrix, matrixByteSize));

    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_inputMatrix, h_inputMatrix.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_convMatrix, h_convMatrix.data(), convByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_outputMatrix, h_outputMatrix.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );

    convolution (
        d_inputMatrix,
        d_convMatrix,
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
    cudaFree(d_convMatrix);
    cudaFree(d_inputMatrix);
    cudaFree(d_outputMatrix);

    printf("Made it to the end!\n");

    // return 0;
}
