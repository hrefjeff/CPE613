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
    unsigned char* N_r, unsigned char* N_g, unsigned char* N_b, float* F, unsigned char* P, int r, int width, int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int filter_radius = 2 * r + 1;
    
    float a_r = 0.0;
    float a_g = 0.0;
    float a_b = 0.0;
    
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
                    a_r += F[fRow*FILTER_SIZE + fCol] * N_r[inputRow*width + inputCol];
                    a_g += F[fRow*FILTER_SIZE + fCol] * N_g[inputRow*width + inputCol];
                    a_b += F[fRow*FILTER_SIZE + fCol] * N_b[inputRow*width + inputCol];
                }
            }
        }
    }
    P[outRow*width+outCol] = (unsigned char)a_r + (unsigned char)a_g + (unsigned char)a_b;
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
    unsigned char* red,
    unsigned char* green,
    unsigned char* blue,
    float* filterMatrix,
    unsigned char* outputMatrix,
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
        red,
        green,
        blue,
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
    int total = img.total(); // total === rows*cols

    vector<unsigned char> h_red(img.rows * img.cols);
    vector<unsigned char> h_green(img.rows * img.cols);
    vector<unsigned char> h_blue(img.rows * img.cols);
    vector<unsigned char> h_output(img.rows * img.cols);

    // Allocate memory in host RAM
    for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
        for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
            auto & vec = img.at<Vec<uchar, 3>>(rowIdx, colIdx);
            int offset = rowIdx * img.cols + colIdx;
            h_red[offset] = vec[2];
            h_green[offset] = vec[1];
            h_blue[offset] = vec[0];
        }
    }

    // Create convolution filter
    vector<float> h_filter = {
        0.11, 0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11, 0.11,
        0.11, 0.11, 0.11, 0.11, 0.11
    };

    // vector<float> h_filter = {
    //     1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1,
    //     1, 1, 1, 1, 1
    // };

    // Allocate memory space on the device
    unsigned char * d_red = nullptr;
    unsigned char * d_green = nullptr;
    unsigned char * d_blue = nullptr;
    float* d_filter = nullptr;
    unsigned char * d_output = nullptr;

    size_t matrixByteSize = total * sizeof(unsigned char);
    size_t filterByteSize = FILTER_SIZE * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_red, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_green, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_blue, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_filter, filterByteSize));
    checkCudaErrors(cudaMalloc(&d_output, matrixByteSize));
    
    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_red, h_red.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_green, h_green.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_blue, h_blue.data(), matrixByteSize, cudaMemcpyHostToDevice)
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
        d_red,
        d_green,
        d_blue,
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

    // Copy result from gray matrix into matlab OpenCV input array format
    Mat opencv_output(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
        for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
            int offset = rowIdx * img.cols + colIdx;
            opencv_output.at<uchar>(rowIdx, colIdx) = h_output[offset];
        }
    }

    // Write img to gray.jpg
    imwrite("grayboiz.jpg", opencv_output);

    // Free memory
    cudaFree(d_red);
    cudaFree(d_blue);
    cudaFree(d_green);
    cudaFree(d_filter);
    cudaFree(d_output);

    printf("Made it to the end!\n");

    return 0;
}
