#include <cstdlib>
#include <iostream>
#include <vector>
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
__constant__ int filter[FILTER_SIZE * FILTER_SIZE];

__global__
void conv_2D_basic_kernel (
    unsigned char* N, float* F, unsigned char* P, int r, int width, int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    
    int filter_radius = 2 * r + 1;
    int accumulator = 0;
    
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
    P[outRow*width+outCol] = (unsigned char)accumulator;
}

void convolution (
    unsigned char* inputMatrix,
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
        (numRows + blockWidth - 1) / blockWidth,
        (numCols + blockWidth - 1) / blockWidth
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

    checkCudaErrors(
        cudaGetLastError()
    );
}

void init_input_matrix(int *m, int n) {
    vector<int> test_matrix = {
        1,2,3,4,5,6,7,
        2,3,4,5,6,7,8,
        3,4,5,6,7,8,9,
        4,5,6,7,8,5,6,
        5,6,7,8,5,6,7,
        6,7,8,9,0,1,2,
        7,8,9,0,1,2,3
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[n * i + j] = test_matrix[n * i + j];
        }
    }
}

void init_result_matrix(int *m, int n) {
    vector<int> test_matrix = {
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[n * i + j] = test_matrix[n * i + j];
        }
    }
}

void init_filter_matrix(float *m, int n) {
    vector<float> filter = {
        0.11,0.11,0.11,
        0.11,0.11,0.11,
        0.11,0.11,0.11
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[n * i + j] = filter[n * i + j];
        }
    }
}

void print_matrix(int *m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << m[n * i + j] << "\t";
        }
        cout << endl;
    }

    cout << endl;
}

int main() {

    // Read the input image as a gray-scale image
    Mat inputImage = imread("3graygoats.jpg", IMREAD_GRAYSCALE);
    int imageTotalSize = inputImage.total();

    // Convert the input image to a vector
    vector<unsigned char> h_input(inputImage.data, inputImage.data + imageTotalSize);
    vector<unsigned char> h_result(inputImage.data, inputImage.data + imageTotalSize);

    // Allocate the matrix and initialize it
    float *h_filter = new float[FILTER_SIZE * FILTER_SIZE];
    init_filter_matrix(h_filter, FILTER_SIZE);

    // Allocate memory space on the device
    unsigned char* d_matrix = nullptr;
    float* d_filter = nullptr;
    unsigned char* d_result = nullptr;
    size_t matrixByteSize = imageTotalSize * sizeof(unsigned char);
    size_t filterByteSize = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_matrix, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_filter, filterByteSize));
    checkCudaErrors(cudaMalloc(&d_result, matrixByteSize));
    
    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_matrix, h_input.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_filter, h_filter, filterByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_result, h_result.data(), matrixByteSize, cudaMemcpyHostToDevice)
    );

    convolution (
        d_matrix,
        d_filter,
        d_result,
        FILTER_RADIUS,
        inputImage.rows,
        inputImage.cols
    );

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (h_result.data(), d_result, matrixByteSize, cudaMemcpyDeviceToHost)
    );

    Mat outputImage;
    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);
    copy(h_result.begin(), h_result.end(), outputImage.data);

    // Display the original and reconstructed images
    imshow("Original", inputImage);
    imshow("Reconstructed", outputImage);
    waitKey(0);

    // Free memory
    cudaFree(d_filter);
    cudaFree(d_matrix);
    cudaFree(d_result);

    printf("Made it to the end!\n");

    return 0;
}
