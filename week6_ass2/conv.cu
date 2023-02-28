#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace cv;
using namespace std;

#define FILTER_RADIUS 1
#define FILTER_SIZE 3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
#define TILE_DIM 32
__constant__ float FILTER_c[FILTER_SIZE * FILTER_SIZE];   // Allocate mask in constant memory

// Question #1 on homework
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

// Question #2 on homework
__global__
void conv_2D_shared_mem_kernel (
    unsigned char* N, float* F, unsigned char* P, int r, int numCols, int numRows
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ float F_s[FILTER_SIZE][FILTER_SIZE];
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
            F_s[fRow][fCol] = F[fRow*FILTER_SIZE + fCol];
        }
        __syncthreads();
    }
    
    float accumulator = 0.0;
    
    // Iterate through the filter rows
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        // Iterate through the filter columns
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
            int inputRow = outRow - r + fRow;
            int inputCol = outCol - r + fCol;
            // If we're within the height of the input matrix
            if (inputRow >= 0 && inputRow < numRows) {
                // If we're within the width of the input matrix
                if (inputCol >= 0 && inputCol < numCols) {
                    accumulator += F_s[fRow][fCol] * N[inputRow*numCols + inputCol];
                }
            }
        }
    }
    P[outRow*numCols + outCol] = (unsigned char)(accumulator);
}

// Question #3 on homework
__global__
void conv_2D_const_mem_kernel (
    unsigned char* inputMatrix,
    unsigned char* outputMatrix,
    int r,
    int numCols,
    int numRows
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    float accumulator = 0.0;
    
    for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
        for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
            int inputRow = outRow - r + fRow;
            int inputCol = outCol - r + fCol;
            if (inputRow >= 0 && inputRow < numRows) {
                if (inputCol >= 0 && inputCol < numCols) {
                    accumulator += FILTER_c[fRow*FILTER_SIZE + fCol] * inputMatrix[inputRow*numCols + inputCol];
                }
            }
        }
    }
    outputMatrix[outRow*numCols + outCol] = (unsigned char)(accumulator);
}

// Question #5 on homework
__global__
void conv_tiled_2D_const_mem_kernel (
    unsigned char* N,
    unsigned char* P,
    int numCols,
    int numRows
) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    
    // loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row >= 0 && row < numRows && col >= 0 && col < numCols) {
        N_s[threadIdx.y][threadIdx.x] = N[row*numCols + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    
    // turning off the threads at the edges of the block
    if (col >= 0 && col < numCols && row >= 0 && row < numRows) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >=0
        && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++){
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++){
                    Pvalue += FILTER_c[fRow*FILTER_SIZE+fCol] * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*numCols + col] = Pvalue;
        }
    }
}

// Question #6 on homework
__global__
void conv_cached_tiled_2D_const_mem_kernel (
    float* N,
    float *P,
    int numCols,
    int numRows
){
    int col = blockIdx.x*TILE_DIM+threadIdx.x;
    int row = blockIdx.y*TILE_DIM+threadIdx.y;
    
    // loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if(row < numRows && col < numCols) {
        N_s[threadIdx.y][threadIdx.x] = N[row*numCols + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    // Calculating output elements
    // turning off the threads at the edges of the block
    if (col < numCols && row < numRows) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (threadIdx.x - FILTER_RADIUS + fCol > 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow > 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM
                ) {
                    Pvalue += FILTER_c[fRow*FILTER_SIZE+fCol]*N_s[threadIdx.y+fRow][threadIdx.x+fCol];
                } else {
                    if (row-FILTER_RADIUS+fRow >=0 &&
                        row-FILTER_RADIUS+fRow < numRows &&
                        col-FILTER_RADIUS+fCol >=0 &&
                        col-FILTER_RADIUS+fCol < numCols
                    ) {
                        Pvalue += FILTER_c[fRow*FILTER_SIZE+fCol]*N[(row-FILTER_RADIUS+fRow)*numCols+col-FILTER_RADIUS+fCol];
                    }
                }
            }
        }
        P[row*numCols+col] = Pvalue;
    }
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
        (numCols + blockWidth - 1) / blockWidth,
        (numRows + blockWidth - 1) / blockWidth
    );

    // Perform CUDA computations on deviceMatrix, Launch Kernel
    // conv_2D_shared_mem_kernel<<<
    //     gridSize,
    //     blockSize
    // >>> (
    //     inputMatrix,
    //     filterMatrix,
    //     outputMatrix,
    //     radius,
    //     numCols,
    //     numRows
    // );
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
    conv_tiled_2D_const_mem_kernel<<<
        gridSize,
        blockSize
    >>> (
        inputMatrix,
        outputMatrix,
        numCols,
        numRows
    );

    checkCudaErrors(
        cudaGetLastError()
    );
}

void init_filter_matrix(float *m, int n) {
    // Guassian Blur filter
    vector<float> filter = {
        0.0625, 0.125, 0.0625,
        0.125, 0.25, 0.125,
        0.0625, 0.125, 0.0625,
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[n * i + j] = filter[n * i + j];
        }
    }
}

int main() {

    // Read the input image as a gray-scale image
    Mat inputImage = imread("3graygoats.jpg", IMREAD_GRAYSCALE);
    int imageTotalSize = inputImage.total();

    // Convert the input image to a vector
    vector<unsigned char> h_input(inputImage.data, inputImage.data + imageTotalSize);
    vector<unsigned char> h_result;
    h_result.resize(imageTotalSize);

    // Allocate the filter matrix and initialize it
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
    checkCudaErrors(
        cudaMemcpyToSymbol(FILTER_c, h_filter, filterByteSize)
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

    // Copy result from gray matrix into matlab OpenCV input array format
    Mat outputImage;
    outputImage.create(inputImage.rows, inputImage.cols, CV_8UC1);
    copy(h_result.begin(), h_result.end(), outputImage.data);

    // Write img to gray.jpg
    //imwrite("grayboiz.jpg", opencv_output);
    imshow("Original", inputImage);
    imshow("Reconstructed", outputImage);
    waitKey(0);

    // Free memory
    cudaFree(d_matrix);
    cudaFree(d_filter);
    cudaFree(d_result);

    printf("Made it to the end!\n");

    return 0;
}
