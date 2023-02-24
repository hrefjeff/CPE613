#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

#define MATRIX_SIZE 7

// 2x2 convolutional mask
#define FILTER_RADIUS 2
// 2*r+1
#define FILTER_SIZE 5
// Allocate mask in constant memory
__constant__ int filter[4 * 4];

__global__
void conv_2D_basic_kernel (
    int* N, int* F, int* P, int r, int width, int height
) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    
    int Pvalue = 0;
    
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inputRow = outRow - r + fRow;
            int inputCol = outCol - r + fCol;
            if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width) {
                Pvalue += F[fRow*width+fCol] * N[inputRow*width + inputCol];
            }
        }
    }
    P[outRow*width+outCol] = Pvalue;
}

void convolution (
    int* inputMatrix,
    int* filterMatrix,
    int* outputMatrix,
    int radius,
    int numRows,
    int numCols
) {
    int blockWidth = 7;

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

    checkCudaErrors(
        cudaGetLastError()
    );
}

void init_test_matrix(int *m, int n) {
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

void init_filter_matrix(int *m, int n) {
    vector<int> test_matrix = {
        1,2,3,2,1,
        2,3,4,3,2,
        3,4,5,4,3,
        2,3,4,3,2,
        1,2,3,2,1
    };
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[n * i + j] = test_matrix[n * i + j];
        }
    }
}

void print_matrix(int *m, int n) {
    if (m[0] != 69) cout << "Matrix is not correct" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << m[n * i + j] << "\t";
        }
        cout << endl;
    }

    cout << endl;
}

int main() {

    // Allocate the matrix and initialize it
    int *h_matrix = new int[MATRIX_SIZE * MATRIX_SIZE];
    int *h_result = new int[MATRIX_SIZE * MATRIX_SIZE];
    init_test_matrix(h_matrix, MATRIX_SIZE);
    init_test_matrix(h_result, MATRIX_SIZE);

    int *h_filter = new int[FILTER_SIZE * FILTER_SIZE];
    init_filter_matrix(h_filter, FILTER_SIZE);

    // Allocate memory space on the device
    int* d_matrix = nullptr;
    int* d_filter = nullptr;
    int* d_result = nullptr;
    size_t matrixByteSize = MATRIX_SIZE * MATRIX_SIZE * sizeof(int);
    size_t filterByteSize = FILTER_SIZE * FILTER_SIZE * sizeof(int);

    checkCudaErrors(cudaMalloc(&d_matrix, matrixByteSize));
    checkCudaErrors(cudaMalloc(&d_filter, filterByteSize));
    checkCudaErrors(cudaMalloc(&d_result, matrixByteSize));
    
    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (d_matrix, h_matrix, matrixByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_filter, h_filter, filterByteSize, cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (d_result, h_result, matrixByteSize, cudaMemcpyHostToDevice)
    );

    convolution (
        d_matrix,
        d_filter,
        d_result,
        FILTER_RADIUS,
        MATRIX_SIZE,
        MATRIX_SIZE
    );

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (h_result, d_result, matrixByteSize, cudaMemcpyDeviceToHost)
    );

    print_matrix(h_result, MATRIX_SIZE);

    // Free memory
    cudaFree(d_filter);
    cudaFree(d_matrix);
    cudaFree(d_result);

    printf("Made it to the end!\n");

    return 0;
}
