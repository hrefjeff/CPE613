#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace std;

/*
    Launch Params: 3 blocks per grid, 4 threads per block
    For a total of 12 threads
    |      0        |       1       |       2       |
    | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 | 0 | 1 | 2 | 3 |
*/

/**
 * blockDim is the width of the block
 * another way of saying it is
 * the number threads in the block
 *
 * @param arr integer array
 * @return nothing
 */
__global__ void blockDim_kernel (int* arr) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    arr[i] = blockDim.x;
}

/**
 * threadIdx is the index of the thread in a block
 *
 * @param arr integer array
 * @return nothing
 */
__global__ void threadIdx_kernel (int* arr) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    arr[i] = threadIdx.x;
}

/**
 * blockIdx is the index of the block in a grid
 *
 * @param arr integer array
 * @return nothing
 */
__global__ void blockIdx_kernel (int* arr) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    arr[i] = blockIdx.x;
}

/**
 * blockIdx * blockDim to calculate the offset of the block
 * blockIdx will be 0, 1, and 2 because of the launch params
 * 3 blocks in the grid, 4 threads per block
 *
 * @param arr integer array
 * @return nothing
 */
__global__ void correct_kernel (int* arr) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    arr[i] = i;
}

int main() {

    int N = 12;

    // allocate host memory
    vector<int> h_arr1(N,0);
    vector<int> h_arr2(N,0);
    vector<int> h_arr3(N,0);
    vector<int> h_arr4(N,0);

    // allocate device memory
    int* d_arr1 = nullptr;
    int* d_arr2 = nullptr;
    int* d_arr3 = nullptr;
    int* d_arr4 = nullptr;
    int byteSize = N * sizeof(int);
    cudaMalloc(&d_arr1, byteSize);
    cudaMalloc(&d_arr2, byteSize);
    cudaMalloc(&d_arr3, byteSize);
    cudaMalloc(&d_arr4, byteSize);

    // Set our block size and threads per thread block
    int blockDimension = 4;
    int gridDimension = 3;

    blockDim_kernel<<<gridDimension, blockDimension>>>(d_arr1);
    threadIdx_kernel<<<gridDimension, blockDimension>>>(d_arr2);
    blockIdx_kernel<<<gridDimension, blockDimension>>>(d_arr3);
    correct_kernel<<<gridDimension, blockDimension>>>(d_arr4);

    checkCudaErrors (
        cudaMemcpy (
            h_arr1.data(),
            d_arr1,
            byteSize,
            cudaMemcpyDeviceToHost
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            h_arr2.data(),
            d_arr2,
            byteSize,
            cudaMemcpyDeviceToHost
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            h_arr3.data(),
            d_arr3,
            byteSize,
            cudaMemcpyDeviceToHost
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            h_arr4.data(),
            d_arr4,
            byteSize,
            cudaMemcpyDeviceToHost
        )
    );

    // Print arr1 2 3 and 4
    cout << "blockDim:\t ";
    for (int j=0; j<N; j++){
        cout << h_arr1[j] << " ";
    }
    cout << endl;
    cout << "threadIdx:\t ";
    for (int j=0; j<N; j++){
        cout << h_arr2[j] << " ";
    }
    cout << endl;
    cout << "blockIdx:\t ";
    for (int j=0; j<N; j++){
        cout << h_arr3[j] << " ";
    }
    cout << endl;
    cout << "All together:\t ";
    for (int j=0; j<N; j++){
        cout << h_arr4[j] << " ";
    }
    cout << endl;
}
