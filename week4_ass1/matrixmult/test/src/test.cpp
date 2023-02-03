#include <matrixmult.h>
/* #include <Timer.hpp> */

#include <cmath>
#include <cstdio>
#include <vector>

int main (int argc, char ** argv) {
  
    // set a size for our vectors
    int VEC_SIZE = 100;

    // allocate vectors x and y_reference
    std::vector<float> matrix1 (
    VEC_SIZE,
    0.0f
    );
    std::vector<float> matrix2 (
    VEC_SIZE,
    0.0f
    );
    std::vector<float> matrix3 (
    VEC_SIZE,
    0.0f
    );
    std::vector<float> matrixCheck (
    VEC_SIZE,
    0.0f
    );
    
    // initialize the vectors x and y to some arbitrary values
    for (int idx = 0; idx < VEC_SIZE; ++idx) {
    matrix1[idx] = rand() % 1000;
    matrix2[idx] = rand() % 1000;
    }

    // allocate device memory
    float * dev_A = nullptr;
    float * dev_B = nullptr;
    float * dev_C = nullptr;
    size_t byteSize_A = matrix1.size() * sizeof(float);
    size_t byteSize_B = matrix2.size() * sizeof(float);
    size_t byteSize_C = matrix3.size() * sizeof(float);
    checkCudaErrors(cudaMalloc(&dev_A, byteSize_A));
    checkCudaErrors(cudaMalloc(&dev_B, byteSize_B));
    checkCudaErrors(cudaMalloc(&dev_C, byteSize_C));
  
  
    // copy input to device
    checkCudaErrors (
        cudaMemcpy (
            dev_A,
            matrix1.data(),
            byteSize_A,
            cudaMemcpyHostToDevice
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            dev_B,
            matrix2.data(),
            byteSize_B,
            cudaMemcpyHostToDevice
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            dev_C,
            matrix3.data(),
            byteSize_C,
            cudaMemcpyHostToDevice
        )
    );

    // execute our matrix multiplication
    matrixMultiplication(matrix1, matrix2, matrix3, VEC_SIZE);

    checkCudaErrors (
        cudaMemcpy (
        matrix3.data(),
        dev_C,
        byteSize_C,
        cudaMemcpyDeviceToHost
        )
    );

    // Now do the matrix multiplication on the CPU
    float sum;
    for (int row=0; row<VEC_SIZE; row++){
        for (int col=0; col<VEC_SIZE; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += matrix1[row*N+n]*matrix2[n*N+col];
            }
            matrixCheck[row*N+col] = sum;
        }
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < VEC_SIZE; ROW++){
        for (int COL=0; COL < VEC_SIZE; COL++){
            err += matrixCheck[ROW * N + COL] - matrix3[ROW * N + COL];
        }
    }

    std::cout << "Error: " << err << std::endl;

    return 0;

}