#include <matrixmult.h>
#include <Timer.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

using namespace std;

int main (int argc, char ** argv) {
  
    // Set size for matrices
    // 64, 128, 256, 512, 1024, 2048, 4096
    int N = 4096;
    int VEC_SIZE = N*N;

    // allocate vectors x and y_reference
    
    vector<float> matrix1(VEC_SIZE, 1.0);
    // vector<float> matrix1 {
    //     1.0, 2.0, 3.0, 4.0,
    //     5.0, 6.0, 7.0, 8.0, 
    //     9.0, 10.0, 11.0, 12.0,
    //     13.0, 14.0, 15.0, 16.0
    // };
    
    vector<float> matrix2(VEC_SIZE, 1.0);
    // vector<float> matrix2 {
    //     16.0, 15.0, 14.0, 13.0,
    //     12.0, 11.0, 10.0, 9.0,
    //     8.0, 7.0, 6.0, 5.0,
    //     4.0, 3.0, 2.0, 1.0
    // };
    vector<float> matrix3(VEC_SIZE, 1.0);

    // Provide arbitrary time value for random seed
    srand((unsigned) time(NULL));

    // // initialize the matrix1 and matrix2 to some arbitrary values
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            matrix1[i*N+j] = rand() % 1000;
            matrix2[i*N+j] = rand() % 1000;
            matrix3[i*N+j] = 0;
        }
    }

    // Print A
    // for (int i=0; i<N; i++){
    //     for (int j=0; j<N; j++){
    //         cout << matrix1[i*N+j] << " ";
    //     }
    //     cout << endl;
    // }

    // Print B
    // for (int i=0; i<N; i++){
    //     for (int j=0; j<N; j++){
    //         cout << matrix2[i*N+j] << " ";
    //     }
    //     cout << endl;
    // }

    // allocate device memory
    float * dev_A = nullptr;
    float * dev_B = nullptr;
    float * dev_C = nullptr;
    size_t byteSize_A = VEC_SIZE * sizeof(float);
    size_t byteSize_B = VEC_SIZE * sizeof(float);
    size_t byteSize_C = VEC_SIZE * sizeof(float);
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
    int numOfRuns = 100;

    double elapsedTime_ms = 0.0f;
    double total_elapsedTime_ms = 0.0f;
    Timer timer;

    double numberOfFlops = 2 * VEC_SIZE;
    double flopRate = 0.0f;
    double numberOfReads = 2 * VEC_SIZE;
    double numberOfWrites = VEC_SIZE;

    float alpha = 1.0;
    float beta = 0.0;
    
    for (int runCount = 0; runCount < numOfRuns; runCount++) {
        // execute reference matrix multiplication using cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        timer.start();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_B, N, dev_A, N, &beta, dev_C, N);
        //matrixMultiplication(dev_A, dev_B, dev_C, N);
        timer.stop();
        cublasDestroy(handle);

        // get elapsed time, estimated flops per second, and effective bandwidth
        elapsedTime_ms = timer.elapsedTime_ms();
        total_elapsedTime_ms += elapsedTime_ms;
    }

    double totalReads = 2.0 * VEC_SIZE * numOfRuns;
    double totalWrites = VEC_SIZE * numOfRuns;
    double totalNumberOfFlops = 2.0 * N*N*N * numOfRuns;

    double avg_elapsedTime_ms = total_elapsedTime_ms / numOfRuns;
    double avg_flopRate = totalNumberOfFlops / (total_elapsedTime_ms / 1.0e3);

    printf (
    "\n\t- Avg Computational Rate:         %20.16e Gflops\n",
        avg_flopRate / 1e9 
    );
    double avg_effectiveBandwidth_bitspersec =
        (totalReads + totalWrites) * sizeof(float) * 8 / 
        (total_elapsedTime_ms / 1.0e3);
    printf (
    "\t- Avg Effective Bandwidth:        %20.16e Gbps\n",
        avg_effectiveBandwidth_bitspersec / 1e9 
    );

    // checkCudaErrors (
    //     cudaMemcpy (
    //         matrix3.data(),
    //         dev_C,
    //         byteSize_C,
    //         cudaMemcpyDeviceToHost
    //     )
    // );

    // // Print C
    // for (int i=0; i<N; i++){
    //     for (int j=0; j<N; j++){
    //         cout << matrix3[i*N+j] << " ";
    //     }
    //     cout << endl << endl;
    // }

    // Now do the matrix multiplication on the CPU

    // vector<float> matrixCheck(VEC_SIZE);

    // float sum;
    // for (int row=0; row<N; row++){
    //     for (int col=0; col<N; col++){
    //         sum = 0;
    //         for (int i=0; i<N; i++){
    //             sum += matrix1[row*N+i]*matrix2[i*N+col];
    //         }
    //         matrixCheck[row*N+col] = sum;
    //     }
    // }

    // bool err = false;
    // for (int ROW=0; ROW < N; ROW++){
    //     for (int COL=0; COL < N; COL++){
    //         if (matrixCheck[ROW * N + COL] != matrix3[ROW * N + COL]) err = true;
    //     }
    // }

    // if (err) cout << "ERROR: The two matricies do not match!!!" << endl;
    // else cout << "SUCCESS: Woo! The matricies match." << endl;

    return 0;

}