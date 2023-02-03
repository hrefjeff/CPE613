#include <matrixmult.h>
/* #include <Timer.hpp> */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

using namespace std;

int main (int argc, char ** argv) {
  
    // set a size for our vectors
    int N = 4;
    int VEC_SIZE = N*N;

    // allocate vectors x and y_reference
    vector<float> matrix1 (
    VEC_SIZE,
    0.0f
    );
    vector<float> matrix2 (
    VEC_SIZE,
    0.0f
    );
    vector<float> matrix3 (
    VEC_SIZE,
    0.0f
    );
    
    // Provide arbitrary time value for random seed
    srand((unsigned) time(NULL));

    // initialize the matrix1 and matrix2 to some arbitrary values
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            matrix1[i*N+j] = rand() % 10;
            matrix2[i*N+j] = rand() % 10;
            matrix3[i*N+j] = 0;
        }
    }

    // Print A
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            cout << matrix1[i*N+j] << " ";
        }
        cout << endl;
    }

    // Print B
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            cout << matrix2[i*N+j] << " ";
        }
        cout << endl;
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
    matrixMultiplication(dev_A, dev_B, dev_C, VEC_SIZE);

    checkCudaErrors (
        cudaMemcpy (
        matrix3.data(),
        dev_C,
        byteSize_C,
        cudaMemcpyDeviceToHost
        )
    );

    // Print C
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            cout << matrix3[i*N+j] << " ";
        }
        cout << endl;
    }

    // Now do the matrix multiplication on the CPU

    vector<float> matrixCheck(VEC_SIZE, 0.0f);

    float sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0;
            for (int n=0; n<VEC_SIZE; n++){
                sum += matrix1[row*N+n]*matrix2[n*N+col];
            }
            matrixCheck[row*N+col] = sum;
        }
    }

    // Print Check
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            cout << matrixCheck[i*N+j] << " ";
        }
        cout << endl;
    }

    // bool err = false;
    // for (int ROW=0; ROW < N; ROW++){
    //     for (int COL=0; COL < N; COL++){
    //         if (matrixCheck[ROW * N + COL] != matrix3[ROW * N + COL]) err = true;
    //     }
    // }

    // if (err) cout << "The two matricies do not match!!!" << endl;
    // else cout << "Woo! The matricies match." << endl;

    return 0;

}