#ifndef MATRIXMULT_CUH_
#define MATRIXMULT_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>

void matrixMultiplication(float *A, float *B, float *C, int N);

#endif