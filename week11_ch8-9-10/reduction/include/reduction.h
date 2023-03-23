#ifndef REDUCTION_CUH_
#define REDUCTION_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>

void reduction(float *A, float *B, float *C, int N);

#endif