#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>

void convolve_1d(float*, float*, float*, int, int);

#endif