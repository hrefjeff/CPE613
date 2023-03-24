#ifndef REDUCTION_CUH_
#define REDUCTION_CUH_

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

float hostReduction (thrust::host_vector<float>);
void sequentialReduction (float* input, float* output, int N);
void reduction (float* input, float* output, int N);

#endif