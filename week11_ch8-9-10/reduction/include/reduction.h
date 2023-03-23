#ifndef REDUCTION_CUH_
#define REDUCTION_CUH_

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

float hostReduction(thrust::host_vector<float>);                    // 0
float sequentialReduction(thrust::device_vector<float>);            // 1
float parallelAtomicsReduction(thrust::device_vector<float>);       // 2
float segmentedReduction(thrust::device_vector<float>);             // 3
float segmentedCoalescingReduction(thrust::device_vector<float>);   // 4
float sharedMemoryReduction(thrust::device_vector<float>);          // 5 
float coarsenedSharedMemoryReduction(thrust::device_vector<float>); // 6

float reduction(thrust::device_vector<float>);

#endif