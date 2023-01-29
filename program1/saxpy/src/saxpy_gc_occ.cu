#include <saxpy.h>
#include <helper_cuda.h>

__global__ void saxpy_kernel (
  int       n,
  float     alpha,
  float *   dev_x,
  int       incx,
  float *   dev_y,
  int       incy
) {

  for (
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx < n;
    idx += gridDim.x * blockDim.x
  ) {

    dev_y[incy * idx] = dev_y[incy * idx] + alpha * dev_x[incx * idx];

  }

}

void saxpy (
  int       n,
  float     alpha,
  float *   dev_x,
  int       incx,
  float *   dev_y,
  int       incy
) {

  int blockSize;    // The launch configurator returned block size 
  int minGridSize;  // The minimum grid size needed to achieve the 
                    // maximum occupancy for a full device launch 
  int gridSize;     // The actual grid size needed, based on input size 

  checkCudaErrors (
    cudaOccupancyMaxPotentialBlockSize(
      &minGridSize,
      &blockSize, 
      saxpy_kernel,
      0,
      0
    )
  ); 
  
  // Round up according to array size 
  gridSize = (n + blockSize - 1) / blockSize;

  saxpy_kernel<<<gridSize, blockSize>>> (
    n,
    alpha,
    dev_x,
    incx,
    dev_y,
    incy
  ); 
  
  checkCudaErrors(cudaGetLastError());

}

