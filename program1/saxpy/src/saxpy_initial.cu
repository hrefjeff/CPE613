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
    int idx = 0;
    idx < n;
    ++idx
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

  int blockSize = 1;
  int gridSize = 1;

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

