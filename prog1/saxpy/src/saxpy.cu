#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void saxpy_kernel (
  int   n,
  float alpha,
  float *dev_x,
  int   incx,
  float *dev_y,
  int   incy
) {
  for (
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    idx < n;
    idx += gridDim.x + blockDim.x
  ) {
    dev_y[idx * incy] = alpha * dev_x[idx * incx] + dev_y[idx * incy];
  }
}

void saxpy (
  int   n,
  float alpha,
  float *dev_x,
  int   incx,
  float *dev_y,
  int   incy
) {
  int blockSize = 512; // number of thread in a block, we'll tune later
  int gridSize; // number of blocks
  
  // Round up according to array size
  gridSize = (n + blockSize - 1) / blockSize;
  
  // call the kernel
  saxpy_kernel <<<gridSize, blockSize>>> (n, alpha, dev_x, incx, dev_y, incy);
  
  checkCudaErrors(cudaGetLastError()); // make sure to check errors
}

double relative_error_12 (
  int   n,
  float *y_reference,
  int   inc_y_reference,
  float *y_computed,
  int   inc_y_computed
) {
  double difference_norm_squared = 0.0;
  double reference_norm_squared = 0.0;
  for (int idx = 0; idx < n; ++idx) {
    auto & reference_value = y_reference[idx * inc_y_reference];
    double difference {
      y_reference[idx * inc_y_reference] -
      y_computed[idx * inc_y_computed]
    };
    difference_norm_squared = difference * difference;
    reference_norm_squared = reference_value * reference_value;
  }

  return sqrt ( difference_norm_squared / reference_norm_squared);
}

int main() {

  // set size and strides
  int n = 5;
  int incx = 1;
  int incy = 1;
  
  // preallocate the memory on the host and device
  std::vector<float> host_x(n * incx, 0.0f);
  std::vector<float> host_y(n * incy, 0.0f);
  float *dev_x = nullptr;
  float *dev_y = nullptr;
  int byte_size_x = n * incx * sizeof(float);
  int byte_size_y = n * incy * sizeof(float);
  checkCudaErrors (cudaMalloc(&dev_x, byte_size_x));
  checkCudaErrors (cudaMalloc(&dev_y, byte_size_y));
  
  // set values of vecx, vecy, and alpha on host, copy to device
  float alpha = 1.0f;
  for (int idx = 0; idx < n; ++idx) {
    host_x[idx * incx] = idx;
    host_y[idx * incy] = n - idx;
  }
  
  // there are certainly better ways to do this
  checkCudaErrors (
    cudaMemcpy (dev_x, host_x.data(), byte_size_x, cudaMemcpyHostToDevice)
  );
  
  checkCudaErrors (
    cudaMemcpy (dev_y, host_y.data(), byte_size_y, cudaMemcpyHostToDevice)
  );
  
  // call our saxpy
  saxpy (n, alpha, dev_x, incx, dev_y, incy);
  
  // copy result down
  checkCudaErrors (
    cudaMemcpy (host_y.data(), dev_y, byte_size_y, cudaMemcpyDeviceToHost)
  );
  
  checkCudaErrors (cudaFree(dev_x));
  checkCudaErrors (cudaFree(dev_y));
  
  // print result
  for (int idx = 0; idx < n; ++idx) {
    printf("y[%d] = %20.16f\n", idx, host_y[idx*incy]);
  }

  /* ~~~~Comparison testing~~~~ */
  // copy result down from device 
  std::vector<float> y_computed (
    y_reference.size(),
    0.0f
  );

  checkCudaErrors (
    cudaMemcpy (

    )
  );

  // output relative error
  printf (
    "\t- Relative Error (12):         %20.16e\n",
    relerr
  );

  if (relerr < 1.0e-7) printf("\t- PASSED\n");
  else printf("\t-FAILED\n");
  
  return 0;
}
