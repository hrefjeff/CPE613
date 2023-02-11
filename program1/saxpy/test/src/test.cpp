#include <saxpy.h>
#include <Timer.hpp>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace std;

// let the compiler know we want to use C style naming and
// calling conventions for the Fortran function
//
// for more details
//    - https://docs.oracle.com/cd/E19422-01/819-3685/11_cfort.html
extern "C" void saxpy_ (
  int   *   n,
  float *   alpha,
  float *   dev_x,
  int   *   incx,
  float *   dev_y,
  int   *   incy
);

double relative_error_l2 (
  int n,
  float  * y_reference,
  int inc_y_reference,
  float * y_computed,
  int inc_y_computed
);

int main (int argc, char ** argv) {

  // set values for the offset for x and y
  int incx = 1;
  int incy = 1;
  
  // set a size for our vectors
  int VEC_SIZE = 10000000;

  // allocate vectors x and y_reference
  vector<float> x (
    VEC_SIZE * incx,
    0.0f
  );
  vector<float> y_reference (
    VEC_SIZE * incy,
    0.0f
  );

  // Provide arbitrary time value for random seed
  srand((unsigned) time(NULL));

  // initialize the vectors x and y to some arbitrary values
  for (int idx = 0; idx < VEC_SIZE; ++idx) {
    x[idx * incx] = rand() % 1000;
    y_reference[idx * incy] = rand() % 1000;
  }

  // set values for the scalar
  // ideally should be random for our test
  float alpha = 1.0f; // the suffix f denotes float as opposed to double

  // allocate device memory
  float * dev_x = nullptr;
  float * dev_y_computed = nullptr;
  size_t byteSize_x = x.size() * sizeof(float);
  size_t byteSize_y_reference = y_reference.size() * sizeof(float);
  checkCudaErrors (
    cudaMalloc (
      &dev_x,
      byteSize_x
    )
  );
  checkCudaErrors (
    cudaMalloc (
      &dev_y_computed,
      byteSize_y_reference
    )
  );

  // copy input to dev_x, dev_y_computed
  checkCudaErrors (
    cudaMemcpy (
      dev_x,
      x.data(),
      byteSize_x,
      cudaMemcpyHostToDevice
    )
  );
  checkCudaErrors (
    cudaMemcpy (
      dev_y_computed,
      y_reference.data(),
      byteSize_y_reference,
      cudaMemcpyHostToDevice
    )
  );

  int numOfRuns = 2;

  // call the Fortran version of the saxpy
  //  - note that we have to pass addresses of the nonpointer arguments
  //  - note that we pre-declared the name the Fortran compiler produced above
  //    (add a trailing underscore to the function name)
  for (int i = 0; i < numOfRuns; i++) {
    saxpy_ (
      &VEC_SIZE,
      &alpha,
      x.data(),
      &incx,
      y_reference.data(),
      &incy  
    );
  }

  double elapsedTime_ms = 0.0f;
  double total_elapsedTime_ms = 0.0f;

  double numberOfFlops = 2 * VEC_SIZE;
  double flopRate = 0.0f;
  double totalFlopRate = 0.0f;
  double numberOfReads = 2 * VEC_SIZE;
  double numberOfWrites = VEC_SIZE;
  
  // Begin saxpy kernel, run it multiple times. print result
  for (int i = 0; i < numOfRuns; i++) {
    // start the timer
    Timer timer;
    timer.start();
    // execute our saxpy
    saxpy (
      VEC_SIZE,
      alpha,
      dev_x,
      incx,
      dev_y_computed,
      incy  
    );
    timer.stop();
    
    // get elapsed time, estimated flops per second, and effective bandwidth
    elapsedTime_ms = timer.elapsedTime_ms();
    total_elapsedTime_ms += elapsedTime_ms;

  }

  double totalNumberOfFlops = numberOfFlops * numOfRuns;
  double totalReads = numberOfReads * numOfRuns;
  double totalWrites = numberOfWrites * numOfRuns;
  double avg_elapsedTime_ms = total_elapsedTime_ms / numOfRuns;

  double avg_flopRate = totalNumberOfFlops / (total_elapsedTime_ms / 1.0e3);
  double avg_effectiveBandwidth_bitspersec =
      (totalReads + totalWrites) * sizeof(float) * 8 / 
      (total_elapsedTime_ms / 1.0e3);
  
  printf (
   "\t- Computational Rate:         %20.16e Gflops\n",
    avg_flopRate / 1e9 
  );
  printf (
   "\t- Effective Bandwidth:        %20.16e Gbps\n",
    avg_effectiveBandwidth_bitspersec / 1e9 
  );

  // output relative error
  // copy result down from device
  vector<float> y_computed (
    y_reference.size(),
    0.0f
  );
  checkCudaErrors (
    cudaMemcpy (
      y_computed.data(),
      dev_y_computed,
      byteSize_y_reference,
      cudaMemcpyDeviceToHost
    )
  );

  double err = relative_error_l2 (
    VEC_SIZE,
    y_reference.data(),
    incy,
    y_computed.data(),
    incy
  );

  printf ("\t- Relative Error (l2):        %20.16e\n", err);
  if (err < 1.0e-7) printf("\t- PASSED\n");
  else printf("\t-FAILED\n");

  return 0;

}

double relative_error_l2 (
  int n,
  float  * y_reference,
  int inc_y_reference,
  float * y_computed,
  int inc_y_computed
) {

  double difference_norm_squared = 0.0;
  double reference_norm_squared = 0.0;
  for (int idx = 0; idx < n; ++idx) {
    auto & reference_value = y_reference[idx * inc_y_reference];
    double difference {
      y_reference[idx * inc_y_reference] - 
      y_computed[idx * inc_y_computed]
    };
    difference_norm_squared += difference * difference;
    reference_norm_squared += reference_value * reference_value;
  }

  return sqrt (
    difference_norm_squared / reference_norm_squared
  );

}