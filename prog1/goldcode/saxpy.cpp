#include <cstdio>
#include <vector>

// to expose the Fortran code as a C function to our C++ code
extern "C" void saxpy_ (
  int   * n,
  float * alpha,
  float * x,
  int   * incx,
  float * y,
  int   * incy
);

int main () {

  // set size and strides
  int n = 5;
  int incx = 1;
  int incy = 1;
  
  // preallocate the memory via RAII classes
  std::vector<float> x(n * incx, 0.0f);
  std::vector<float> y(n * incy, 0.0f);
  
  // set values of \vec{x}, \vec{y}, and \alpha
  float alpha = 1.0f;
  for (int idx = 0; idx < n; ++idx) {
    x[idx * incx] = idx;
    y[idx * incy] = n - idx;
  }
  
  // call the saxpy, i.e., y <- alpha * x + y
  saxpy_ (
    &n, // since the Fortran expects an address
    &alpha,
    x.data(), // the pointer to the internal buffer
    &incx,
    y.data(),
    &incy
  );
  
  // print result, should see the size in each field of the result
  // based on how we set the values of vec{x} vec{y} and alpha
  for (int idx = 0; idx < n; ++idx) {
      printf("y[%d] = %20.16f\n", idx, y[idx*incy]);
  }
  
  return 0;
}
