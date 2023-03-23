#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Different methods of initializing input array
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
  // generate random data serially
  thrust::host_vector<int> h_vec(100);

  thrust::sequence(h_vec.begin(), h_vec.end());
  // thrust::fill(h_vec.begin(), h_vec.end(), 1);

  // transfer to device 
  thrust::device_vector<int> d_vec = h_vec;
  
  // comput the sum
  int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

  std::cout << x << std::endl;
  return 0;
}