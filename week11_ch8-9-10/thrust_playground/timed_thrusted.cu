#include <chrono>
#include <unistd.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

using namespace std;

int main(void)
{
    // generate random data serially
    thrust::host_vector<float> h_vec(10000000);

    // Fill it with 1.0's
    thrust::fill(h_vec.begin(), h_vec.end(), 1.0);

    // transfer to device 
    thrust::device_vector<int> d_vec = h_vec;

    // compute the sum
    auto start = chrono::steady_clock::now();
    int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<float>());
    auto end = chrono::steady_clock::now();

    cout << "Elapsed time in milliseconds: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" 
    << endl;

    cout << x << endl;
    
    return 0;
}