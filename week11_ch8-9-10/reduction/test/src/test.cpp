#include <reduction.h>
#include <Timer.hpp>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

int main(int argc, char ** argv) {

    // Create both host thrust and STL vector
    thrust::host_vector<float> h_thrust_vec(100);

    // Fill host with 1's
    thrust::fill(h_thrust_vec.begin(), h_thrust_vec.end(), 1.0);

    float result = hostReduction(h_thrust_vec);
    //float result = reduction(h_thrust_vec);

    cout << result << endl;

    return 0;
}