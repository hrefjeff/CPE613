#include <reduction.h>
#include <Timer.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

int main(int argc, char ** argv) {

    int numOfElements = 100;

    // Create both host thrust and STL vector
    thrust::host_vector<float> h_thrust_vec(numOfElements);
    vector<float> h_stl_vect(numOfElements);

    // Fill host with 1's
    thrust::fill(h_thrust_vec.begin(), h_thrust_vec.end(), 1.0);

    // copy a device_vector into an STL vector
    thrust::copy(h_thrust_vec.begin(), h_thrust_vec.end(), h_stl_vect.begin());

    float h_output = 0.0;

    // allocate device memory
    float * d_input = nullptr;
    float * d_output = nullptr;
    size_t byteSize_input = numOfElements * sizeof(float);
    size_t byteSize_output = sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, byteSize_input));
    checkCudaErrors(cudaMalloc(&d_output, byteSize_output));
  
    // copy input to device
    checkCudaErrors (
        cudaMemcpy (
            d_input,
            h_stl_vect.data(),
            byteSize_input,
            cudaMemcpyHostToDevice
        )
    );
    checkCudaErrors (
        cudaMemcpy (
            d_output,
            &h_output,
            byteSize_output,
            cudaMemcpyHostToDevice
        )
    );

    //float result = hostReduction(h_thrust_vec);
    //sequentialReduction(d_input, d_output, numOfElements);
    reduction(d_input, d_output, numOfElements);

    checkCudaErrors (
        cudaMemcpy (
            &h_output,
            d_output,
            byteSize_output,
            cudaMemcpyDeviceToHost
        )
    );

    cout << h_output << endl;

    return 0;
}