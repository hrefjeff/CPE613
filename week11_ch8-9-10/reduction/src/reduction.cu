#include <reduction.h>

float hostReduction (thrust::host_vector<float> input){
    float sum = 0;
    for(int i = 0; i < input.size(); ++i)
        sum += input[i];
    return sum;
}

float reduction(thrust::device_vector<float> input){
    return 0.0;
}