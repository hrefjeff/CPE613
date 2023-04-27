#include <Convolution.hpp>

#include <vector>

Convolution::Convolution(
    std::vector<float>& signal,
    std::vector<float>& filter,
    int batch_size
) : _signal(signal), _filter(filter), _batch_size(batch_size) {
    if (batch_size <= 500) {
        _method = Method::TimeBased;
    } else {
        _method = Method::FFTBased;
    }
}

std::vector<float> Convolution::compute(){
    if (_method == Method::TimeBased) {
        return time_based_convolution(_signal, _filter);
    } else {
        return fft_based_convolution(_signal, _filter);
    }
}

int Convolution::next_power_of_two(int num) {
    return 1 << (int(log2(num - 1)) + 1);
}

std::vector<float> Convolution::time_based_convolution(
    std::vector<float>& signal,
    std::vector<float>& filter) {
        // Implement FFT-based convolution here
}

std::vector<float> Convolution::fft_based_convolution(
    std::vector<float>& signal,
    std::vector<float>& filter) {
        // Implement FFT-based convolution here
}