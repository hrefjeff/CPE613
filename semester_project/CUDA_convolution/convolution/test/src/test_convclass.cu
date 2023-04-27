#include <convolution.h>

int main() {
    std::vector<float> signal = {}; // Your signal data
    std::vector<float> filter = {}; // Your filter data

    Convolution conv(signal, filter);
    std::vector<float> result = conv.compute();

    // Process the result
    // ...

    return 0;
}