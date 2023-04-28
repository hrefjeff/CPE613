/* TODO: Implement Callbacks

https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/

*/

/* Include C++ stuff */
#include <complex>
#include <string.h>
#include <cstdio>
#include <cstdlib>

/* Include CUDA stuff */
#include <cuda_runtime.h>
#include <cufftXt.h>

/* Include my stuff */
#include <convo_utils.h>
#include <Convolution.h>
#include <Timer.hpp>

#define N 4096
#define K 4096
#define BATCH_SIZE 1

using namespace std;

int main() {

    Convolution conv(N, BATCH_SIZE);
    conv.allocate_complex_memory();

    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_4096.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_4096.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_4096.txt";

    // Initialize the signal
    vector<cufftComplex> signal;
    signal = conv.get_signal_complex();
    vector<cufftComplex> filter;
    filter = conv.get_filter_complex();
    conv.read_file_into_complex_array(signal_file_name, signal);
    conv.read_file_into_complex_array(filter_file_name, filter);
    

    Timer timer;
    timer.start();
    conv.compute();
    timer.stop();
    
    conv.write_complex_results_to_file(output_file_name);

    double elapsedTime_ms = timer.elapsedTime_ms();

    printf (
    "\n- Elapsed Time:             %20.16e Ms\n\n",
        elapsedTime_ms / 1.0e3
    );

    
    return EXIT_SUCCESS;
}