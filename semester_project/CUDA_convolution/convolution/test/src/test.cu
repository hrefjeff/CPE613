/* TODO: Implement Callbacks

https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/

*/

/* Include C++ stuff */
#include <complex>
#include <string.h>
#include <cstdio>
#include <cstdlib>

/* Include my stuff */
#include <Convolution.h>
#include <Timer.hpp>

#define N 512
#define K 512
#define BATCH_SIZE 1

using namespace std;

int main() {

    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_512.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_512.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/conv_512.txt";

    Convolution conv(N, BATCH_SIZE);

    // Allocate memory for signal and filter
    conv.allocate_memory();

    // Initialize the signal and filter
    conv.read_file_into_signal(signal_file_name);
    conv.read_file_into_filter(filter_file_name);

    // Convolve the signal
    Timer timer;
    timer.start();
    conv.compute();
    timer.stop();
    
    conv.write_results_to_file(output_file_name);

    double elapsedTime_ms = timer.elapsedTime_ms();

    printf (
    "\n- Elapsed Time:             %20.16e Ms\n\n",
        elapsedTime_ms / 1.0e3
    );
    
    return EXIT_SUCCESS;
}