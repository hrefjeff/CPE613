/*
    Testing for 1D Time Domain Convolution
    To compile: nvcc test.cu -o test.o -g -G
    To debug: cuda-gdb test.o
    Useful debug tools:
        set cuda coalescing off
        break main
        break 28
        run
        continue
        info cuda threads
        print result
*/

/* Include C++ stuff */
#include <iostream>
#include <string.h>
#include <iostream>
#include <complex>

#include <convo_utils.h>
#include <Convolution.h>
#include <Timer.hpp>

#define N 8192
#define K 8192
#define BATCH_SIZE 1

using namespace std;

int main() {

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_8192.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_8192.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_time_8192.txt";

    Convolution conv(N, BATCH_SIZE);
    conv.allocate_float_memory();

    conv.read_file_into_array(signal_file_name, conv.get_signal());
    conv.read_file_into_array(filter_file_name, conv.get_filter());

    // Perform convolution
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
    
    return 0;
}
