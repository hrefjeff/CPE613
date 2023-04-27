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

#include <convolution.h>
#include <Timer.hpp>

#define N 8192
#define K 8192

using namespace std;

int main() {
    float *h_input = new float[N];
    float *h_filter = new float[K];
    float *h_output = new float[N + K - 1];
    float *d_input, *d_filter, *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_input, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_filter, K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_output, (N+K-1) * sizeof(float)));

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_8192.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_8192.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_time_8192.txt";

    ifstream signal_file(signal_file_name);
    ifstream filter_file(filter_file_name);

    if (signal_file.is_open()) {
        int index = 0;
        float value;
        while (signal_file >> value) {
            h_input[index++] = (float)(value);
        }
        signal_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
        return EXIT_FAILURE;
    }

    if (filter_file.is_open()) {
        int index = 0;
        float value;
        while (filter_file >> value) {
            h_filter[index++] = (float)(value);
        }
        filter_file.close();
    } else {
        std::cout << "Unable to open filter file." << std::endl;
        return EXIT_FAILURE;
    }

    
    checkCudaErrors(
        cudaMemcpy(
            d_input, h_input,
            N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );
    checkCudaErrors(
        cudaMemcpy(
            d_filter, h_filter,
            K * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    Timer timer;
    timer.start();
    convolve_1d_time(d_input, d_filter, d_output, N, K);
    timer.stop();
    
    checkCudaErrors(
        cudaMemcpy(
            h_output, d_output,
            (N + K - 1) * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    double elapsedTime_ms = timer.elapsedTime_ms();

    printf (
    "\n- Elapsed Time:             %20.16e Ms\n\n",
        elapsedTime_ms / 1.0e3
    );

    FILE* filePtr = fopen(output_file_name, "w");
    for (int i = 0; i < N + K - 1; i++) {
        fprintf (filePtr, "%20.16e\n", h_output[i]);
    }
    fclose(filePtr);

    delete[] h_input;
    delete[] h_filter;
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    return 0;
}
