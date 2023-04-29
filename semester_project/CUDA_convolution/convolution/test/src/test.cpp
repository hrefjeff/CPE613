/* Include my stuff */
#include <Convolution.h>
#include <Timer.hpp>

/* Include C++ stuff */
#include <cmath>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

#define N 1024
#define K 1024

using namespace std;

int main() {

    int TOTAL_SIZE = N + K - 1;

    // Allocate space on host and device
    float *h_input = new float[N];
    float *h_filter = new float[K];
    float *h_output = new float[TOTAL_SIZE];
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_filter, K * sizeof(float));
    cudaMalloc((void **)&d_output, TOTAL_SIZE * sizeof(float));

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_time_1024.txt";

    ifstream signal_file(signal_file_name);
    ifstream filter_file(filter_file_name);

    // Read signal/filter info into host array
    if (signal_file.is_open()) {
        int index = 0;
        float value;
        while (signal_file >> value) {
            h_input[index++] = value;
        }
        signal_file.close();
    } else {
        cerr << "Unable to open signal file." << endl;
        return 1;
    }

    if (filter_file.is_open()) {
        int index = 0;
        float value;
        while (filter_file >> value) {
            h_filter[index++] = value;
        }
        filter_file.close();
    } else {
        cerr << "Unable to open filter file." << endl;
        return 1;
    }

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, K * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution
    convolve_1d_time(d_input, d_filter, d_output, N, K);

    // Copy data from device to host
    cudaMemcpy(
        h_output,
        d_output,
        TOTAL_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Save the array to an output file
    FILE * fp;
    fp = fopen (output_file_name, "w+");
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        fprintf (fp, "%20.16e\n", h_output[i]);
    }
    fclose(fp);

    delete[] h_input;
    delete[] h_filter;
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    return 0;
}
