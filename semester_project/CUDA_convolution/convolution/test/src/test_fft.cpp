/* Include C++ stuff */
#include <complex>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

/* Include my stuff */
#include <convolution.h>
#include <Timer.hpp>

/* Include CUDA stuff */
#include <cuda_runtime.h>
#include <cufftXt.h>

#define N 1024
#define K 1024
#define BATCH_SIZE 1

using namespace std;

int main() {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    int TOTAL_SIZE = N + K - 1;

    // Allocate space on host and device
    input_type *h_input = new input_type[N];
    input_type *h_filter = new input_type[K];
    output_type *h_output = new output_type[TOTAL_SIZE];

    input_type *d_input, *d_filter;
    cufftComplex *d_output = nullptr;
    cudaMalloc((void **)&d_input, N * sizeof(input_type));
    cudaMalloc((void **)&d_filter, K * sizeof(input_type));
    cudaMalloc((void **)&d_output, TOTAL_SIZE * sizeof(cufftComplex));

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_1024.txt";

    ifstream signal_file(signal_file_name);
    ifstream filter_file(filter_file_name);

    // Read signal/filter info into host array
    if (signal_file.is_open()) {
        int index = 0;
        float value;
        while (signal_file >> value) {
            h_input[index++] = static_cast<input_type>(value);
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
            h_filter[index++] = static_cast<input_type>(value);
        }
        filter_file.close();
    } else {
        cerr << "Unable to open filter file." << endl;
        return 1;
    }

    cufftCreate(&plan);
    cufftPlan1d(&plan, N, CUFFT_R2C, BATCH_SIZE);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan, stream);

    cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * N);
    cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * K);
    cudaMemcpyAsync(d_input, h_input, sizeof(input_type) * N,
                                 cudaMemcpyHostToDevice, stream);

    // Step 2. Calculate the fast Fourier transforms 
    // of the time-domain signals
    cufftExecR2C(plan, d_input, d_output);
    cufftExecR2C(plan, d_input, d_output);

    // Step 3. Perform circular convolution in the frequency domain

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, K * sizeof(float), cudaMemcpyHostToDevice);

    // Perform convolution

    // Copy data from device to host

    // Step 4. Go back to time domain

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
