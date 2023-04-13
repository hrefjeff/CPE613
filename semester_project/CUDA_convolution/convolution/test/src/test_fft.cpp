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

bool read_file_into_array(string, float*);
void multiply_arrays_elementwise(const float*, const float*, float*, int);

int main() {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = complex<scalar_type>;

    int TOTAL_SIZE = N + K - 1;

    // Allocate space on host and device
    input_type *h_signal = new input_type[N];
    input_type *h_filter = new input_type[K];
    output_type *h_output = new output_type[TOTAL_SIZE];

    input_type* d_signal = nullptr;
    input_type* d_filter = nullptr;
    cufftComplex* d_output = nullptr;
    cudaMalloc((void **)&d_signal, N * sizeof(input_type));
    cudaMalloc((void **)&d_filter, K * sizeof(input_type));
    cudaMalloc((void **)&d_output, TOTAL_SIZE * sizeof(cufftComplex));

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_1024.txt";

    bool file_status = false;
    file_status = read_file_into_array(signal_file_name, h_signal);
    if (file_status == false) return 1;
    file_status = read_file_into_array(filter_file_name, h_filter);
    if (file_status == false) return 1;

    cudaMemcpy(d_signal, h_signal, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, K * sizeof(float), cudaMemcpyHostToDevice);

    cufftCreate(&plan);
    cufftPlan1d(&plan, N, CUFFT_R2C, BATCH_SIZE);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan, stream);

    // Step 2. Calculate the fast Fourier transforms of the time-domain
    //         signal and filter
    cufftExecR2C(plan, (cufftReal*)d_signal, d_output);
    cudaMemcpy(h_signal, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cufftExecR2C(plan, (cufftReal*)d_filter, d_output);
    cudaMemcpy(h_filter, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 3. Perform circular convolution in the frequency domain

    // Perform convolution (multiply 2 matricies together)
    // First do it on the host, then optimize to perform on host
    multiply_arrays_elementwise(h_signal, h_filter, h_signal, K);

    // Step 4. Go back to time domain

    // Destroy the cuFFT plan and create a new one for the inverse FFT
    cufftDestroy(plan);
    cufftPlan1d(&plan, TOTAL_SIZE, CUFFT_C2R, 1);
    cudaMemcpy(d_signal, h_signal, N * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the inverse FFT on the result
    cufftExecC2R(plan, d_output, (cufftReal*)d_signal);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the array to an output file
    FILE * fp;
    fp = fopen (output_file_name, "w+");
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        fprintf (fp, "%20.16e\n", h_output[i]);
    }
    fclose(fp);

    delete[] h_signal;
    delete[] h_filter;
    cudaFree(d_signal);
    cudaFree(d_filter);
    cudaFree(d_output);
    return 0;
}

bool read_file_into_array(string filename, float* arr) {
    ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            arr[index++] = (float)(value);
        }
        the_file.close();
    } else {
        cerr << "Unable to open signal file." << endl;
        return false;
    }
    return true;
}

void multiply_arrays_elementwise(const float* array1,
                                 const float* array2,
                                 float* result,
                                 int length
                                ) {
    for (int i = 0; i < length; ++i) {
        result[i] = array1[i] * array2[i];
    }
}