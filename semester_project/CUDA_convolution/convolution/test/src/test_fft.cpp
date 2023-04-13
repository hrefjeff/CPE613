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

typedef float2 Complex;
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);

#define N 1024
#define K 1024
#define BATCH_SIZE 1

using namespace std;

bool read_file_into_array(string, float*);
void multiply_arrays_elementwise(const cufftComplex*, const cufftComplex*, cufftComplex*, int);

int main() {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = complex<scalar_type>;

    int TOTAL_SIZE = N + K - 1;

    // Allocate space on host and device
    float *h_signal = new float[N];
    cufftComplex* hc_signal = new cufftComplex[N];

    float *h_filter = new float[K];
    cufftComplex* hc_filter = new cufftComplex[K];
    
    float *h_output = new float[TOTAL_SIZE];
    cufftComplex *hc_output = new cufftComplex[TOTAL_SIZE];

    float* d_signal = nullptr;
    cufftComplex* dc_signal = nullptr;

    float* d_filter = nullptr;
    cufftComplex* dc_filter = nullptr;

    float* dr_output = nullptr;
    cufftComplex* d_output = nullptr;
    cudaMalloc((void **)&d_signal, N * sizeof(float));
    cudaMalloc((void **)&dc_signal, N * sizeof(cufftComplex));
    
    cudaMalloc((void **)&d_filter, K * sizeof(float));
    cudaMalloc((void **)&dc_filter, K * sizeof(cufftComplex));

    cudaMalloc((void **)&dr_output, TOTAL_SIZE * sizeof(cufftComplex));
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
    cudaMemcpyAsync(hc_signal, d_output, sizeof(cufftComplex) * K,
                                 cudaMemcpyDeviceToHost, stream);

    std::printf("Signal array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_signal[i].x, hc_signal[i].y);
    }
    std::printf("=====\n");

    cudaStreamSynchronize(stream); // force CPU thread to wait

    cufftExecR2C(plan, (cufftReal*)d_filter, d_output);
    cudaMemcpyAsync(hc_filter, d_output, sizeof(cufftComplex) * K,
                                 cudaMemcpyDeviceToHost, stream);

    std::printf("Filter array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_filter[i].x, hc_filter[i].y);
    }
    std::printf("=====\n");

    cudaStreamSynchronize(stream); // force CPU thread to wait
    
    // Step 3. Perform circular convolution in the frequency domain

    // Perform convolution (multiply 2 matricies together)
    // First do it on the host, then optimize to perform on host
    multiply_arrays_elementwise(hc_signal, hc_filter, hc_output, K);

    std::printf("Host complex output array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_output[i].x, hc_output[i].y);
    }
    std::printf("=====\n");

    // Step 4. Go back to time domain

    // Destroy the cuFFT plan and create a new one for the inverse FFT
    cufftDestroy(plan);
    cufftPlan1d(&plan, TOTAL_SIZE, CUFFT_C2R, BATCH_SIZE);
    cudaMemcpyAsync(d_output, hc_output, sizeof(cufftComplex) * K,
                                 cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    // Execute the inverse FFT on the result
    cufftExecC2R(plan, d_output, (cufftReal*)dr_output);

    // Copy the result back to the host
    cudaMemcpyAsync(h_output, dr_output, sizeof(cufftComplex) * N,
                                 cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    std::printf("Host real output array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f\n", h_output[i]);
    }
    std::printf("=====\n");

    // Save the array to an output file
    FILE * fp;
    fp = fopen (output_file_name, "w+");
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        fprintf (fp, "%20.16e\n", h_output[i]);
    }
    fclose(fp);

    delete[] h_signal;
    delete[] h_filter;
    delete[] h_output;
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

void multiply_arrays_elementwise(const cufftComplex* array1,
                                 const cufftComplex* array2,
                                 cufftComplex* result,
                                 int length
                                ) {
    for (int i = 0; i < length; ++i) {
        result[i] = ComplexMul(array1[i], array2[i]);
    }
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
