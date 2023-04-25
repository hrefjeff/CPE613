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
#include <convolution.h>
#include <Timer.hpp>

#define N 8192
#define K 8192
#define BATCH_SIZE 1

using namespace std;

int main() {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int FFT_SIZE = next_power_of_2(N + K - 1);
    
    bool file_status = false;
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_8192.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_8192.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_8192.txt";

    // Initialize the signal

    // host signal initialization
    vector<cufftComplex> h_signal(FFT_SIZE, cufftComplex{0});
    
    file_status = read_file_into_vector(signal_file_name, h_signal);
    if (file_status == false) return EXIT_FAILURE;

    // device signal initialization
    cufftComplex *d_signal = nullptr;
    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void **>(&d_signal),
            sizeof(cufftComplex) * h_signal.size()
        )
    );

    // Initialize the filter

    // host filter initialization
    vector<cufftComplex> h_filter(FFT_SIZE, cufftComplex{0});

    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return EXIT_FAILURE;

    // device filter initialization    
    cufftComplex *d_filter = nullptr;
    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void **>(&d_filter),
            sizeof(cufftComplex) * h_filter.size()
        )
    );

    // Initialize the result

    // host result initialization
    vector<cufftComplex> h_convolved_result(FFT_SIZE, cufftComplex{0});
    
    // device result inintialization
    cufftComplex *d_convolved_fft = nullptr;
    cufftComplex *d_product_fft = nullptr;

    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void **>(&d_convolved_fft),
            sizeof(cufftComplex) * h_convolved_result.size()
        )
    );
    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void **>(&d_product_fft),
            sizeof(cufftComplex) * h_convolved_result.size()
        )
    );

    // Copy host data to device
    Timer timer;
    timer.start();
    checkCudaErrors(
        cudaMemcpyAsync(
            d_signal, h_signal.data(),
            sizeof(cufftComplex) * h_signal.size(),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    checkCudaErrors(
        cudaMemcpyAsync(
            d_filter, h_filter.data(),
            sizeof(cufftComplex) * h_filter.size(),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    cufftCreate(&plan);
    cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, BATCH_SIZE);

    // Process signal    
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);

    // Process filter
    cufftExecC2C(plan, d_filter, d_filter, CUFFT_FORWARD);

    // Multiply the signals in their frequency form
    complexMulAndScaleGPU(
        d_signal,
        d_filter,
        d_product_fft,
        FFT_SIZE
    );

    // Execute the inverse FFT on the result
    cufftExecC2C(plan, d_product_fft, d_product_fft, CUFFT_INVERSE);
    
    checkCudaErrors(
        cudaMemcpyAsync(
            h_convolved_result.data(), d_product_fft,
            sizeof(cufftComplex) * FFT_SIZE,
            cudaMemcpyDeviceToHost,
            stream
        )
    );
    timer.stop();

    double elapsedTime_ms = timer.elapsedTime_ms();

    printf (
    "\n- Avg Elapsed Time:             %20.16e Ms\n\n",
        elapsedTime_ms / 1.0e3
    );

    FILE* filePtr = fopen(output_file_name, "w");
    float tmp;
    for (int i = 0; i < FFT_SIZE - 1; i++) {
        tmp = complex_to_float(h_convolved_result[i]);
        typeSpecificfprintf(filePtr, tmp);
    }
    fclose(filePtr);
    
    /* free resources */
    cudaFree(d_signal);
    cudaFree(d_filter);
    cudaFree(d_product_fft);
    cudaFree(d_convolved_fft);

    cufftDestroy(plan);
    cudaStreamDestroy(stream);

    return EXIT_SUCCESS;
}