/* TODO: Implement Callbacks

https://developer.nvidia.com/blog/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/


*/

/* Include C++ stuff */
#include <complex>
#include <string.h>
#include <cstdio>
#include <cstdlib>

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
    cufftHandle plan1;
    cufftHandle plan2;
    cudaStream_t stream = NULL;

    int FFT_SIZE = next_power_of_2(N + K - 1);
    
    bool file_status = false;
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_1024.txt";

    // Initialize the signal
    vector<cufftComplex> h_signal(FFT_SIZE, cufftComplex{0});
    vector<cufftComplex> h_signal_fft(FFT_SIZE, cufftComplex{0});
    
    file_status = read_file_into_vector(signal_file_name, h_signal);
    if (file_status == false) return EXIT_FAILURE;

    cufftComplex *d_signal = nullptr;
    cufftComplex *d_signal_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_signal),
                sizeof(cufftComplex) * h_signal.size());
    cudaMalloc(reinterpret_cast<void **>(&d_signal_fft),
                sizeof(cufftComplex) * h_signal_fft.size());

    // Initialize the filter
    vector<cufftComplex> h_filter(FFT_SIZE, cufftComplex{0});
    vector<cufftComplex> h_filter_fft(FFT_SIZE, cufftComplex{0});

    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return EXIT_FAILURE;
    
    cufftComplex *d_filter = nullptr;
    cufftComplex *d_filter_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_filter),
                sizeof(cufftComplex) * h_filter.size());
    cudaMalloc(reinterpret_cast<void **>(&d_filter_fft),
                sizeof(cufftComplex) * h_filter_fft.size());

    // Initialize the product
    vector<cufftComplex> h_convolved_result(FFT_SIZE, cufftComplex{0});
    
    cufftComplex *d_convolved_fft = nullptr;
    cufftComplex *d_product_fft = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_convolved_fft),
                sizeof(cufftComplex) * FFT_SIZE);
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(cufftComplex) * FFT_SIZE);

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

    cufftCreate(&plan1);
    cufftPlan1d(&plan1, FFT_SIZE, CUFFT_C2C, BATCH_SIZE);
    //cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    //cufftSetStream(plan1, stream);

    // Process signal    
    cufftExecC2C(plan1, d_signal, d_signal_fft, CUFFT_FORWARD);

    // Process filter
    cufftExecC2C(plan1, d_filter, d_filter_fft, CUFFT_FORWARD);

    //cudaStreamSynchronize(stream); // force CPU thread to wait

    checkCudaErrors(cudaGetLastError());

    // dumpGPUDataToFile(d_signal_fft, {FFT_SIZE,1}, "cuda-fft-signal.txt");
    // dumpGPUDataToFile(d_filter_fft, {FFT_SIZE,1}, "cuda-fft-filter.txt");

    // Multiplication section
    complexMulAndScaleGPU(
        d_signal_fft,
        d_filter_fft,
        d_product_fft,
        FFT_SIZE
    );

    // dumpGPUDataToFile(d_product_fft, {FFT_SIZE,1}, "test.txt");

    // Perform inverse
    cufftCreate(&plan2);
    cufftPlan1d(&plan2, FFT_SIZE, CUFFT_C2C, BATCH_SIZE);

    // Execute the inverse FFT on the result
    cufftExecC2C(plan2, d_product_fft, d_convolved_fft, CUFFT_INVERSE);
    
    cudaMemcpyAsync(
        h_convolved_result.data(), d_convolved_fft,
        sizeof(cufftComplex) * FFT_SIZE,
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream); // force CPU thread to wait

    FILE* filePtr = fopen(output_file_name, "w");
    float tmp;
    for (int i = 0; i < FFT_SIZE - 1; i++) {
        tmp = complex_to_float(h_convolved_result[i]);
        typeSpecificfprintf(filePtr, tmp);
    }
    fclose(filePtr);
    
    /* free resources */
    cudaFree(d_signal);
    cudaFree(d_signal_fft);
    cudaFree(d_filter);
    cudaFree(d_filter_fft);
    cudaFree(d_product_fft);
    cudaFree(d_convolved_fft);

    cufftDestroy(plan1);
    cufftDestroy(plan2);
    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}