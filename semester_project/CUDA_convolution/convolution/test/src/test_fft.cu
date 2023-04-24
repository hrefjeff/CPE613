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
    cufftHandle plan1; // Forward FFT Plan
    cufftHandle plan2; // Inverse FFT Plan
    cudaStream_t stream = NULL;
    
    bool file_status = false;
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    const char *output_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_1024.txt";

    // Initialize the signal
    vector<cufftComplex> h_signal(N);
    vector<cufftComplex> h_signal_fft(N + K - 1);
    
    file_status = read_file_into_vector(signal_file_name, h_signal);
    if (file_status == false) return EXIT_FAILURE;

    cufftComplex *d_signal = nullptr;
    cufftComplex *d_signal_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_signal),
                sizeof(cufftComplex) * h_signal.size());
    cudaMalloc(reinterpret_cast<void **>(&d_signal_fft),
                sizeof(cufftComplex) * h_signal_fft.size());

    // Initialize the filter
    vector<cufftComplex> h_filter(K);
    vector<cufftComplex> h_filter_fft(N + K - 1);

    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return EXIT_FAILURE;
    
    cufftComplex *d_filter = nullptr;
    cufftComplex *d_filter_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_filter),
                sizeof(cufftComplex) * h_filter.size());
    cudaMalloc(reinterpret_cast<void **>(&d_filter_fft),
                sizeof(cufftComplex) * h_filter_fft.size());

    // Initialize the product
    vector<cufftComplex> h_product_fft(N + K - 1);
    cufftComplex *d_product_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(cufftComplex) * N + K - 1);

    vector<cufftComplex> h_result(N + K - 1);

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
    cufftPlan1d(&plan1, N + K - 1, CUFFT_C2C, BATCH_SIZE);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan1, stream);

    // Process signal    
    cufftExecC2C(plan1, d_signal, d_signal_fft, CUFFT_FORWARD);

    // Process filter
    cufftExecC2C(plan1, d_filter, d_filter_fft, CUFFT_FORWARD);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    checkCudaErrors(cudaGetLastError());

    dumpGPUDataToFile(d_signal_fft, {N + K - 1,1}, "cuda-fft-signal.txt");
    dumpGPUDataToFile(d_filter_fft, {N + K - 1,1}, "cuda-fft-filter.txt");

    // cudaMemcpyAsync(h_signal_fft.data(), d_signal, sizeof(cufftComplex) * N,
    //                              cudaMemcpyDeviceToHost, stream);
    // cudaMemcpyAsync(h_filter_fft.data(), d_filter, sizeof(cufftComplex) * K,
    //                              cudaMemcpyDeviceToHost, stream);

    // std::printf("Host signal fft array:\n");
    // for (int i = 0; i < 20; i++) {
    //     std::printf("%f + %fj\n", h_signal_fft[i].x, h_signal_fft[i].y);
    // }
    // std::printf("=====\n");

    // Multiplication section
    complexMulGPU(
        d_signal,
        d_filter,
        d_product_fft,
        N + K - 1
    );

    // printf("Host product fft:\n");
    // int z = 0;
    // for (auto &i : h_product_fft) {
    //     printf("%d : %f\n", z++, i.x);
    // }
    // printf("=====\n");

    // cudaMemcpyAsync(d_product_fft, h_product_fft.data(), sizeof(cufftComplex) * N,
    //                              cudaMemcpyHostToDevice, stream);

    // dumpGPUDataToFile(d_product_fft, {N,1}, "test3.txt");

    // Perform inverse
    cufftCreate(&plan2);
    cufftPlan1d(&plan2, h_product_fft.size(), CUFFT_C2C, BATCH_SIZE);

    // Execute the inverse FFT on the result
    cufftExecC2C(plan2, d_product_fft, d_product_fft, CUFFT_INVERSE);

    cudaStreamSynchronize(stream); // force CPU thread to wait
    
    cudaMemcpyAsync(
        h_result.data(), d_product_fft,
        sizeof(cufftComplex) * N + K - 1,
        cudaMemcpyDeviceToHost,
        stream
    );

    //dumpGPUDataToFile(d_product_fft, {N,1}, output_file_name);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    FILE* filePtr = fopen(output_file_name, "w");
    float tmp;
    for(auto elt : h_result) {
        tmp = complex_to_float(elt);
        // support multiple types or use C++
        typeSpecificfprintf(filePtr, tmp);
    }
    fclose(filePtr);
    
    /* free resources */
    cudaFree(d_signal);
    cudaFree(d_signal_fft);
    cudaFree(d_filter);
    cudaFree(d_filter_fft);
    cudaFree(d_product_fft);

    cufftDestroy(plan1);
    cufftDestroy(plan2);
    cudaStreamDestroy(stream);
    
    

    cudaDeviceReset();

    return EXIT_SUCCESS;
}