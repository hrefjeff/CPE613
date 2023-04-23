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
    cufftHandle plan3;
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
    vector<cufftComplex> h_signal_fft(N);
    
    file_status = read_file_into_vector(signal_file_name, h_signal);
    if (file_status == false) return EXIT_FAILURE;

    cufftComplex *d_signal = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_signal),
                sizeof(cufftComplex) * h_signal.size());

    // Initial the filter
    vector<cufftComplex> h_filter(K);

    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return EXIT_FAILURE;
    
    cufftComplex *d_filter = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_filter),
                sizeof(cufftComplex) * h_filter.size());

    // Initial the product
    vector<cufftComplex> h_product_fft(N);
    cufftComplex *d_product_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(cufftComplex) * N);

    vector<cufftComplex> h_result(N);

    // printf("Signal array:\n");
    // int x = 0;
    // for (auto &i : h_signal) {
    //     printf("%d : %f\n", x++, i.x);
    // }
    // printf("=====\n");

    // printf("Filter array:\n");
    // int y = 0;
    // for (auto &i : h_filter) {
    //     printf("%d : %f\n", y++, i);
    // }
    // printf("=====\n");

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
    cufftPlan1d(&plan1, h_signal.size(), CUFFT_C2C, BATCH_SIZE);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan1, stream);
    
    // Perform transformation in place since idata&odata are the same
    cufftExecC2C(plan1, d_signal, d_signal, CUFFT_FORWARD);

    cufftCreate(&plan2);
    cufftPlan1d(&plan2, h_filter.size(), CUFFT_C2C, BATCH_SIZE);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan2, stream);

    cufftExecC2C(plan2, d_filter, d_filter, CUFFT_FORWARD);
    cufftDestroy(plan2);

    cudaStreamSynchronize(stream); // force CPU thread to wait

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
        N
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
    cufftCreate(&plan3);
    cufftPlan1d(&plan3, h_product_fft.size(), CUFFT_C2C, BATCH_SIZE);

    // Execute the inverse FFT on the result
    cufftExecC2C(plan3, d_product_fft, d_product_fft, CUFFT_INVERSE);

    cudaStreamSynchronize(stream); // force CPU thread to wait
    
    cudaMemcpyAsync(h_result.data(), d_product_fft, sizeof(cufftComplex) * N,
                                 cudaMemcpyDeviceToHost, stream);

    //dumpGPUDataToFile(d_product_fft, {N,1}, output_file_name);

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
    cudaFree(d_filter);
    cudaFree(d_product_fft);

    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}