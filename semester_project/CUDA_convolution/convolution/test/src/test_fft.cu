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

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = complex<scalar_type>;

    vector<input_type> h_signal(N, 0);
    vector<cufftComplex> h_signal_fft(static_cast<int>((N / 2 + 1)));
    input_type *d_signal = nullptr;
    cufftComplex *d_signal_fft = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signal),
                sizeof(input_type) * h_signal.size());
    cudaMalloc(reinterpret_cast<void **>(&d_signal_fft),
                sizeof(output_type) * h_signal_fft.size());

    vector<input_type> h_filter(K, 0);
    vector<cufftComplex> h_filter_fft(static_cast<int>((K / 2 + 1)));
    input_type *d_filter = nullptr;
    cufftComplex *d_filter_fft = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_filter),
                sizeof(input_type) * h_filter.size());
    cudaMalloc(reinterpret_cast<void **>(&d_filter_fft),
                sizeof(output_type) * h_filter_fft.size());

    vector<cufftComplex> h_product_fft(static_cast<int>((N / 2 + 1)));
    cufftComplex *d_product_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(output_type) * static_cast<int>((N / 2 + 1)));

    cufftReal* d_result = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(cufftReal) * N);

    vector<input_type> h_result(N, 0);
    
    cudaStreamSynchronize(stream); // force CPU thread to wait

    // Prepare to read signal and filter information from files
    string signal_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr1_1024.txt";
    string filter_file_name =
        "/home/jeff/code/CPE613/semester_project/test_data_gold/arr2_1024.txt";
    // const char *output_file_name =
    //     "/home/jeff/code/CPE613/semester_project/test_data/cuda_fft_1024.txt";

    bool file_status = false;
    file_status = read_file_into_vector(signal_file_name, h_signal);
    if (file_status == false) return 1;
    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return 1;

    printf("Signal array:\n");
    int x = 0;
    for (auto &i : h_signal) {
        printf("%d : %f\n", x, i);
        x++;
    }
    printf("=====\n");

    printf("Filter array:\n");
    int y = 0;
    for (auto &i : h_filter) {
        printf("%d : %f\n", y, i);
        y++;
    }
    printf("=====\n");

    return 0;

    cudaMemcpyAsync(d_signal, h_signal.data(),
                    sizeof(input_type) * h_signal.size(),
                    cudaMemcpyHostToDevice,
                    stream
                );

    cudaMemcpyAsync(d_filter, h_signal.data(),
                    sizeof(input_type) * h_filter.size(),
                    cudaMemcpyHostToDevice,
                    stream
                );

    cufftCreate(&plan1);
    cufftPlan1d(&plan1, h_signal.size(), CUFFT_R2C, N);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan1, stream);

    cufftExecR2C(plan1, d_signal, d_signal_fft);
    cufftDestroy(plan1);

    cufftCreate(&plan2);
    cufftPlan1d(&plan2, h_filter.size(), CUFFT_R2C, K);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan2, stream);

    cufftExecR2C(plan2, d_filter, d_filter_fft);
    cufftDestroy(plan2);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    // Multiplication section
    complexMulGPU(
        d_signal_fft,
        d_filter_fft,
        d_product_fft,
        static_cast<int>((N / 2 + 1))
    );

    // Perform inverse
    cufftCreate(&plan3);
    cufftPlan1d(&plan3, h_product_fft.size(), CUFFT_C2R, BATCH_SIZE);

    // Execute the inverse FFT on the result
    cufftExecC2R(plan3, d_product_fft, (cufftReal*)d_result);
    
    cudaMemcpyAsync(h_result.data(), d_result, sizeof(cufftReal) * N,
                                 cudaMemcpyDeviceToHost, stream);

    printf("Real result array:\n");
    for (auto &i : h_result) {
        printf("%f\n", i);
    }
    printf("=====\n");
    
    /* free resources */
    cudaFree(d_signal);
    cudaFree(d_signal_fft);
    cudaFree(d_filter);
    cudaFree(d_filter_fft);
    cudaFree(d_result);
    cudaFree(d_product_fft);

    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

/*

std::printf("Signal array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_signal[i].x, hc_signal[i].y);
    }
    std::printf("=====\n");

std::printf("Filter array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_filter[i].x, hc_filter[i].y);
    }
    std::printf("=====\n");


std::printf("Host complex output array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f + %fj\n", hc_output[i].x, hc_output[i].y);
    }
    std::printf("=====\n");

std::printf("Host real output array:\n");
    for (int i = 0; i < 5; i++) {
        std::printf("%f\n", h_output[i]);
    }
    std::printf("=====\n");

*/