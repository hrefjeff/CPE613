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

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
void multiply_arrays_elementwise(const cufftComplex* array1,
                                 const cufftComplex* array2,
                                 vector<cufftComplex> & result,
                                 int length
                                );

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
    cufftComplex *d_signal_fft = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_signal),
                sizeof(cufftComplex) * h_signal.size());
    cudaMalloc(reinterpret_cast<void **>(&d_signal_fft),
                sizeof(cufftComplex) * h_signal_fft.size());

    // Initial the filter
    vector<cufftComplex> h_filter(K);
    vector<cufftComplex> h_filter_fft(K);

    file_status = read_file_into_vector(filter_file_name, h_filter);
    if (file_status == false) return EXIT_FAILURE;
    
    cufftComplex *d_filter = nullptr;
    cufftComplex *d_filter_fft = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_filter),
                sizeof(cufftComplex) * h_filter.size());
    cudaMalloc(reinterpret_cast<void **>(&d_filter_fft),
                sizeof(cufftComplex) * h_filter_fft.size());

    // Initial the product
    vector<cufftComplex> h_product_fft(N);
    cufftComplex *d_product_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(cufftComplex) * N);

    vector<cufftComplex> h_result(N);
    cufftComplex* d_result = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(cufftComplex) * N);

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
    
    cufftExecC2C(plan1, d_signal, d_signal, CUFFT_FORWARD);

    cufftCreate(&plan2);
    cufftPlan1d(&plan2, h_filter.size(), CUFFT_C2C, BATCH_SIZE);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan2, stream);

    cufftExecC2C(plan2, d_filter, d_filter, CUFFT_FORWARD);
    cufftDestroy(plan2);

    cudaStreamSynchronize(stream); // force CPU thread to wait

    cudaMemcpyAsync(h_signal_fft.data(), d_signal, sizeof(cufftComplex) * N,
                                 cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_filter_fft.data(), d_filter, sizeof(cufftComplex) * K,
                                 cudaMemcpyDeviceToHost, stream);

    // std::printf("Host signal fft array:\n");
    // for (int i = 0; i < 5; i++) {
    //     std::printf("%f + %fj\n", h_signal_fft[i].x, h_signal_fft[i].y);
    // }
    // std::printf("=====\n");

    // Multiplication section
    // complexMulGPU(
    //     d_signal_fft,
    //     d_filter_fft,
    //     d_product_fft,
    //     N
    // );

    multiply_arrays_elementwise(h_signal_fft.data(),
                                h_filter_fft.data(),
                                h_product_fft, 
                                N);

    // printf("Host product fft:\n");
    // int z = 0;
    // for (auto &i : h_product_fft) {
    //     printf("%d : %f\n", z++, i.x);
    // }
    // printf("=====\n");

    cudaMemcpyAsync(d_product_fft, h_product_fft.data(), sizeof(cufftComplex) * N,
                                 cudaMemcpyHostToDevice, stream);

    // dumpGPUDataToFile(d_product_fft, {N,1}, "test3.txt");

    // Perform inverse
    cufftCreate(&plan3);
    cufftPlan1d(&plan3, h_product_fft.size(), CUFFT_C2C, BATCH_SIZE);

    // Execute the inverse FFT on the result
    cufftExecC2C(plan3, d_product_fft, d_result, CUFFT_INVERSE);

    cudaStreamSynchronize(stream); // force CPU thread to wait
    
    cudaMemcpyAsync(h_result.data(), d_result, sizeof(cufftComplex) * N,
                                 cudaMemcpyDeviceToHost, stream);

    dumpGPUDataToFile(d_result, {N,1}, output_file_name);
    
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

cufftComplex float_to_complex(float value) {
    cufftComplex complex_value;
    complex_value.x = value;  // Assign the float value to the real part
    complex_value.y = 0.0f;    // Set the imaginary part to zero
    return complex_value;
}

void multiply_arrays_elementwise(const cufftComplex* array1,
                                 const cufftComplex* array2,
                                 vector<cufftComplex> & result,
                                 int length
                                ) {
    for (int i = 0; i < length; ++i) {
        result[i] = ComplexMul(array1[i], array2[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
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