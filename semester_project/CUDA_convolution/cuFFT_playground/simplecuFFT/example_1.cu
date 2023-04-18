/*
 * How to build:
 * nvcc example_1.cu -I/usr/local/cuda-12.0/include -I/usr/local/cuda-12.0/cuda-samples/Common -o ex.o -L/usr/local/cuda-12.0/lib64 -lcudart -lcufft
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <complex>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cufftXt.h>

using namespace std;

typedef float2 cufftComplex;

static __device__ __host__ inline
cufftComplex ComplexMul(cufftComplex, cufftComplex);

__global__ 
void complexMulGPUKernel(
                        cufftComplex* input1,
                        cufftComplex* input2,
                        cufftComplex* output,
                        int size) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ){
        output[idx] = ComplexMul(input1[idx], input2[idx]);
    }
}

void complexMulGPU(
        cufftComplex* input1,
        cufftComplex* input2,
        cufftComplex* output,
        int size) {
    int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

    complexMulGPUKernel<<<gridSize, blockSize>>>(input1, input2, output, size);

    checkCudaErrors(cudaGetLastError());
}

void multiply_arrays_elementwise(const cufftComplex* array1,
                                 const cufftComplex* array2,
                                 cufftComplex* result,
                                 int length);


int main(int argc, char *argv[]) {
    cufftHandle plan1;
    cufftHandle plan2;
    cufftHandle plan3;
    cudaStream_t stream = NULL;

    int n = 8;
    int batch_size = 2;
    int fft_size = batch_size * n;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = complex<scalar_type>;

    // Filter section to convert to fft

    vector<input_type> h_signal(fft_size, 0);
    vector<cufftComplex> h_signal_fft(static_cast<int>((fft_size / 2 + 1)));

    for (int i = 0; i < fft_size; i++) {
        h_signal[i] = static_cast<input_type>(i);
    }

    printf("Signal array:\n");
    for (auto &i : h_signal) {
        printf("%f\n", i);
    }
    printf("=====\n");

    input_type *d_signal = nullptr;
    cufftComplex *d_signal_fft = nullptr;

    cufftCreate(&plan1);
    cufftPlan1d(&plan1, h_signal.size(), CUFFT_R2C, batch_size);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan1, stream);

    // Create device arrays
    cudaMalloc(reinterpret_cast<void **>(&d_signal), sizeof(input_type) * h_signal.size());
    cudaMalloc(reinterpret_cast<void **>(&d_signal_fft), sizeof(output_type) * h_signal_fft.size());
    cudaMemcpyAsync(d_signal, h_signal.data(), sizeof(input_type) * h_signal.size(),
                                 cudaMemcpyHostToDevice, stream);

    cufftExecR2C(plan1, d_signal, d_signal_fft);

    cudaMemcpyAsync(h_signal_fft.data(), d_signal_fft, sizeof(output_type) * h_signal_fft.size(),
                                 cudaMemcpyDeviceToHost, stream);

    // cudaStreamQuery(stream1);   // test if stream is idle
    cudaStreamSynchronize(stream); // force CPU thread to wait

    printf("Signal FFT array:\n");
    for (auto &i : h_signal_fft) {
        printf("%f + %fj\n", i.x, i.y);
    }
    printf("=====\n");

    cufftDestroy(plan1);

    // =================== Filter Section ===================

    vector<input_type> h_filter(fft_size, 0);
    vector<cufftComplex> h_filter_fft(static_cast<int>((fft_size / 2 + 1)));

    for (int i = 0; i < fft_size; i++) {
        h_filter[i] = static_cast<input_type>(i);
    }

    printf("Filter array:\n");
    for (auto &i : h_filter) {
        printf("%f\n", i);
    }
    printf("=====\n");

    input_type *d_filter = nullptr;
    cufftComplex *d_filter_fft = nullptr;

    cufftCreate(&plan2);
    cufftPlan1d(&plan2, h_filter.size(), CUFFT_R2C, batch_size);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cufftSetStream(plan2, stream);

    // Create device arrays
    cudaMalloc(reinterpret_cast<void **>(&d_filter), sizeof(input_type) * h_filter.size());
    cudaMalloc(reinterpret_cast<void **>(&d_filter_fft), sizeof(output_type) * h_filter_fft.size());
    cudaMemcpyAsync(d_filter, h_filter.data(), sizeof(input_type) * h_filter.size(),
                                    cudaMemcpyHostToDevice, stream);

    cufftExecR2C(plan2, d_filter, d_filter_fft);

    cudaMemcpyAsync(h_filter_fft.data(), d_filter_fft, sizeof(output_type) * h_filter_fft.size(),
                                    cudaMemcpyDeviceToHost, stream);

    // cudaStreamQuery(stream1);   // test if stream is idle
    cudaStreamSynchronize(stream); // force CPU thread to wait

    printf("Filter FFT array:\n");
    for (auto &i : h_filter_fft) {
        printf("%f + %fj\n", i.x, i.y);
    }
    printf("=====\n");

    cufftDestroy(plan2);

    // Multiplication section
    vector<cufftComplex> h_product_fft(static_cast<int>((fft_size / 2 + 1)));
    cufftComplex *d_product_fft = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_product_fft),
                sizeof(output_type) * static_cast<int>((fft_size / 2 + 1)));
    complexMulGPU(
        d_signal_fft,
        d_filter_fft,
        d_product_fft,
        static_cast<int>((fft_size / 2 + 1))
    );

    cudaMemcpyAsync(h_product_fft.data(),
                    d_product_fft,
                    sizeof(output_type) * static_cast<int>((fft_size / 2 + 1)),
                    cudaMemcpyDeviceToHost,
                    stream
    );

    printf("Multiplied signal+filter result array:\n");
    for (auto &i : h_product_fft) {
        printf("%f + %fj\n", i.x, i.y);
    }
    printf("=====\n");

    // Perform inverse
    cufftCreate(&plan3);
    cufftPlan1d(&plan3, h_product_fft.size(), CUFFT_C2R, batch_size);

    // Execute the inverse FFT on the result
    cufftReal* d_result = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(cufftReal) * fft_size);
    cufftExecC2R(plan3, d_product_fft, (cufftReal*)d_result);

    vector<input_type> h_result(fft_size, 0);
    cudaMemcpyAsync(h_result.data(), d_result, sizeof(cufftReal) * fft_size,
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

// Complex multiplication
static __device__ __host__ inline
cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
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
