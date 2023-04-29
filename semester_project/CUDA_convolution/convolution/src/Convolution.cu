#include <Convolution.h>

/**
 * Constructor for Convolution class
 * @param sizeOfSignals: size of input signal(s)
 * @param numOfSignals: number of input signal(s)
*/
Convolution::Convolution(
    int sizeOfSignals,
    int numOfSignals
) : _signal_size(sizeOfSignals), _batch_size(numOfSignals) {
    if (sizeOfSignals <= 512) {
        _method = Method::TimeBased;
    } else {
        _method = Method::FFTBased;
    }
}

std::vector<cufftComplex> Convolution::get_signal() {
    return _hc_signal;
}

std::vector<cufftComplex> Convolution::get_filter() {
    return _hc_filter;
}

void Convolution::allocate_memory() {
    _output_size = next_power_of_two(_signal_size + _signal_size - 1);
    
    if (_method == Method::TimeBased) {
        _hc_signal.resize(_signal_size, cufftComplex{0});
        _hc_filter.resize(_signal_size, cufftComplex{0});
    } else if (_method == Method::FFTBased) {
        _hc_signal.resize(_output_size, cufftComplex{0});
        _hc_filter.resize(_output_size, cufftComplex{0});
    }
    
    _hc_output.resize(_output_size, cufftComplex{0});

    checkCudaErrors(
        cudaMalloc(
            (void **)&_dc_signal,
            _hc_signal.size() * sizeof(cufftComplex)
        )
    );
    checkCudaErrors(
        cudaMalloc(
            (void **)&_dc_filter,
            _hc_filter.size() * sizeof(cufftComplex)
        )
    );
    checkCudaErrors(
        cudaMalloc(
            (void **)&_dc_output,
            _hc_output.size() * sizeof(cufftComplex)
        )
    );
}

/***
* Reads a file into a signal vector of type cufftComplex
* @param filename - the name of the file to read
***/
void Convolution::read_file_into_signal(std::string filename) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            _hc_signal[index++] = float_to_complex(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
    }
}

/***
* Reads a file into a filter vector of type cufftComplex
* @param filename - the name of the file to read
***/
void Convolution::read_file_into_filter(std::string filename) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            _hc_filter[index++] = float_to_complex(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
    }
    checkCudaErrors(
        cudaMemcpy(
            _dc_filter, _hc_filter.data(),
            _hc_filter.size() * sizeof(cufftComplex),
            cudaMemcpyHostToDevice
        )
    );
}

/***
 * Write results to file
*/
void Convolution::write_results_to_file(const char* file_name) {
    checkCudaErrors(
        cudaMemcpy(
            _hc_output.data(), _dc_output,
            _hc_output.size() - 1 * sizeof(cufftComplex),
            cudaMemcpyDeviceToHost
        )
    );

    FILE* filePtr = fopen(file_name, "w");
    for (int i = 0; i < _hc_output.size() - 1; i++) {
        fprintf(
            filePtr,
            "%20.16e\n",
            complex_to_float(_hc_output[i])
        );
    }
    fclose(filePtr);
}

/***
 * Compute the convolution
*/
void Convolution::compute(){
    if (_method == Method::TimeBased) {
        checkCudaErrors(
            cudaMemcpy(
                _dc_signal, _hc_signal.data(),
                _signal_size * sizeof(cufftComplex),
                cudaMemcpyHostToDevice
            )
        );
        checkCudaErrors(
            cudaMemcpy(
                _dc_filter, _hc_filter.data(),
                _signal_size * sizeof(cufftComplex),
                cudaMemcpyHostToDevice
            )
        );
        convolve_1d_time(
            _dc_signal,
            _dc_filter,
            _dc_output,
            _signal_size,
            _signal_size
        );
    } else {
        
        cufftHandle plan;
        cufftCreate(&plan);
        cufftPlan1d(&plan, _output_size, CUFFT_C2C, _batch_size);

        checkCudaErrors(
            cudaMemcpy(
                _dc_signal, _hc_signal.data(),
                _hc_signal.size() * sizeof(cufftComplex),
                cudaMemcpyHostToDevice
            )
        );
        checkCudaErrors(
            cudaMemcpy(
                _dc_filter, _hc_filter.data(),
                _hc_filter.size() * sizeof(cufftComplex),
                cudaMemcpyHostToDevice
            )
        );
        
        // Process signal    
        checkCudaErrors(
            cufftExecC2C(plan, _dc_signal, _dc_signal, CUFFT_FORWARD)
        );

        // Process filter
        checkCudaErrors(
            cufftExecC2C(plan, _dc_filter, _dc_filter, CUFFT_FORWARD)
        );

        checkCudaErrors(cudaDeviceSynchronize());

        convolve_1d_fft(
            _dc_signal,
            _dc_filter,
            _dc_output,
            _output_size
        );
        
        // Perform inverse to get result
        checkCudaErrors(
            cufftExecC2C(
                plan, _dc_output, _dc_output, CUFFT_INVERSE
            )
        );
        
        cufftDestroy(plan);
    }
}

/***
 * Find the next power of two above the number provided
*/
int Convolution::next_power_of_two(int num) {
    return 1 << (int(log2(num - 1)) + 1);
}

/***
 * Scale the value based on the size of the signal
*/
static __device__ __host__ inline
cufftComplex ComplexScale(cufftComplex a, float s) {
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

/***
 * Perform multiplication of two complex numbers
*/
static __device__ __host__ inline
cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex addition
static __device__ __host__ inline
cufftComplex ComplexAdd(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

/***
 * @brief: 1D convolution in time domain
 * @param input: input signal
 * @param kernel: filter
 * @param output: output signal
 * @param N: length of input signal
 * @param K: length of filter
*/
__global__
void convolve_1d_time_kernel (
    cufftComplex *input, 
    cufftComplex *kernel,
    cufftComplex *output,
    int N,
    int K
) {
    int idxInput = threadIdx.x + blockIdx.x * blockDim.x;
    if (idxInput > N + K - 1) return;

    cufftComplex result = cufftComplex{0.0};
    for (int idxFilter = 0; idxFilter < K; idxFilter++) {
        if((idxInput - idxFilter) < 0 || (idxInput - idxFilter) >= N) {
            result = ComplexAdd(result, cufftComplex{0.0});
        }
        else {
            result = ComplexAdd(
                        result,
                        ComplexMul(
                            kernel[idxFilter],
                            input[idxInput - idxFilter]
                        )
                );
        }
    }
    output[idxInput] = result;
}

/***
 * @brief: 1D convolution in frequency domain
 * @param input: input signal
 * @param kernel: filter
 * @param output: output signal
 * @param N: length of input signal
 * @param K: length of filter
*/
void Convolution::convolve_1d_time (
    cufftComplex* input,
    cufftComplex* filter,
    cufftComplex* output,
    int N,
    int K
) {

    int numOfThreads = 1024;
    int numOfBlocks = ((N + K - 1) + numOfThreads - 1) / numOfThreads;

    convolve_1d_time_kernel<<<numOfBlocks, numOfThreads>>> 
    (
        input,
        filter,
        output,
        N,
        K
    );
  
    checkCudaErrors(cudaGetLastError());
}

/***
 * @brief: 1D convolution in frequency domain
 * @param input: input signal
 * @param kernel: filter
 * @param output: output signal
 * @param size: size of FFT
*/
__global__
void complexMulGPUKernel(
    cufftComplex* input1,
    cufftComplex* input2,
    cufftComplex* output,
    int size
) {
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ){
        output[idx] = ComplexScale(
                        ComplexMul(input1[idx], input2[idx]),
                        1.0 / size
                    );
    }
}

/***
 * Perform DFT on input signal
 * @param input: input signal
 * @param input2: filter kernel
 * @param output: output signal
 * @param size: size of FFT
*/
void Convolution::convolve_1d_fft(
        cufftComplex* input1,
        cufftComplex* input2,
        cufftComplex* output,
        int size 
) {
    int blockSize = 1024;
    int gridSize = (size + blockSize - 1) / blockSize;

    complexMulGPUKernel<<<gridSize, blockSize>>>(input1, input2, output, size);

    checkCudaErrors(cudaGetLastError());
}


/***
 * Dump GPU data to a file
 * @param devicePtrToData: pointer to data on GPU
 * @param dimensionsOfData: dimensions of data
 * @param filename: name of file to dump data to
*/
template<typename T>
void dumpGPUDataToFile(
    T* devicePtrToData,
    std::vector<int> dimensionsOfData,
    std::string filename
) {

    //checkCudaErrors(cudaDeviceSynchronize()); // force GPU thread to wait

    int totalNumElements = 1;
    for(auto elts : dimensionsOfData) {
        totalNumElements *= elts;
    }

    std::vector<T> hostData(totalNumElements, T{0});

    checkCudaErrors(cudaMemcpy(
        hostData.data(),
        devicePtrToData,
        totalNumElements * sizeof(T),
        cudaMemcpyDeviceToHost
    ));


    // size of vector of dims
    FILE* filePtr = fopen(filename.c_str(), "w");
    // write how many dims we have
    fprintf(filePtr, "%zu\n", dimensionsOfData.size());
    for(auto elts : dimensionsOfData) {
        fprintf(filePtr,"%d\n", elts);
    }

    dataTypeWriter<T>(filePtr);

    for(auto elt : hostData) {
        // support multiple types or use C++
        typeSpecificfprintf(filePtr, elt);
    }
    fclose(filePtr);
}


template <typename T>
void dataTypeWriter(FILE* filePtr);

template<>
void dataTypeWriter<double>(FILE* filePtr){
    fprintf(filePtr, "double\n");
}

template<>
void dataTypeWriter<cufftComplex>(FILE* filePtr){
    fprintf(filePtr, "complex\n");
}

template<>
void dataTypeWriter<float>(FILE* filePtr){
    fprintf(filePtr, "float\n");
}

template<>
void typeSpecificfprintf<cufftComplex>(FILE* fptr, cufftComplex const & data){

    fprintf(fptr, "%20.16e %20.16e\n", data.x, data.y);

}

template<>
void typeSpecificfprintf<double>(FILE* fptr, double const & data){

    fprintf(fptr, "%20.16f\n", data);

}

template<>
void typeSpecificfprintf<float>(FILE* fptr, float const & data){

    fprintf(fptr, "%20.16e\n", data);

}