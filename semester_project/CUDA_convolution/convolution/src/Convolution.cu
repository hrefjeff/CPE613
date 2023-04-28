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
    if (sizeOfSignals <= 10000) {
        _method = Method::TimeBased;
    } else {
        _method = Method::FFTBased;
        _fft_size = (sizeOfSignals + sizeOfSignals - 1);
    }
}

/***
 * Return pointer to host signal array
*/
float* Convolution::get_signal() {
    return _hf_signal;
}

/***
 * Return pointer to host filter array
*/
float* Convolution::get_filter() {
    return _hf_filter;
}

/***
 * Allocate memory for signal, filter, and output on host and device
*/
void Convolution::allocate_float_memory(){
    // Allocate memory for signal, filter, and output
    int total_size = _signal_size + _signal_size - 1;
    _hf_signal = new float[_signal_size];
    _hf_filter = new float[_signal_size];
    _hf_output = new float[total_size - 1];
    checkCudaErrors(
        cudaMalloc((void **)&_df_signal, _signal_size * sizeof(float))
    );
    checkCudaErrors(
        cudaMalloc((void **)&_df_filter, _signal_size * sizeof(float))
    );
    checkCudaErrors(
        cudaMalloc((void **)&_df_output, total_size * sizeof(float))
    );
}

/***
 * Read data from file and put it into array
 * @param filename: name of file to read from
 * @param host_arr: array to put data into
*/
void Convolution::read_file_into_array(
    std::string filename,
    float* host_arr
) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            host_arr[index++] = (float)(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
    }
}

/***
 * Write results to file
*/
void Convolution::write_results_to_file(const char* file_name) {
    checkCudaErrors(
        cudaMemcpy(
            _hf_output, _df_output,
            (_signal_size + _signal_size - 1) * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );
    FILE* filePtr = fopen(file_name, "w");
    for (int i = 0; i < _signal_size + _signal_size - 1; i++) {
        fprintf (filePtr, "%20.16e\n", _hf_output[i]);
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
                _df_signal, _hf_signal,
                _signal_size * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
        checkCudaErrors(
            cudaMemcpy(
                _df_filter, _hf_filter,
                _signal_size * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
        convolve_1d_time(
            _df_signal,
            _df_filter,
            _df_output,
            _signal_size,
            _signal_size
        );
    } else {
        //convolve_1d_fft(_signal, _filter);
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
    float *input, 
    float *kernel,
    float *output,
    int N,
    int K
) {
    int idxInput = threadIdx.x + blockIdx.x * blockDim.x;
    if (idxInput > N + K - 1) return;

    float result = 0.0;
    for (int idxFilter = 0; idxFilter < K; idxFilter++) {
        if((idxInput - idxFilter) < 0 || (idxInput - idxFilter) >= N)
            result += 0;
        else
            result += (float)(kernel[idxFilter] * input[idxInput - idxFilter]);
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
    float* input,
    float* filter,
    float* output,
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
void convolve_1d_fft(
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
