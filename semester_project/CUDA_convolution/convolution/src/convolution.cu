#include <convolution.h>

static __device__ __host__ inline
Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

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
void convolve_1d_time (
    float* input,
    float* filter,
    float* output,
    int N,
    int K
) {

    int numOfThreads = 32;
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
    int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

    complexMulGPUKernel<<<gridSize, blockSize>>>(input1, input2, output, size);

    checkCudaErrors(cudaGetLastError());
}

/***
 * Calculate the next closest power of 2
 * @param num: number to calculate the next power of 2
*/
int next_power_of_2(int num) {
    return 1 << (int(log2(num - 1)) + 1);
}

/***
 * Convert a complex number to a float
 * @param value: complex number to convert
*/
float complex_to_float(cufftComplex value) {
    float float_value;
    float_value = value.x; // Remove imaginary part of number
    return float_value;
}

/***
 * Convert a float to a complex number
 * @param value: float to convert
*/
cufftComplex float_to_complex(float value) {
    cufftComplex complex_value;
    complex_value.x = value;   // Assign the float value to the real part
    complex_value.y = 0.0f;    // Set the imaginary part to zero
    return complex_value;
}

/***
 * Read a list of floating point numbers from a file into an array
 * @param filename - the name of the file to read
 * @param arr - the array to read the file into
*/
bool read_file_into_array(std::string filename, cufftComplex arr[]) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            arr[index++] = float_to_complex(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
        return false;
    }
    return true;
}

/***
* Reads a file into a vector of Complex numbers
* @param filename - the name of the file to read
* @param arr - the vector to read the file into
***/
bool read_file_into_vector(std::string filename, std::vector<Complex> & arr) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            arr[index++] = float_to_complex(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
        return false;
    }
    return true;
}

/***
 * Write a file into a vector of Complex numbers
 * @param filename - the name of the file to write to
 * @param arr - the vector to get the data from
*/
bool write_results_to_file(const char* fname, std::vector<cufftComplex> arr) {
    FILE* filePtr = fopen(fname, "w");
    if (filePtr != NULL) {
        float tmp;
        for (int i = 0; i < arr.size() - 1; i++) {
            tmp = complex_to_float(arr[i]);
            typeSpecificfprintf(filePtr, tmp);
        }
        fclose(filePtr);
    }
    else {
        std::cout << "Unable to open file." << std::endl;
        return false;
    }
    return true;
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