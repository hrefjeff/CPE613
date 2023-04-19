#include <convolution.h>

static __device__ __host__ inline
cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__global__ void convolve_1d_time_kernel (
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

void convolve_1d (
    float* input,
    float* filter,
    float* output,
    int N,
    int K
){

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

__global__ void complexMulGPUKernel(
                    cufftComplex* input1,
                    cufftComplex* input2,
                    cufftComplex* output,
                    int size
                ){
                            
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
        int size 
    ) {
    int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

    complexMulGPUKernel<<<gridSize, blockSize>>>(output, input1, input2, size);

    checkCudaErrors(cudaGetLastError());
}

bool read_file_into_vector(std::string filename, std::vector<float>& arr) {
    std::ifstream the_file(filename);

    if (the_file.is_open()) {
        int index = 0;
        float value;
        while (the_file >> value) {
            arr[index++] = (float)(value);
        }
        the_file.close();
    } else {
        std::cout << "Unable to open signal file." << std::endl;
        return false;
    }
    return true;
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
void typeSpecificfprintf<cufftComplex>(FILE* fptr, cufftComplex const & data){

    fprintf(fptr, "%20.16f %20.16f\n", data.x, data.y);

}

template<>
void typeSpecificfprintf<double>(FILE* fptr, double const & data){

    fprintf(fptr, "%20.16f\n", data);

}