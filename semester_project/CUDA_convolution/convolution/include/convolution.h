#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <vector>
#include <fstream>

typedef float2 Complex;
typedef float2 cufftComplex;

void convolve_1d(float*, float*, float*, int, int);
void complexMulGPU(cufftComplex*, cufftComplex*, cufftComplex* , int);

// Utility functions

bool read_file_into_array(std::string fname, Complex arr[]);
bool read_file_into_vector(std::string fname, std::vector<cufftComplex> & arr);

template <typename T>
void dataTypeWriter(FILE*);
template<>
void dataTypeWriter<double>(FILE*);
template<>
void dataTypeWriter<cufftComplex>(FILE*);
template<>
void dataTypeWriter<float>(FILE*);

template<typename T>
void typeSpecificfprintf(FILE* fptr, T const & data);
template<>
void typeSpecificfprintf<cufftComplex>(FILE* fptr, cufftComplex const & data);
template<>
void typeSpecificfprintf<double>(FILE* fptr, double const & data);
template<>
void typeSpecificfprintf<float>(FILE* fptr, float const & data);

template<typename T>
void dumpGPUDataToFile(
                T* devicePtrToData,
                std::vector<int> dimensionsOfData,
                std::string filename
            ){

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

#endif