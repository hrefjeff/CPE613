#ifndef CONVO_UTILS_CUH_
#define CONVO_UTILS_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <vector>
#include <fstream>

typedef float2 cufftComplex;

// Utility functions
float complex_to_float(cufftComplex value);
cufftComplex float_to_complex(float value);
bool read_file_into_array(std::string fname, cufftComplex arr[]);
bool read_file_into_vector(std::string fname, std::vector<cufftComplex> & arr);
bool write_results_to_file(const char* fname, std::vector<cufftComplex> arr);

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
void dumpGPUDataToFile(T*,std::vector<int>,std::string);

#endif