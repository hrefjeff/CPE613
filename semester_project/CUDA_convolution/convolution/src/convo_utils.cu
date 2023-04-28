#include <convo_utils.h>

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
bool read_file_into_vector(std::string filename, std::vector<cufftComplex> & arr) {
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