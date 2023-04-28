#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector>

#include <convo_utils.h>

typedef float2 cufftComplex;

class Convolution {
    public:
        Convolution(
            int sizeOfSignals,
            int numberOfSignals
        );

        void allocate_float_memory();
        void allocate_complex_memory();

        void read_file_into_array(std::string, float*);
        void read_file_into_complex_array(std::string, std::vector<cufftComplex> &);

        void write_results_to_file(const char*);
        void write_complex_results_to_file(const char*);
        
        void compute();

        int next_power_of_two(int n);

        float* get_signal();
        float* get_filter();
        std::vector<cufftComplex> get_signal_complex();
        std::vector<cufftComplex> get_filter_complex();

    private:
        enum class Method {
            TimeBased,
            FFTBased
        };
        
        void convolve_1d_time(float*, float*, float*, int, int);
        void convolve_1d_fft(cufftComplex*, cufftComplex*, cufftComplex* , int);

        Method _method;

        // Time-based vars
        float* _hf_signal;
        float* _hf_filter;
        float* _hf_output;

        float* _df_signal;
        float* _df_filter;
        float* _df_output;
        
        //  FFT-based variables
        std::vector<cufftComplex> _hc_signal;
        std::vector<cufftComplex> _hc_filter;
        std::vector<cufftComplex> _hc_convolved_result;

        cufftComplex* _dc_signal;
        cufftComplex* _dc_filter;
        cufftComplex* _dc_convolved_result;

        int _signal_size;
        int _fft_size;
        int _batch_size;
};

#endif