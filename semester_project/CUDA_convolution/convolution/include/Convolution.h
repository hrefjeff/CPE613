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

        void allocate_memory();

        void read_file_into_signal(std::string);
        void read_file_into_filter(std::string);
        void write_results_to_file(const char*);
        
        void compute();

        int next_power_of_two(int n);

        std::vector<cufftComplex> get_signal();
        std::vector<cufftComplex> get_filter();

    private:
        enum class Method {
            TimeBased,
            FFTBased
        };
        
        void convolve_1d_time(
            cufftComplex*, cufftComplex*, cufftComplex*, int, int
        );
        void convolve_1d_fft(
            cufftComplex*, cufftComplex*, cufftComplex* , int
        );

        Method _method;

        std::vector<cufftComplex> _hc_signal;
        std::vector<cufftComplex> _hc_filter;
        std::vector<cufftComplex> _hc_output;

        cufftComplex* _dc_signal;
        cufftComplex* _dc_filter;
        cufftComplex* _dc_output;

        int _signal_size;
        int _output_size;
        int _batch_size;
};

#endif