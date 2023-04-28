#ifndef CONVOLUTION_CUH_
#define CONVOLUTION_CUH_

#include <cuda_runtime.h>
#include <helper_cuda.h>
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
        void read_file_into_array(std::string, float*);
        void write_results_to_file(const char*);
        void compute();
        static int next_power_of_two(int n);

        float* get_signal();
        float* get_filter();

        // std::vector<cufftComplex> get_signal();
        // std::vector<cufftComplex> get_filter();

    private:
        enum class Method {
            TimeBased,
            FFTBased
        };
        
        void convolve_1d_time(float*, float*, float*, int, int);
        void convolve_1d_fft(cufftComplex*, cufftComplex*, cufftComplex* , int);

        Method _method;

        std::vector<float> _fsignal;
        std::vector<float> _ffilter;

        std::vector<cufftComplex> _csignal;
        std::vector<cufftComplex> _cfilter;

        float *_hf_signal;
        float *_hf_filter;
        float *_hf_output;

        float *_df_signal;
        float *_df_filter;
        float *_df_output;
        
        cufftComplex *dc_signal;
        cufftComplex *dc_filter;
        cufftComplex *dc_result;

        int _signal_size;
        int _fft_size;
        int _batch_size;
};

#endif