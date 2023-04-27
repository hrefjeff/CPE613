class Convolution {
    public:
        Convolution(
            std::vector<float>& signal,
            std::vector<float>& filter,
            const int batch_size
        );
        std::vector<float> compute();
        static int next_power_of_two(int n);

    private:
        enum class Method {
            TimeBased,
            FFTBased
        };

        std::vector<float> time_based_convolution(
            std::vector<float> & signal,
            std::vector<float> & filter
        );

        std::vector<float> fft_based_convolution(
            std::vector<float> & signal,
            std::vector<float> & filter
        );

        Method _method;
        std::vector<float> _signal;
        std::vector<float> _filter;
        int _batch_size;
};