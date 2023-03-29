# CUDA implementation

## How to build on DMC

1. Open a shell and navigate to this directory
2. Open the `Makefile`
3. Ensure the settings under `Remote Alabama ASC` are uncommented
4. Run the command `make` in the terminal
5. Run the command `echo -ne "gpu\n1\n\n10gb\n1\nampere\nconvolution_test\n" | run_gpu build/bin/convolution_test > /dev/null`
6. View the results
