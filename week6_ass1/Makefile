.PHONY: clean

# Local Machine
CUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/cuda-samples/Common

# Alabama ASC
#CUDAPATH=/opt/asn/apps/cuda_11.7.0
#SAMPLESPATH=/samples/common/inc
# ------- On the DMC, run this to avoid having to type in all the parameters
# echo -ne "gpu\n1\n\n10gb\n1\nampere\nmm_test\n" | run_gpu build/bin/matrixmult_test > /dev/null

main: 1d-matrix-demo.cu
	nvcc -o main -I$(CUDAPATH)$(SAMPLESPATH) 1d-matrix-demo.cu -g -G

clean:
	rm -f main mainGPU.*
