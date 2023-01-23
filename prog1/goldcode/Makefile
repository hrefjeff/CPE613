.PHONY: clean

#main: saxpy.cpp
#	g++ -o main saxpy.cpp -lblas -lgfortran

main: saxpy.cu
	nvcc -o main -I/opt/asn/apps/cuda_11.7.0/samples/common/inc saxpy.cu

clean:
	rm -f main
