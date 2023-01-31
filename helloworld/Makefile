.PHONY: clean

main: main.cpp
	nvcc -o main -I/opt/asn/apps/cuda_11.7.0/samples/common/inc main.cpp

clean:
	rm -f main mainGPU.*
