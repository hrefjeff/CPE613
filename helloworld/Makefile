.PHONY: clean

LOCSAMPLESPATH=/usr/local/cuda/cuda-samples/Common

main: main.cpp
	nvcc -o main -I$(LOCSAMPLESPATH) main.cpp

clean:
	rm -f main mainGPU.*
