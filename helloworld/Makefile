.PHONY: clean

LOCSAMPLESPATH=/usr/local/cuda/cuda-samples/Common

main:
	nvcc -o main.o -I$(LOCSAMPLESPATH) main.cpp

clean:
	rm -f main.o mainGPU.* tgb.o
