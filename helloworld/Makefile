.PHONY: clean

LOCSAMPLESPATH=/usr/local/cuda/cuda-samples/Common

main:
	nvcc -o tgb.o -I$(LOCSAMPLESPATH) threadgridblock.cu

clean:
	rm -f main mainGPU.* tgb.o
