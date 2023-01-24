NVCC = nvcc
NVCCFLAGS = 
NVCCARCHS =
CC = g++
CUDAPATH = /opt/asn/apps/cuda_11.7.0

build/lib/libsaxpy.so: modules saxpy/src/saxpy.cu
	@mkdir -p build/.objects/saxpy
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Isaxpy/include -I$(CUDAPATH)/samples/common/inc \
		-dc -o build/.objects/saxpy/saxpy.o \
		saxpy/src/saxpy.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/saxpy/saxpy-dlink.o build/.objects/saxpy/saxpy.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libsaxpy.so build/.objects/saxpy/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudaart
	@mkdir  -p build/include
	@ln -sf ../../saxpy/include/saxpy.h build/include/saxpy.h

clean:
	rm -f build/.objects/saxpy/* *.o *~