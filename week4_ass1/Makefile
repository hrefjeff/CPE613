CC=gcc
CXX=g++
NVCC=nvcc

#BLASPATH=/opt/asn/apps/blas_gcc610_avx
# Local Machine
#CUDAPATH=/usr/local/cuda-12.0
#SAMPLESPATH=/cuda-samples/Common

# Alabama ASC
CUDAPATH=/opt/asn/apps/cuda_11.7.0
SAMPLESPATH=/samples/common/inc
# ------- On the DMC, run this to avoid having to type in all the parameters
# echo -ne "gpu\n1\n\n10gb\n1\nampere\nmatrixmult_test\n" | run_gpu build/bin/matrixmult_test > /dev/null

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

#NVCCARCHS=-gencode arch=compute_86,code=sm_86 # Local Machine
#NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 # Alabama ASC

INCPATH=-Imatrixmult/include -I$(CUDAPATH)/include -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean

all: build/lib/libTimer.so build/lib/libmatrixmult.so build/bin/matrixmult_test

build/lib/libTimer.so: Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(CUDAPATH)/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
	@mkdir -p build/lib
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(CUDAPATH)/lib64 -lcudart_static
	@mkdir -p build/include
	@ln -sf ./Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libmatrixmult.so: matrixmult/src/matrixmult.cu
	@mkdir -p build/.objects/matrixmult
	$(NVCC) -pg $(NVCCFLAGS) -Xcompiler -fPIC -Imatrixmult/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-dc -o build/.objects/matrixmult/matrixmult.o \
		matrixmult/src/matrixmult.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/matrixmult/matrixmult-dlink.o build/.objects/matrixmult/matrixmult.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libmatrixmult.so build/.objects/matrixmult/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../matrixmult/include/matrixmult.h build/include/matrixmult.h

build/bin/matrixmult_test: build/lib/libTimer.so build/lib/libmatrixmult.so \
	matrixmult/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-I$(CUDAPATH)/include -ITimer/include \
		-o build/bin/matrixmult_test matrixmult/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lmatrixmult -lcudart

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f matrixmult_test.*
