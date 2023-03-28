CC=gcc
CXX=g++
NVCC=nvcc

# Local Laptop
CUDAPATH=/usr/local/cuda-11.7
# Local Desktop
#CUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/cuda-samples/Common

# Remote Alabama ASC
# CUDAPATH=/opt/asn/apps/cuda_11.7.0
# SAMPLESPATH=/samples/common/inc
# ------- On the DMC, run this to avoid having to type in all the parameters
# echo -ne "gpu\n1\n\n10gb\n1\nampere\nconvolution_test\n" | run_gpu build/bin/convolution_test > /dev/null

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

#NVCCARCHS=-gencode arch=compute_86,code=sm_86 # Local Machine
NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 # Alabama ASC

INCPATH=-Iconvolution/include -I$(CUDAPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean

all: build/lib/libTimer.so build/lib/libconvolution.so build/bin/convolution_test

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

build/lib/libconvolution.so: convolution/src/convolution.cu
	@mkdir -p build/.objects/convolution
	$(NVCC) -pg $(NVCCFLAGS) -Xcompiler -fPIC -Iconvolution/include \
		-I$(CUDAPATH)$(SAMPLESPATH) -dc -o \
		build/.objects/convolution/convolution.o \
		convolution/src/convolution.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/convolution/convolution-dlink.o \
		build/.objects/convolution/convolution.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libconvolution.so build/.objects/convolution/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../convolution/include/convolution.h build/include/convolution.h

build/bin/convolution_test: build/lib/libTimer.so build/lib/libconvolution.so \
	convolution/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-I$(CUDAPATH)/include -ITimer/include \
		-o build/bin/convolution_test convolution/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lconvolution -lcudart

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f convolution_test.*
