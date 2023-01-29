CC=gcc
CXX=g++
NVCC=nvcc

BLASPATH=/opt/asn/apps/blas_gcc610_avx
CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Isaxpy/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean

all: build/lib/libTimer.so build/lib/libsaxpy.so build/bin/saxpy_test

build/lib/libTimer.so: Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc\
		-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
	@mkdir -p build/lib
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(CUDAPATH)/lib64 -lcudart_static
	@mkdir -p build/include
	@ln -sf ../../Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libsaxpy.so: saxpy/src/saxpy.cu
	@mkdir -p build/.objects/saxpy
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-Isaxpy/include -I$(CUDAPATH)/samples/common/inc \
		-dc -o build/.objects/saxpy/saxpy.o \
		saxpy/src/saxpy.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/saxpy/saxpy-dlink.o build/.objects/saxpy/saxpy.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libsaxpy.so build/.objects/saxpy/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../saxpy/include/saxpy.h build/include/saxpy.h

build/bin/saxpy_test: build/lib/libTimer.so build/lib/libsaxpy.so \
	saxpy/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-ITimer/include -Isaxpy/include \
		-o build/bin/saxpy_test saxpy/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lsaxpy -lcudart -lblas -lgfortran

run: build/bin/saxpy_test
	@rm -f *.nsys-rep saxpy_test.i* saxpy_test.o* core.*
	@echo -ne "gpu\n1\n\n10gb\n1\nampere\nsaxpy_test\n" | \
		run_gpu .runTests.sh > /dev/null
	@sleep 5
	@tail -f saxpy_test.o*

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f saxpy_test.*
