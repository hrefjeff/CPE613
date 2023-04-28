CC=gcc
CXX=g++
NVCC=nvcc

SAMPLESPATH=/cuda-samples/Common
# === Local Laptop Settings ===
CUDAPATH=/usr/local/cuda-11.7
# === Local Desktop Settings ===
#CUDAPATH=/usr/local/cuda-12.0

# Remote Alabama ASC
#CUDAPATH=/opt/asn/apps/cuda_11.7.0
#SAMPLESPATH=/samples/common/inc
# ------- On the DMC, run this to avoid having to type in all the parameters
# echo -ne "gpu\n1\n\n10gb\n1\nampere\nconvolution_test\n" | run_gpu build/bin/convolution_test > /dev/null

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

#NVCCARCHS=-gencode arch=compute_75,code=sm_75 # Local Desktop
NVCCARCHS=-gencode arch=compute_86,code=sm_86 # Local Laptop
#NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 # Alabama ASC

INCPATH=-Iconvolution/include -I$(CUDAPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean

all: build/lib/libTimer.so build/lib/libconvolution.so \
	build/lib/libconvo_utils.so build/bin/convolution_time_test \
	#build/bin/convolution_fft_test build/bin/nvidia_example_test

build/lib/libTimer.so: Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(CUDAPATH)/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-o build/.objects/Timer/Timer.so Timer/src/Timer.cpp
	@mkdir -p build/lib
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(CUDAPATH)/lib64 -lcudart_static
	@mkdir -p build/include
	@ln -sf ./Timer/include/Timer.hpp build/include/Timer.hpp

build/lib/libconvo_utils.so: convolution/src/convo_utils.cu
	@mkdir -p build/.objects/convo_utils
	$(NVCC) -pg $(NVCCFLAGS) -Xcompiler -fPIC -Iconvolution/include \
		-I$(CUDAPATH)$(SAMPLESPATH) -dc -o \
		build/.objects/convolution/convo_utils.o \
		convolution/src/convo_utils.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/convolution/convolutionutils-dlink.o \
		build/.objects/convolution/convo_utils.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libconvo_utils.so build/.objects/convolution/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../convolution/include/convo_utils.h build/include/convo_utils.h

build/lib/libconvolution.so: convolution/src/Convolution.cu
	@mkdir -p build/.objects/convolution
	$(NVCC) -pg $(NVCCFLAGS) -Xcompiler -fPIC -Iconvolution/include \
		-I$(CUDAPATH)$(SAMPLESPATH) -dc -o \
		build/.objects/convolution/convolutionclass.o \
		convolution/src/Convolution.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/convolution/convolution-dlink.o \
		build/.objects/convolution/convolutionclass.o
	mkdir -p build/lib
	$(CC) -shared -o build/lib/libconvolutionclass.so build/.objects/convolution/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../convolution/include/Convolution.h build/include/Convolution.h

build/bin/convolution_time_test: build/lib/libTimer.so build/lib/libconvolution.so \
	build/lib/libconvo_utils.so convolution/test/src/conv_1d_time.cu
	@mkdir -p build/bin
	$(NVCC) -Ibuild/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-I$(CUDAPATH)/include -ITimer/include \
		-o build/bin/convolution_time_test convolution/test/src/conv_1d_time.cu \
		-Xlinker=-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lconvolutionclass -lconvo_utils -lcudart -lcufft

build/bin/convolution_fft_test: build/lib/libTimer.so build/lib/libconvolution.so \
	convolution/test/src/test_fft.cu
	@mkdir -p build/bin
	$(NVCC) -Ibuild/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-I$(CUDAPATH)/include -ITimer/include \
		-o build/bin/convolution_fft_test convolution/test/src/test_fft.cu \
		-Xlinker=-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lconvolution -lconvo_utils -lcudart -lcufft

build/bin/nvidia_example_test: build/lib/libTimer.so build/lib/libconvolution.so \
	convolution/test/src/nvidia_example.cu
	@mkdir -p build/bin
	$(NVCC) -Ibuild/include -I$(CUDAPATH)$(SAMPLESPATH) \
		-I$(CUDAPATH)/include -ITimer/include \
		-o build/bin/nvidia_example_test convolution/test/src/nvidia_example.cu \
		-Xlinker=-rpath=$(PWD)/build/lib \
		-Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lTimer -lconvolution -lconvo_utils -lcudart -lcufft

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f convolution_test.*
