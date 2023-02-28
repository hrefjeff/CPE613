# https://geeksww.com/tutorials/libraries/libpng/installation/installing_libpng_on_ubuntu_linux.php
# Update rpath to where you download it
# https://linuxhint.com/install-opencv-ubuntu/ install opencv

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

# Laptop cuda path
LAPTOPCUDAPATH=/usr/local/cuda-11.7
DESKTOPCUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/cuda-samples/Common
OPENCVPATH=/usr/local/include/opencv4
SHAREDLIBS=/usr/local/lib

all: build/lib/libTimer.so main

build/lib/libTimer.so: Timer/src/Timer.cpp
	@mkdir -p build/.objects/Timer
	$(CXX) $(CXXFLAGS) -c -fPIC -ITimer/include \
		-I$(LAPTOPCUDAPATH)/include -I$(LAPTOPCUDAPATH)$(SAMPLESPATH) \
		-o build/.objects/Timer/Timer.os Timer/src/Timer.cpp
	@mkdir -p build/lib
	$(CXX) -shared -o build/lib/libTimer.so build/.objects/Timer/* \
		-L$(LAPTOPCUDAPATH)/lib64 -lcudart_static
	@mkdir -p build/include
	@ln -sf ./Timer/include/Timer.hpp build/include/Timer.hpp

main: conv.cu build/lib/libTimer.so
	nvcc -std=c++11 -o c2g.o conv.cu \
	-I$(LAPTOPCUDAPATH)$(SAMPLESPATH) -ITimer/include \
	-I$(OPENCVPATH) \
	-L$(SHAREDLIBS) \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
	-Lbuild/lib -lTimer \
	-Xlinker -rpath=$(SHAREDLIBS):$(PWD)/build/lib

basic: basic2dconv.cu
	nvcc -std=c++11 -o basic.o basic2dconv.cu -g -G \
	-I$(DESKTOPCUDAPATH)$(SAMPLESPATH)

advanced: adv2dconv.cu
	nvcc -std=c++11 -o adv.o adv2dconv.cu \
	-I$(LAPTOPCUDAPATH)$(SAMPLESPATH) \
	-I$(OPENCVPATH) \
	-L$(SHAREDLIBS) \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
	-Xlinker -rpath=$(SHAREDLIBS)

example: opencv-example.cpp
	g++ -std=c++11 -o ocv.o opencv-example.cpp \
	-I$(OPENCVPATH) \
	-L$(SHAREDLIBS) \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
	-Xlinker -rpath=$(SHAREDLIBS)

clean:
	rm -rf build jeffc2g* grayboiz.jpg
