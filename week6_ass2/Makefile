# https://geeksww.com/tutorials/libraries/libpng/installation/installing_libpng_on_ubuntu_linux.php
# Update rpath to where you download it
# https://linuxhint.com/install-opencv-ubuntu/ install opencv

# Laptop cuda path
LAPTOPCUDAPATH=/usr/local/cuda-11.7
DESKTOPCUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/cuda-samples/Common
OPENCVPATH=/usr/local/include/opencv4
SHAREDLIBS=/usr/local/lib

main: conv.cu
	nvcc -std=c++11 -o c2g.o conv.cu \
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
	rm -f jeffc2g* grayboiz.jpg
