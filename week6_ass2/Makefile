# https://geeksww.com/tutorials/libraries/libpng/installation/installing_libpng_on_ubuntu_linux.php
# Update rpath to where you download it

CUDAPATH=/usr/local/cuda-12.0
SAMPLESPATH=/cuda-samples/Common
OPENCVPATH=/usr/local/include/opencv4
SHAREDLIBS=/usr/local/lib

main: color2gray.cu
	nvcc -std=c++11 -o c2g.o color2gray.cu \
	-I$(CUDAPATH)$(SAMPLESPATH) \
	-I$(OPENCVPATH) \
	-L$(SHAREDLIBS) \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
	-Xlinker -rpath=$(SHAREDLIBS)

clean:
	rm -f jeffc2g* grayboiz.jpg
