main: color2gray.cu
	nvcc -std=c++11 -o jeffc2g color2gray.cu \
	-I/opt/asn/apps/cuda_11.7.0/samples/common/inc \
	-I/home/uahclsc0002/wise_class/opencv-install/include \
	-L/home/uahclsc0002/wise_class/opencv-install/lib64 \
	-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
	-Xlinker -rpath=/home/uahclsc0002/wise_class/opencv-install/lib64:/opt/asn/apps/intel_2022.2.0/mkl/2022.1.0/lib/intel64

clean:
	rm -f jeffc2g* grayboiz.jpg
