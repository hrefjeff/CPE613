main: color2gray.cu
	nvcc -std=c++11 -o jeffc2g color2gray.cu -Iopencv_example/opencv-install/include -Lopencv_example/opencv-install/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -Xlinker -rpath=`pwd`/opencv_example/opencv-install/lib64

clean:
	rm -f jeffc2g*
