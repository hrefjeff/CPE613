main: main.cpp
	nvcc -std=c++11 -o main main.cpp -Iopencv-install/include -Lopencv-install/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -Xlinker -rpath=`pwd`/opencv-install/lib64
