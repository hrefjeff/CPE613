#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>

using namespace cv;
using namespace std;

__global__ 
void rgb2gray_kernel (
    unsigned char* red_d,
    unsigned char* green_d,
    unsigned char* blue_d,
    unsigned char* gray_d,
    int width,
    int height
) {

    for (
        int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
        rowIdx < height;
        rowIdx += blockDim.y * gridDim.y
    ) {
        for (
            int colIdx = threadIdx.y + blockIdx.y * blockDim.y;
            colIdx < width;
            colIdx += blockDim.x * gridDim.x
        ) {
            int offset = rowIdx * width + colIdx;
            gray_d[offset] = (unsigned char)(
                (float)red_d[offset] * 3.0 / 10.0 +
                (float)green_d[offset] * 6.0 / 10.0 +
                (float)blue_d[offset] * 1.0 / 10.0
            );
        }
    }

}

void device_rgb2grayscale (
    unsigned char * deviceRed,
    unsigned char * deviceGreen,
    unsigned char * deviceBlue,
    unsigned char * deviceGray,
    int numRows,
    int numCols
){
    int blockWidth = 16;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 numThreadsPerBlock(blockWidth, blockWidth);
    dim3 numBlocks(
        (numCols + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
        (numRows + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y
    );

    // Perform CUDA computations on deviceMatrix, Launch Kernel
    rgb2gray_kernel<<<
        numBlocks,
        numThreadsPerBlock
    >>>(
        deviceRed,
        deviceGreen,
        deviceBlue,
        deviceGray,
        numCols,
        numRows
    );

    checkCudaErrors(
        cudaGetLastError()
    );
}


int main() {

    Mat img = imread("thethreeamigos.jpeg", IMREAD_COLOR);
    //imshow("Goat!", img);

    // Allocate memory in host RAM
    std::vector<unsigned char> hostRed(img.rows * img.cols);
    std::vector<unsigned char> hostGreen(img.rows * img.cols);
    std::vector<unsigned char> hostBlue(img.rows * img.cols);
    std::vector<unsigned char> hostGray(img.rows * img.cols);

    // Fill the host matrices with data
    Mat greyMat(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
        for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
            auto & vec = img.at<cv::Vec<uchar, 3>>(rowIdx, colIdx);
            int offset = rowIdx * img.cols + colIdx;
            hostRed[offset] = vec[2];
            hostGreen[offset] = vec[1];
            hostBlue[offset] = vec[0];
        }
    }

    // Allocate memory space on the device
    unsigned char * deviceRed = nullptr;
    unsigned char * deviceGreen = nullptr;
    unsigned char * deviceBlue = nullptr;
    unsigned char * deviceGray = nullptr;
    size_t byteSize = HEIGHT * WIDTH * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&deviceRed,byteSize));
    checkCudaErrors(cudaMalloc(&deviceGreen,byteSize));
    checkCudaErrors(cudaMalloc(&deviceBlue,byteSize));
    checkCudaErrors(cudaMalloc(&deviceGray,byteSize));

    // Upload data to device
    checkCudaErrors(
        cudaMemcpy (deviceRed, hostRed.data(),byteSize,cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (deviceGreen, hostGreen.data(),byteSize,cudaMemcpyHostToDevice)
    );
    checkCudaErrors(
        cudaMemcpy (deviceBlue, hostBlue.data(),byteSize,cudaMemcpyHostToDevice)
    );

    device_rgb2grayscale (
        deviceRed,
        deviceGreen,
        deviceBlue,
        deviceGray,
        HEIGHT,
        WIDTH
    );

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpy (hostGray.data(), deviceGray, byteSize, cudaMemcpyDeviceToHost)
    );

    // Copy result from gray matrix into matlab OpenCV input array format
    for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
        for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
            int offset = rowIdx * img.cols + colIdx;
            greyMat.at<uchar>(rowIdx, colIdx) = hostGray[offset];
        }
    }

    // Write img to gray.jpg
    imwrite("grayboiz.jpg", greyMat);

    // Free memory
    cudaFree(deviceRed);
    cudaFree(deviceGreen);
    cudaFree(deviceBlue);
    cudaFree(deviceGray);

    printf("Made it to the end!\n");

    return 0;
}
