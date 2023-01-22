#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

__global__ 
void rgb2gray_kernel(unsigned char* red,unsigned char* green, unsigned char* blue, 
                    unsigned char* gray, int width, int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int pixelIndex = (row*width) + col;

        // Convert the pixel
        gray[pixelIndex] = (red[pixelIndex]*3.0/10.0) + (green[pixelIndex]*6.0/10.0) + (blue[pixelIndex]*1.0/10.0);
    }
}

int main() {

    Mat img = imread("thethreeamigos.jpeg", IMREAD_COLOR);
    // imshow("Goat!", img);

    // Set our problem size
    const int WIDTH = 810;
    const int HEIGHT = 456;
    const int TOTAL_SIZE = WIDTH * HEIGHT;

    // Allocate memory in host RAM
    unsigned char *h_red, *h_green, *h_blue, *h_gray;
    cudaMallocHost((void **) &h_red, sizeof(char)*TOTAL_SIZE);
    cudaMallocHost((void **) &h_green, sizeof(char)*TOTAL_SIZE);
    cudaMallocHost((void **) &h_blue, sizeof(char)*TOTAL_SIZE);
    cudaMallocHost((void **) &h_gray, sizeof(char)*TOTAL_SIZE);

    // Fill the host matrices with data
    Mat greyMat(img.rows, img.cols, CV_8UC1, Scalar(0));
    for (int rowIdx = 0; rowIdx < img.rows; ++rowIdx) {
        for (int colIdx = 0; colIdx < img.cols; ++colIdx) {
            auto & vec = img.at<cv::Vec<uchar, 3>>(rowIdx, colIdx);
            h_blue[rowIdx+colIdx] = vec[0]; 
            h_green[rowIdx+colIdx] = vec[1]; 
            h_red[rowIdx+colIdx] = vec[2];
        }
    }

    // Allocate memory space on the device 
    unsigned char *d_red, *d_green, *d_blue, *d_gray;
    cudaMalloc((void **) &d_red, sizeof(char)*TOTAL_SIZE);
    cudaMalloc((void **) &d_green, sizeof(char)*TOTAL_SIZE);
    cudaMalloc((void **) &d_blue, sizeof(char)*TOTAL_SIZE);
    cudaMalloc((void **) &d_gray, sizeof(char)*TOTAL_SIZE);

    // Copy matrices from host to device memory
    cudaMemcpy(d_red, h_red, sizeof(char)*TOTAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, h_green, sizeof(char)*TOTAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, h_blue, sizeof(char)*TOTAL_SIZE, cudaMemcpyHostToDevice);

    // Set our block size and threads per thread block
    const int THREADS = 32;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 numThreadsPerBlock(THREADS, THREADS);
    dim3 numBlocks( (WIDTH + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
                    (HEIGHT + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);

    // Perform CUDA computations on deviceMatrix, Launch Kernel
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(d_red, d_green, d_blue, d_gray, HEIGHT, WIDTH);

    // Copy result from device to host
    cudaMemcpy(d_gray, h_gray, TOTAL_SIZE, cudaMemcpyDeviceToHost);

    // Copy result from gray matrix into matlab OpenCV input array format
    for (int rowIdx = 0; rowIdx < HEIGHT; ++rowIdx) {
    for (int colIdx = 0; colIdx < WIDTH; ++colIdx)
      greyMat.at<uchar>(rowIdx, colIdx) = h_gray[rowIdx + colIdx];
    }

    // Write img to gray.jpg
    imwrite("grayboiz.jpg", greyMat);

    // Free memory
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_gray);
    cudaFreeHost(h_red);
    cudaFreeHost(h_green);
    cudaFreeHost(h_blue);
    cudaFreeHost(h_gray);

    return 0;
}
