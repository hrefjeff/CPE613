#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ 
void rgb2gray_kernel(unsigned char* red,unsigned char* green, unsigned char* blue, 
                    unsigned char* gray, int width, int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int pixelIndex = row*width + col;

        // Convert the pixel
        gray[pixelIndex] = red[pixelIndex]*3/10 + green[pixelIndex]*6/10 + blue[pixelIndex]*1/10;
    }
}

int main() {

    // Set our problem size
    const int WIDTH = 810;
    const int HEIGHT = 456;
    const int TOTAL_SIZE = WIDTH * HEIGHT;
    unsigned char *red_h, *green_h, *blue_h, *gray_h;
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    
    // Allocate memory on the host
    cudaMallocHost((void**)&red_h, TOTAL_SIZE);
    cudaMallocHost((void**)&green_h, TOTAL_SIZE);
    cudaMallocHost((void**)&blue_h, TOTAL_SIZE);
    cudaMallocHost((void**)&gray_h, TOTAL_SIZE);

    // Fill the host matrix with data
    FILE *red_file = fopen("reds.txt", "r");
    FILE *green_file = fopen("greens.txt", "r");
    FILE *blue_file = fopen("blues.txt", "r");
    if (red_file == NULL || green_file == NULL || blue_file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fscanf(red_file, "%d", &red_h[i+j]);
            fscanf(green_file, "%d", &green_h[i+j]);
            fscanf(blue_file, "%d", &blue_h[i+j]);
        }
    }

    fclose(red_file);
    fclose(green_file);
    fclose(blue_file);

    // Allocate memory on the device
    cudaMalloc(&red_d, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&green_d, TOTAL_SIZE);
    cudaMalloc(&blue_d, TOTAL_SIZE);
    cudaMalloc(&gray_d, TOTAL_SIZE);

    // Set our block size and threads per thread block
    const int THREADS = 32;

    // Set up kernel launch parameters, so we can create grid/blocks
    dim3 numThreadsPerBlock(THREADS, THREADS);
    dim3 numBlocks( (WIDTH + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
                    (HEIGHT + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);

    // Copy data from host to device
    cudaMemcpy(red_d, red_h, TOTAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green_h, TOTAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue_h, TOTAL_SIZE, cudaMemcpyHostToDevice);

    // Perform CUDA computations on deviceMatrix
    // Launch our kernel
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, HEIGHT, WIDTH);

    // Free memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaFreeHost(red_h);
    cudaFreeHost(green_h);
    cudaFreeHost(blue_h);
    cudaFreeHost(gray_h);

    return 0;
}
