#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ void matrixMult(int* a, int* b, int* c, int N) {

    // calculate global row and col for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int tmp = 0;
        for (int i=0; i<N; i++) {
            tmp += a[row * N + i] * b[i * N + col];
        }

        //write back the result
        c[row * N + col] = tmp;
    }
}

// Initialize square matrix with random numbers 1-100
void init_matrix(int *matrix, int N) {
    for(int i=0; i<N * N; i++) {
        matrix[i] = rand() % 100;
    }
}

void verify_result(int* a, int* b, int* c, int N) {
    int tmp;

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            tmp = 0;
            for(int k=0; k<N; k++) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            // Assert 
            assert(tmp == c[i * N * j]);
        }
    }
    
}

int main() {

    // Set our problem size
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    // Allocate memory for our matrices
    int *a, *b, *c;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);

    init_matrix(a, N);
    init_matrix(b, N);

    // Set our block size and threads per thread block
    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    // Set up kernel launch parameters, so we can create 3d grid
    dim3 dimBlocks(threads, threads); // same as THREADS.x = 16
    dim3 dimGrid(blocks, blocks);

    // Launch our kernel
    matrixMult<<<dimGrid, dimBlocks>>>(a, b, c, N);
    cudaDeviceSynchronize(); // cudaMemcpy

    // Verify result by computing on CPU and comparing it to GPU
    verify_result(a, b, c, N);

    cout << "Program completed successfully." << endl;

    return 0;
}
