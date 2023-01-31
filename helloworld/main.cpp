#include <cstdio>

#include <cuda_runtime.h>
#include <helper_cuda.h>

int main() {

	int deviceIdx = 0;
	checkCudaErrors (
		cudaGetDevice (
			&deviceIdx
		)
	);
	printf (
		"hello, CUDA Device %d!\n",
		deviceIdx
	);

	return 0;
}
