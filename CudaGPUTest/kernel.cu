
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdio>
#include <ctime>

cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size);

__global__ void addKernel(int n, int *c, int *a, int *b)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i += stride)
		c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 1 << 22;
	int *a = new int[arraySize];
	int *b = new int[arraySize];
	int *c = new int[arraySize];

	for (int i = 0; i < arraySize; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	// Variables to keep track of, to only count adding
	std::clock_t start;
	double duration;
	start = std::clock();

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{%d,%d,%d,%d,%d}\n", c[50], c[51], c[20000], c[322], c[434]);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC * 1000.0;

	fprintf(stderr, "Time: %d ms", duration);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, 256>>>(size, dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
