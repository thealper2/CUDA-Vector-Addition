#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c)
{
	for (int i=0; i<N; i++)
		c[i] = a[i] + b[i];
}

__global__ void device_add(int *a, int *b, int *c)
{
	// Thread + Block
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];

	// Thread only
	// c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

	// Block only
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void gpu_alloc(void **devPtr, size_t size)
{
	cudaError_t result = cudaMalloc(devPtr, size);
	if (result != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(result));
		exit(1);
	}
}

void gpu_free(void *devPtr)
{
	cudaError_t result = cudaFree(devPtr);
	if (result != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(result));
	}
}

void cpu_to_gpu(void *devPtr, const void *hostPtr, size_t size)
{
	cudaError_t result = cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CPU -> GPU cudaMemcpy failed: %s\n", cudaGetErrorString(result));
	}
}

void gpu_to_cpu(void *hostPtr, const void *devPtr, size_t size)
{
	cudaError_t result = cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		fprintf(stderr, "GPU -> CPU cudaMemcpy failed: %s\n", cudaGetErrorString(result));
	}
}

void fill(int *data)
{
	for (int i=0; i<N; i++)
		data[i] = i;
}

void print(int *a, int *b, int *c)
{
	for (int i=0; i<N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
}

int main()
{
	int *a, *b, *c;
	int *d_a = NULL, *d_b = NULL, *d_c = NULL;
	int allocation_size = N * sizeof(int);

	a = (int *)malloc(allocation_size);
	fill(a);

	b = (int *)malloc(allocation_size);
	fill(b);

	c = (int *)malloc(allocation_size);

	gpu_alloc((void **)&d_a, allocation_size);
	gpu_alloc((void **)&d_b, allocation_size);
	gpu_alloc((void **)&d_c, allocation_size);

	cpu_to_gpu(d_a, a, allocation_size);
	cpu_to_gpu(d_b, b, allocation_size);

	int threads_per_block = 128;
	int no_of_blocks = (N + threads_per_block - 1) / threads_per_block;

	device_add <<< no_of_blocks, threads_per_block >>> (d_a, d_b, d_c);

	gpu_to_cpu(c, d_c, allocation_size);

	print(a, b, c);

	free(a);
	free(b);
	free(c);

	gpu_free(d_a);
	gpu_free(d_b);
	gpu_free(d_c);

	return 0;
}

