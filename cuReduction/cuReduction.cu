/*
 * cuda-exercises / cuAddVector
 *
 * (c) 2013 Alberto Fernández <albertofem@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Desarrollado por Sergio Orts-Escolano
 * Copyright Universidad de Alicante, 2012
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// assuming one CUDA device
#define DEVICE 0

__global__ void cuReductionSimple(float * d_out, float * d_in)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for(int s=1; s<blockDim.x; s<<=1)
	{
		if(tid % (2*s) == 0)
			d_in[tid] += d_in[tid+s];

		__syncthreads();
	}

	if(threadIdx.x == 0) d_out[blockIdx.x] = d_in[tid];
}

__global__ void cuReductionAdvanced(float * d_out, float * d_in)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	for(int s=blockDim.x/2; s>0; s>>=1)
	{
		if(threadIdx.x < s)
			d_in[tid] += d_in[tid+s];

		__syncthreads();
	}

	if(threadIdx.x == 0) d_out[blockIdx.x] = d_in[tid];
}

__global__ void cuReductionSharedMemory(float * d_out, const float * d_in)
{
	extern __shared__ float d_in_sm[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * blockDim.x;

	d_in_sm[tid] = d_in[i];

	__syncthreads();

	for(int s=blockDim.x/2; s>0; s>>=1)
	{
		if(tid < s)
			d_in_sm[tid] += d_in_sm[tid+s];

		__syncthreads();
	}

	if(tid == 0) d_out[blockIdx.x] = d_in_sm[tid];
}

void reduce(float * d_out, float * d_intermediate, float * d_in,
			int size, bool usesSharedMemory)
{
	// assumes that size is not greater than maxThreadsPerBlock^2
	// and that size is a multiple of maxThreadsPerBlock
	const int maxThreadsPerBlock = 512;

	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;

	if (usesSharedMemory)
	{
		int sharedMemory = threads * sizeof(float);

		cuReductionSharedMemory<<< blocks, threads, sharedMemory >>>
			(d_intermediate, d_in);
	}
	else
	{
		cuReductionAdvanced<<< blocks, threads >>>
			(d_intermediate, d_in);
	}

	// now we're down to one block left, so reduce it
	threads = blocks; // launch one thread for each block in prev step
	blocks = 1;

	if (usesSharedMemory)
	{
		int sharedMemory = threads * sizeof(float);

		cuReductionSharedMemory<<< blocks, threads, sharedMemory >>>
			(d_intermediate, d_in);
	}
	else
	{
		cuReductionSimple<<< blocks, threads >>>
			(d_out, d_intermediate);
	}
}

int main(int argc, char **argv)
{
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		fprintf(stderr, "Fatal error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(DEVICE);

	cudaDeviceProp devProps;

	if(cudaGetDeviceProperties(&devProps, DEVICE) == 0)
	{
		printf("Using device %d:\n", DEVICE);
		printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (int)devProps.totalGlobalMem,
			(int)devProps.major, (int)devProps.minor,
			(int)devProps.clockRate);
	}

	const int ARRAY_SIZE = 1 << 18;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float *h_in = new float[ARRAY_SIZE];
	float sum = 0.0f;

	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		// generate random float in [-1.0f, 1.0f]
		h_in[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
		sum += h_in[i];
	}

	// declare GPU memory pointers
	float * d_in, * d_intermediate, * d_out;

	// allocate GPU memory
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_intermediate, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, sizeof(float));

	// transfer the input array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Pasar por el primer parámetro versíon reduction ( 0: memoria global, 1: memoria compartida)
	int whichKernel = 0;

	if (argc == 2)
		whichKernel = atoi(argv[1]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// launch the kernel
	switch(whichKernel)
	{
		case 0:
			printf("Running global reduce\n");

			cudaEventRecord(start, 0);
				reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
			cudaEventRecord(stop, 0);

			break;
		case 1:
			printf("Running reduce with shared mem\n");

			cudaEventRecord(start, 0);
				reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
			cudaEventRecord(stop, 0);

			break;
		default:
			fprintf(stderr, "Fatal error: no kernel executed\n");
			exit(EXIT_FAILURE);
	}

	cudaEventSynchronize(stop);

	float elapsedTime;

	cudaEventElapsedTime(&elapsedTime, start, stop);

	// copy back the sum from GPU
	float h_out;

	cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	printf("average time elapsed: %f\n", elapsedTime);

	if(abs(sum-h_out) < 0.001)
		printf("PASSED \n");
	else
		printf("FAILED \n");

	// free GPU memory allocation
	cudaFree(d_in);
	cudaFree(d_intermediate);
	cudaFree(d_out);

#if _DEBUG
	cudaDeviceReset();
#endif

	return 0;
}