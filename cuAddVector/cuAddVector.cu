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

#include <stdio.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// number of vector elements to sum
#define VECTOR_SIZE 80000000

// assuming only one CUDA device
#define DEVICE 0

// default threads per block
#define THREADS_PER_BLOCK 1024

// forward declaration
cudaError_t allocateCudaBytes(void*, size_t, char*);
cudaError_t copyCudaBytes(void*, void*, size_t, char*, cudaMemcpyKind);

// kernel
__global__ void cuAddVector(int *vectorA, int *vectorB, int *vectorC, int totalBlocks)
{
	int threadAbsoluteId = threadIdx.x + blockDim.x * blockIdx.x;

	while(threadAbsoluteId < VECTOR_SIZE)
	{
		vectorC[threadAbsoluteId] = vectorA[threadAbsoluteId] + vectorC[threadAbsoluteId];
		threadAbsoluteId = threadAbsoluteId+totalBlocks*blockDim.x;
	}
}

int main(void)
{
	int *vectorA, *vectorB, *vectorC;
	int *deviceVectorA, *deviceVectorB, *deviceVectorC;

	// get device info to avoid problems with block / threads configuration
	cudaSetDevice(DEVICE);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DEVICE);

	int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	int threadsPerBlock = THREADS_PER_BLOCK;

	if(threadsPerBlock > maxThreadsPerBlock)
	{
		fprintf(stderr, "Maximun num of threads per block (%d) overflowed, using max!\n", maxThreadsPerBlock);
		threadsPerBlock = maxThreadsPerBlock;
	}

	int totalBlocks = (VECTOR_SIZE + (threadsPerBlock-1))/(threadsPerBlock);
	int maxBlocks = deviceProp.maxGridSize[0];

	if(totalBlocks > deviceProp.maxGridSize[0])
	{
		fprintf(stderr, "Maximun number of blocks (%d) reached overflowed, using max!\n", maxBlocks);
		totalBlocks = maxBlocks;
	}

	int memorySize = VECTOR_SIZE * sizeof(int);

	printf("Total blocks: %d\n", totalBlocks);
	printf("Threads per block: %d\n", threadsPerBlock);

	vectorA = (int*)malloc(memorySize);
	vectorB = (int*)malloc(memorySize);
	vectorC = (int*)malloc(memorySize);

	if(allocateCudaBytes(&deviceVectorA, memorySize, "deviceVectorA") != 0)
		return -1;

	if(allocateCudaBytes(&deviceVectorB, memorySize, "deviceVectorB") != 0)
		return -1;

	if(allocateCudaBytes(&deviceVectorC, memorySize, "deviceVectorC") != 0)
		return -1;

	for (int i=0; i<VECTOR_SIZE; i++) {
		vectorA[i] = i;
		vectorB[i] = 2 * i;
	}

	if(copyCudaBytes(deviceVectorA, vectorA, memorySize, "deviceVectorA", cudaMemcpyHostToDevice) != 0)
		return -1;

	if(copyCudaBytes(deviceVectorB, vectorB, memorySize, "deviceVectorB", cudaMemcpyHostToDevice) != 0)
		return -1;

	cuAddVector<<< totalBlocks, threadsPerBlock >>>
		(deviceVectorA, deviceVectorB, deviceVectorC, totalBlocks);

	cudaError_t cudaStatus;

	cudaStatus = cudaDeviceSynchronize();

	if (cudaStatus != 0) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		return -1;
	}

	if(copyCudaBytes(vectorC, deviceVectorC, memorySize, "deviceVectorA", cudaMemcpyDeviceToHost) != 0)
		return -1;

	bool success = true;

	for (int i=0; i<VECTOR_SIZE; i++) {
		if ((vectorA[i] + vectorB[i]) != vectorC[i]) {
			printf("Error:  %d + %d != %d\n", vectorA[i], vectorB[i], vectorC[i]);
			success = false;
		}
	}

	if (success) 
		printf("Success adding numbers in CUDA!\n");

	cudaFree(deviceVectorA);
	cudaFree(deviceVectorB);
	cudaFree(deviceVectorC);

	free(deviceVectorA);
	free(deviceVectorB);
	free(deviceVectorC);

#ifdef _DEBUG
	cudaDeviceReset();
#endif

	return 0;
}

// wrapper for cudaMalloc
cudaError_t allocateCudaBytes(void* element, size_t size, char* element_name)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)element, size);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to allocate %d bytes on '%s' on device with cudaStatus: %d!\n", size, element_name, cudaStatus);
	} else {
		printf("Allocated %d bytes on '%s' with cudaStatus: %d\n", size, element_name, cudaStatus);
	}

	return cudaStatus;
}

// wrapper for cudaMemcpy
cudaError_t copyCudaBytes(void* destiny, void* source, size_t size, char* element_name, cudaMemcpyKind mode)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(destiny, source, size, mode);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to copy %d bytes on '%s' to device with cudaStatus: %d\n", size, element_name, cudaStatus);
		return cudaStatus;
	} else {
		printf("Copied %d bytes to device on '%s' with cudaStatus: %d\n", size, element_name, cudaStatus);
	}

	return cudaStatus;
}