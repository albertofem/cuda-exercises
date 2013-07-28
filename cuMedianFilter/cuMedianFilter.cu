/*
 * cuda-exercises / cuMedianFilter
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
#include <stdlib.h>
#include <math.h>

#if _WIN32
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/time.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "EasyBMP.h"

#if _WIN32
typedef LARGE_INTEGER timeStamp;
void getCurrentTimeStamp(timeStamp& _time);
timeStamp getCurrentTimeStamp();
double getTimeMili(const timeStamp& start, const timeStamp& end);
double getTimeSecs(const timeStamp& start, const timeStamp& end);
#endif

double getCurrentTime();
void checkCUDAError();

// image size in pixels
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

// filter FILTER_ITERATIONS
#define FILTER_ITERATIONS 10

// cuda grid and block size 
#define CUDA_GRID_W  64
#define CUDA_GRID_H  64
#define CUDA_BLOCK_W 16
#define CUDA_BLOCK_H 16

unsigned char hostInput[IMAGE_HEIGHT+2][IMAGE_WIDTH+2];
unsigned char gpuOutput[IMAGE_HEIGHT+2][IMAGE_WIDTH+2];
unsigned char hostOutput[IMAGE_HEIGHT+2][IMAGE_WIDTH+2];

// median filter kernel in 1 dimension, processing rows
__global__ void cuMedianFilter1DRow(unsigned char *d_output, unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = IMAGE_WIDTH + 2;

	// calculate current row
	row = blockIdx.x * blockDim.x + threadIdx.x + 1;

	
	for (col=1; col<=IMAGE_WIDTH; col++)
	{
		unsigned char neighborhood[9];

		idx = row * numcols + col;
		idx_south = (row - 1) * numcols + col;
		idx_north = (row + 1) * numcols + col;

		idx_west = row * numcols + (col - 1);
		idx_east = row * numcols + (col + 1);

		idx_north_east = (row + 1) * numcols + (col + 1);
		idx_north_west = (row + 1) * numcols + (col - 1);
		idx_south_east = (row - 1) * numcols + (col + 1);
		idx_south_west = (row - 1) * numcols + (col - 1);

		neighborhood[0]= d_input[ idx_south_west ];
		neighborhood[1]= d_input[ idx_south ];
		neighborhood[2]= d_input[ idx_south_east ];
		neighborhood[3]= d_input[ idx_west ];
		neighborhood[4]= d_input[ idx ];
		neighborhood[5]= d_input[ idx_east ];
		neighborhood[6]= d_input[ idx_north_west ];
		neighborhood[7]= d_input[ idx_north ];
		neighborhood[8]= d_input[ idx_north_east ];

		for (unsigned int j=0; j<5; ++j)
		{
			int min = j;

			for (unsigned int i=j+1; i<9; ++i)
				if (neighborhood[i] < neighborhood[min])
					min = i;

			temp = neighborhood[j];
			neighborhood[j] = neighborhood[min];
			neighborhood[min] = temp;
		}

		d_output[idx]=neighborhood[4];
	}
}

__global__ void cuMedianFilter1DCol(unsigned char *d_output, unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numrows = IMAGE_HEIGHT + 2 ;

	// calculate current column
	col = blockIdx.x * blockDim.x + threadIdx.x + 1;

	for (row=1;row<=IMAGE_HEIGHT;row++)
	{
		unsigned char neighborhood[9];

		idx = col * numrows + row;
		idx_south = (col + 1) * numrows + row;
		idx_north = (col - 1) * numrows + row;

		idx_west = col * numrows + (row + 1);
		idx_east = col * numrows + (row - 1);

		idx_north_east = (col - 1) * numrows + (row - 1);
		idx_north_west = (col - 1) * numrows + (row + 1);
		idx_south_east = (col + 1) * numrows + (row - 1);
		idx_south_west = (col + 1) * numrows + (row + 1);

		neighborhood[0] = d_input[idx_south_west];
		neighborhood[1] = d_input[idx_south];
		neighborhood[2] = d_input[idx_south_east];
		neighborhood[3] = d_input[idx_west];
		neighborhood[4] = d_input[idx];
		neighborhood[5] = d_input[idx_east];
		neighborhood[6] = d_input[idx_north_west];
		neighborhood[7] = d_input[idx_north];
		neighborhood[8] = d_input[idx_north_east];

		for (unsigned int j=0; j<5; ++j)
		{
			int min=j;

			for (unsigned int i=j+1; i<9; ++i)
				if (neighborhood[i] < neighborhood[min])
					min = i;

			temp = neighborhood[j];
			neighborhood[j] = neighborhood[min];
			neighborhood[min] = temp;
		}

		d_output[idx] = neighborhood[4];
	}
}

__global__ void cuMedianFilter2D(unsigned char *d_output, unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = IMAGE_WIDTH + 2;

	col = blockIdx.y * blockDim.y + threadIdx.y + 1;
	row = blockIdx.x * blockDim.x + threadIdx.x + 1;

	unsigned char neighborhood[9];

	idx = row * numcols + col;
	idx_south = (row - 1) * numcols + col;
	idx_north = (row + 1) * numcols + col;

	idx_west = row * numcols + (col - 1);
	idx_east = row * numcols + (col + 1);

	idx_north_east = (row + 1) * numcols + (col + 1);
	idx_north_west = (row + 1) * numcols + (col - 1);
	idx_south_east = (row - 1) * numcols + (col + 1);
	idx_south_west = (row - 1) * numcols + (col - 1);

	neighborhood[0] = d_input[idx_south_west];
	neighborhood[1] = d_input[idx_south];
	neighborhood[2] = d_input[idx_south_east];
	neighborhood[3] = d_input[idx_west];
	neighborhood[4] = d_input[idx];
	neighborhood[5] = d_input[idx_east];
	neighborhood[6] = d_input[idx_north_west];
	neighborhood[7] = d_input[idx_north];
	neighborhood[8] = d_input[idx_north_east];

	for (unsigned int j=0; j<5; ++j)
	{
		int min = j;

		for (unsigned int i=j+1; i<9; ++i)
			if (neighborhood[i] < neighborhood[min])
				min = i;

		temp = neighborhood[j];
		neighborhood[j] = neighborhood[min];
		neighborhood[min] = temp;
	}

	d_output[idx] = neighborhood[4];
}

__global__ void cuMedianFilter2DSharedMemory(unsigned char *d_output, unsigned char *d_input)
{
	int col, row, col_sm, row_sm;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;

	int numcols = IMAGE_WIDTH + 2;
	int numcols_sm = CUDA_BLOCK_W + 2;

	unsigned char temp;

	__shared__ unsigned char d_input_sm[(CUDA_BLOCK_H+2)*(CUDA_BLOCK_W+2)];

	col = blockIdx.x * (blockDim.x-2) + threadIdx.x;
	row = blockIdx.y * (blockDim.y-2) + threadIdx.y;

	col_sm = threadIdx.x;
	row_sm = threadIdx.y;

	d_input_sm[row_sm*numcols_sm+col_sm] = d_input[row*numcols+col];

	__syncthreads();

	if (row_sm > 0 && row_sm <= CUDA_BLOCK_H &&
		col_sm > 0 && col_sm <= CUDA_BLOCK_W )
	{
		unsigned char neighborhood[9];

		idx = row * numcols_sm + col_sm;
		idx_south = (row_sm - 1) * numcols_sm + col_sm;
		idx_north = (row_sm + 1) * numcols_sm + col_sm;

		idx_west = row_sm * numcols_sm + (col_sm - 1);
		idx_east = row_sm * numcols_sm + (col_sm + 1);

		idx_north_east = (row_sm + 1) * numcols_sm + (col_sm + 1);
		idx_north_west = (row_sm + 1) * numcols_sm + (col_sm - 1);
		idx_south_east = (row_sm - 1) * numcols_sm + (col_sm + 1);
		idx_south_west = (row_sm - 1) * numcols_sm + (col_sm - 1);

		neighborhood[0] = d_input_sm[idx_south_west];
		neighborhood[1] = d_input_sm[idx_south];
		neighborhood[2] = d_input_sm[idx_south_east];
		neighborhood[3] = d_input_sm[idx_west];
		neighborhood[4] = d_input_sm[idx];
		neighborhood[5] = d_input_sm[idx_east];
		neighborhood[6] = d_input_sm[idx_north_west];
		neighborhood[7] = d_input_sm[idx_north];
		neighborhood[8] = d_input_sm[idx_north_east];

		for (unsigned int j=0; j<5; ++j)
		{
			int min = j;

			for (unsigned int i=j+1; i<9; ++i)
				if (neighborhood[i] < neighborhood[min])
					min = i;

			temp = neighborhood[j];

			neighborhood[j] = neighborhood[min];
			neighborhood[min] = temp;
		}

		d_output[idx] = neighborhood[4];
	}
}

void cuMedianFilterCPU()
{
	unsigned char temp;

	unsigned char neighborhood[9];

	for(unsigned int i=0; i<FILTER_ITERATIONS; i++)
	{
		for(unsigned int y=0; y<IMAGE_HEIGHT; y++)
		{
			for(unsigned int x=0; x<IMAGE_WIDTH; x++)
			{
				neighborhood[0] = hostInput[y][x];
				neighborhood[1] = hostInput[y][x+1];
				neighborhood[2] = hostInput[y][x+2];
				neighborhood[3] = hostInput[y+1][x];
				neighborhood[4] = hostInput[y+1][x+1];
				neighborhood[5] = hostInput[y+1][x+2];
				neighborhood[6] = hostInput[y+2][x];
				neighborhood[7] = hostInput[y+2][x+1];
				neighborhood[8] = hostInput[y+2][x+2];

				unsigned int j = 0;

				for (j=0; j<5; ++j)
				{
					int mini=j;

					for (int l=j+1; l<9; ++l){
						if (neighborhood[l] < neighborhood[mini])
							mini = l;
					}

					temp = neighborhood[j];
					neighborhood[j] = neighborhood[mini];
					neighborhood[mini] = temp;
				}

				hostOutput[y+1][x+1]=neighborhood[4];
			}
		}

		for (unsigned int y = 0; y<IMAGE_HEIGHT; y++) {
			for (unsigned int x = 0; x<IMAGE_WIDTH; x++) {
				hostInput[y+1][x+1] = hostOutput[y+1][x+1];
			}
		}
	}
}

int main(int argc, char *argv[])
{
	int x, y;
	int i;
	int errors;

	int option = atoi(argv[0]);

	double start_time_inc_data, end_time_inc_data;
	double cpu_start_time, cpu_end_time;

	unsigned char *d_input, *d_output, *tmp;

	unsigned char *input_image;
	unsigned char *output_image;

	input_image = (unsigned char*)calloc(((IMAGE_HEIGHT * IMAGE_WIDTH) * 1), sizeof(unsigned char));

	BMP Image;
	Image.ReadFromFile("lena_1024_noise.bmp");

	for(int i=0; i<Image.TellHeight(); i++)
	{
		for(int j=0; j<Image.TellWidth(); j++)
		{
			input_image[i*IMAGE_WIDTH+j] = Image(i,j)->Red;
		}
	}

	size_t memSize = (IMAGE_WIDTH+2) * (IMAGE_HEIGHT+2) * sizeof(unsigned char);

	printf("Grid size: %dx%d\n", CUDA_GRID_W, CUDA_GRID_H);
	printf("Block size: %dx%d\n", CUDA_BLOCK_W, CUDA_BLOCK_H);

	cudaMalloc(&d_input, memSize);
	cudaMalloc(&d_output, memSize);

	for (y=0; y<IMAGE_HEIGHT+2; y++) {
		for (x=0; x<IMAGE_WIDTH+2; x++) {
			hostInput[y][x] = 0;
		}
	}

	for (y=0; y<IMAGE_HEIGHT; y++) {
		for (x=0; x<IMAGE_WIDTH; x++) {
			hostInput[y+1][x+1] = input_image[y*IMAGE_WIDTH+x];
		}
	}

	start_time_inc_data = getCurrentTime();
	cudaMemcpy( d_input, hostInput, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy( d_output, hostInput, memSize, cudaMemcpyHostToDevice);

	dim3 blocksPerGrid;
	dim3 threadsPerBlock;

	for (i=0; i<FILTER_ITERATIONS; i++)
	{
		switch(option)
		{
			case 0:
				blocksPerGrid.x = blocksPerGrid.y = 
				threadsPerBlock.x = threadsPerBlock.y = 
					CUDA_GRID_H;

				cuMedianFilter1DCol<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input);
			break;

			case 1:
				blocksPerGrid.x = blocksPerGrid.y = 
				threadsPerBlock.x = threadsPerBlock.y = 
					CUDA_GRID_W;

				cuMedianFilter1DRow<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input);
			break;

			case 2:
				blocksPerGrid.x = CUDA_GRID_W;
				blocksPerGrid.y = CUDA_GRID_H;

				threadsPerBlock.x = CUDA_BLOCK_W;
				threadsPerBlock.y = CUDA_BLOCK_H;

				cuMedianFilter2D<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input);
			break;

			case 3:
				blocksPerGrid.x = CUDA_GRID_W;
				blocksPerGrid.y = CUDA_GRID_H;

				threadsPerBlock.x = CUDA_BLOCK_W + 2;
				threadsPerBlock.y = CUDA_BLOCK_H + 2;

				cuMedianFilter2DSharedMemory<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input);
			break;
		}

		cudaThreadSynchronize();

		tmp = d_input;
		d_input = d_output;
		d_output = tmp;
	}

	cudaMemcpy(gpuOutput, d_input, memSize, cudaMemcpyDeviceToHost);
	end_time_inc_data = getCurrentTime();

	checkCUDAError();

	cpu_start_time = getCurrentTime();

	cuMedianFilterCPU();

	cpu_end_time = getCurrentTime();

	errors = 0;
	for (y=0; y<IMAGE_HEIGHT; y++)
	{
		for (x=0; x<IMAGE_WIDTH; x++)
		{
			if ( hostInput[y+1][x+1] != gpuOutput[y+1][x+1])
			{
				errors++;

				printf("Error en %d,%d (CPU=%i, GPU=%i)\n", x, y, \
					hostOutput[y+1][x+1], gpuOutput[y+1][x+1]);
			}
		}
	}

	if (errors == 0) 
		printf("\n\n *** TEST PASSED *** \n\n\n");

	output_image = (unsigned char*)calloc(((IMAGE_WIDTH*IMAGE_HEIGHT) * 1), sizeof(unsigned char));

	for (y=0; y<IMAGE_HEIGHT; y++)
	{
		for (x=0; x<IMAGE_WIDTH; x++)
		{
			output_image[y*IMAGE_WIDTH+x] = gpuOutput[y+1][x+1];
		}
	}

	cudaFree(d_input);
	cudaFree(d_output);

	printf("Tiempo ejecución GPU (Incluyendo transferencia de datos): %fs\n", \
		end_time_inc_data - start_time_inc_data);

	printf("Tiempo de ejecución en la CPU                          : %fs\n", \
		cpu_end_time - cpu_start_time);

	for(int i=0; i<Image.TellHeight(); i++)
	{
		for(int j=0; j<Image.TellWidth(); j++)
		{
			Image(i,j)->Red = output_image[i*IMAGE_WIDTH+j];
			Image(i,j)->Green = output_image[i*IMAGE_WIDTH+j];
			Image(i,j)->Blue = output_image[i*IMAGE_WIDTH+j];
		}
	}

	Image.WriteToFile("lena_1024_median.bmp");

#if _DEBUG
	cudaDeviceReset();
#endif

	return 0;
}

/* Funciones auxiliares */

#if _WIN32
void getCurrentTimeStamp(timeStamp& _time)
{
	QueryPerformanceCounter(&_time);
}

timeStamp getCurrentTimeStamp()
{
	timeStamp tmp;
	QueryPerformanceCounter(&tmp);
	return tmp;
}

double getTimeMili()
{
	timeStamp start;
	timeStamp dwFreq;

	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);

	return double(start.QuadPart) / double(dwFreq.QuadPart);
}
#endif

double getCurrentTime()
{
#if _WIN32
	return getTimeMili();
#else
	static int start = 0, startu = 0;
	struct timeval tval;
	double result;

	if (gettimeofday(&tval, NULL) == -1)
		result = -1.0;
	else if(!start) {
		start = tval.tv_sec;
		startu = tval.tv_usec;
		result = 0.0;
	}
	else
		result = (double) (tval.tv_sec - start) + 1.0e-6*(tval.tv_usec - startu);

	return result;
#endif
}

void checkCUDAError()
{
	cudaError_t err = cudaGetLastError();

	if(cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}