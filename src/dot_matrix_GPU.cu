/**************************************************************
 *
 * --== Simple CUDA kernel ==--
 * author: ampereira
 *
 *
 * Fill the rest of the code
 *
 * Insert the functions for time measurement in the correct
 * sections (i.e. do not account for filling the vectors with random data)
 *
 * Before compile choose the CPU/CUDA version by running the bash command:
 *     export CUDA=yes    or    export CUDA=no
 *
 * The stencil array size must be set to the SIZE #define, i.e., float stencil[SIZE];
 **************************************************************/
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include "more/matrix.h"

#define TIME_RESOLUTION 1000000	// time measuring resolution (us)

#define SIZE 1024
#define SIZE_MATRIX SIZE*SIZE
#define NUM_BLOCKS SIZE
#define NUM_THREADS_PER_BLOCK SIZE
#define NUM_THREADS NUM_BLOCKS*NUM_THREADS_PER_BLOCK

void printfMatrix(float *c){
	for(int i=0; i<SIZE; i++){
		for(int j=0; j<SIZE; j++){
			printf("c[i][j] = %f; ", c[i*SIZE+j]);
		}
		printf("\n");
	}
}

using namespace std;

long long unsigned cpu_time;
cudaEvent_t start, stop;
struct timeval t;

void startTime (void) {
	gettimeofday(&t, NULL);
	cpu_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;
}

void stopTime (void) {
	gettimeofday(&t, NULL);
	long long unsigned final_time = t.tv_sec * TIME_RESOLUTION + t.tv_usec;

	final_time -= cpu_time;

	cout << final_time << " us have elapsed" << endl;
}

// These are specific to measure the execution of only the kernel execution - might be useful
void startKernelTime (void) {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
}

void stopKernelTime (void) {
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << milliseconds << " ms have elapsed for the kernel execution" << endl;
}

void checkCUDAError (const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		cerr << "Cuda error: " << msg << ", " << cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

// Fill the input parameters and kernel qualifier
__global__ void stencilKernel (const float *_a,const float *_b, float *_c) {
	int i = blockIdx.x*SIZE;
	int j = threadIdx.x;
	int j_b = threadIdx.x*SIZE;

	float temp=0;
	__shared__ float As[SIZE];

    As[j] = _a[i+j];

    __syncthreads();

	for (int k=0;k<SIZE;++k)
		temp+=As[k]*_b[j_b+k];

	_c[i+j] = temp;
}

// Fill the input parameters and kernel qualifier
__global__ void teste (float *_a, float *_b, float *_c) {
	int i = blockIdx.x*SIZE;
	int j = threadIdx.x;
	_c[i+j] = i+j;
}

// Fill with the code required for the GPU stencil (mem allocation, transfers, kernel launch of stencilKernel)
void stencilGPU (void) {
	// Size of the array
	int bytes = SIZE_MATRIX*(sizeof(float));

	printf("Bytes: %d\n",bytes);

	// pointers to the device
	float *a_device,*b_device,*c_device;
	float *a,*b,*c;

	a = (float*)malloc(bytes);
	b = (float*)malloc(bytes);
	c = (float*)malloc(bytes);

	// allocate the memory on the device
	cudaMalloc(&a_device, bytes);
	cudaMalloc(&b_device, bytes);
	cudaMalloc(&c_device, bytes);

	checkCUDAError("mem allocation\n");

	// fills the arrays
	for (int i = 0; i < SIZE; ++i) {
		for(int j = 0; j < SIZE; ++j){
			a[i*SIZE+j] = 1;
			b[i*SIZE+j] = i+1;
			c[i*SIZE+j] = 0;
		}
	}

	// copy inputs to the device
	cudaMemcpy(a_device, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(c_device, c, bytes, cudaMemcpyHostToDevice);

	checkCUDAError("memcpy h->d");
	// declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

	startTime();
	stencilKernel <<<NUM_BLOCKS,NUM_THREADS_PER_BLOCK>>> (a_device, b_device, c_device);
	stopTime();

	checkCUDAError("kernel invocation");
	// copy the output to the host
	cudaMemcpy(c, c_device,bytes, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy d->h");

	//printfMatrix(c);

	// free the device memory
	cudaFree(a_device);cudaFree(b_device);cudaFree(c_device);
	free(a);free(b);free(c);

	checkCUDAError("mem free");
}

// Fill with the code required for the CPU stencil
int main (int argc, char** argv) {

	printf("NUM_THREADS %d\n", NUM_THREADS_PER_BLOCK);
	
	printf("GPU ");
	startTime();
	stencilGPU();
	stopTime();

	return 0;
}
