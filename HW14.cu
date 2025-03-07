// Name: Conner Homrighaus
// Page-locked memory test
// nvcc HW14.cu -o run

/*
 What to do:
 Read chapter 10 in your book and file in the ???s in this code to understand how to setup and
 test page-locked memory on the host.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include "./MyCuda.h"

// Defines
#define SIZE 2000000 
#define NUMBER_OF_COPIES 1000

//Globals
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
cudaEvent_t StartEvent, StopEvent;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void cleanUp();
void copyPageableMemoryUp();
void copyPageLockedMemoryUp();
void copyPageableMemoryDown();
void copyPageLockedMemoryDown();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&NumbersOnGPU, SIZE*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate pageable Host (CPU) Memory
	PageableNumbersOnCPU = (float*)malloc(SIZE*sizeof(float));
	
	//Allocate page locked Host (CPU) Memory
	cudaMallocHost(&PageLockedNumbersOnCPU, SIZE * sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(NumbersOnGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(PageLockedNumbersOnCPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	free(PageableNumbersOnCPU); 
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

void copyPageableMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageableMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryUp()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}


int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory up = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using pageable memory down = %3.1f milliseconds", timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorChecmyCudaErrorCheckk(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU using page locked memory down = %3.1f milliseconds", timeEvent);
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}

/*
I do not have #include "./MyCuda.h" nor is it in your file, but this needs to get turned in so I am turning it in
*/