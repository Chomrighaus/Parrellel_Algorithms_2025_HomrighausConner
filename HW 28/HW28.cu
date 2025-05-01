// Name: Conner Homrighaus
// CPU random walk. 
// nvcc HW28.cu -o temp

/*
 What to do:
 This is some code that runs a random walk for 10000 steps.
 Use cudaRand and run 10 of these runs at once with diferent seeds on the GPU.
 Print out all 10 final positions.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

// Defines
#define WALKS 10

// Globals
int NumberOfRandomSteps = 10000; // 10,000
//float MidPoint = (float)RAND_MAX/2.0f; Unneccessary! I wrote down what Kyle and Mason did! That someNumber += 2*(rand % 2) - 1;
int *p_CPU; // Position pointer for CPU
int *p_GPU; // Position pointer for GPU
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *file, int line); // cudaErrorCheck(__FILE__, __LINE__); to call
void setup();
void allocateMem(); // decided to do this somewhere else instead of setup for readability
void clean();
__global__ void randomWalk(int*, int, int, unsigned long long);
int main(int, char**);

// Error check
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will set up the kernal!
void setup()
{
	// Setup Block Size
	BlockSize.x = WALKS; // Number of Walks
	BlockSize.y = 1; BlockSize.z = 1;

	// Setup Grid Size
	GridSize.x = 1; // I know I could do the math thing, but I remember us saying just to do this
	// in one block
	GridSize.y = 1; GridSize.z = 1;
}

void allocateMem()
{
	// Allocate memory on the CPU
	p_CPU = (int*)malloc(WALKS * sizeof(int));

	// Allocate memory on the GPU 
	cudaMalloc(&p_GPU, WALKS * sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
}

void clean()
{
	// Free up CPU
	free(p_CPU);

	// Free up GPU
	cudaFree(p_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void randomWalk(int* finalP_GPU, int steps, int walk, unsigned long long time)
{
	// Global Index of the thread
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= walk) return; // Verification that we don't touch someone else's yard

	// Need to keep track of the position
	int p = 0;

	curandState state; // Initialize state!
	curand_init(id, time, 0, &state); // Initialize the random state! Give it a seed!!!!

	for(int i = 0; i < steps; i++) 
	{
		p += 2*(curand(&state) % 2) - 1; // Okay, this seriously saves so much time, I'm glad I wrote this down when Kyle and Mason did this!
		// Essentially this will return either a 1 or a -1! Which I think is really cool!
	}

	// Store the final position in the array
	finalP_GPU[id] = p;
}

int main(int argc, char** argv)
{
	// Setup device dimensions
	setup();

	// Allocate memory on CPU and GPU
	allocateMem();
	
	// Call random walk! Should perform 10 random walks!
	randomWalk<<<GridSize, BlockSize>>>(p_GPU, NumberOfRandomSteps, WALKS, (unsigned long long)time(NULL));
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy the results to CPU
	cudaMemcpy(p_CPU, p_GPU, WALKS * sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);

	// Make sure GPU is done!
	cudaDeviceSynchronize(); 
	cudaErrorCheck(__FILE__, __LINE__);

	// Print out all walks
	for(int i = 0; i < WALKS; i++)
	{
		printf("Walk [%d] Final Position: %d\n", i+1, p_CPU[i]);
	}
	
	// We are done so clean up your yard!
	clean();

	// classic return, can't delete it, it's our friend
	return 0;
}

