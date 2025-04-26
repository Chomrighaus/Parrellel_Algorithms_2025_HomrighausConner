// Name: Conner Homrighaus
// CPU random walk. 
// nvcc HW27.cu -o temp

/*
 What to do:
 Create a function that returns a random number that is either -1 or 1.
 Start at 0 and call this function to move you left (-1) or right (1) one step each call.
 Do this 10000 times and print out your final position.
*/

/*
Conner's Idea of How To:
Global function to perform the actual walk. Using random numbers on the GPU. Then copy
the last number, or position, over to the cpu. Then I will ave the computer print out the 
number by copying the number over to the cpu.
N will be the number of iterations, in this case 10000. Because we are just dealing with
+1 and -1 I can just use all integers. Now, in order to actually do this, I need to have
the gpu go over all 10000 iterations, and add them as we go. My plan for this is
to use attomic add and a variable called FP_GPU or finale position.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h> // Randoms on the GPU
#include <unistd.h> // For usleep on Linux

// Defines
#define N 10000

// Globals
int FP_CPU;
int *FP_GPU;
dim3 BlockSize;
dim3 GridSize;
curandState *devStates;
unsigned long seed;

// Function prototypes
void cudaErrorCheck(const char *file, int line);
__global__ void randomWalk(curandState *, unsigned long, int *); 
void setup();
void clean();
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

// This will be where we perform the random walk on the GPU
__global__ void randomWalk(curandState *state, unsigned long seed, int *fp)
{
	// Global Index!
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= N) 
	{
		return; // This simply means if id >= n then do nothing!
	}

    // Initialize random state
    curand_init(seed, id, 0, &state[id]); 
	/*
	seed	A starting value for randomness, like time(NULL)
	id	A sequence number (unique per thread)
	0	The "offset" into the sequence (you can skip ahead if needed)
	&state[id]	Pointer to where this thread's random state will be stored
	*/

    // Generate random number
    float randNum = curand_uniform(&state[id]); // The way this works is very simple! The current range is (0, 1]
	// Essentially curand_uniform(&state[id]); is an inequality 0 <= u <= 1! 
	// So to calculate a different range we can just play like that! For example, say I wanted it from [-2 to 2], then I would
	// -2 <= 4u - 2 <= 2, that is to say 2.0f * 2.0f * curand_uniform(&state[id]) - 2.0f; 

    int step = (randNum < 0.5f) ? -1 : 1; // If the step is > 0.5, then 1, if it is < .5 then it is -1!

    // Atomically add to final position with atomicAdd to avoid raceing or any weird things happening!
    atomicAdd(fp, step);
}

// This will set up the kernal!
void setup()
{
	// Setup Block Size
	BlockSize.x = 1000; // 1000 threads per block means 
	BlockSize.y = 1; BlockSize.z = 1;

	// Setup Grid Size
	GridSize.x = (N-1)/BlockSize.x + 1; // 10 blocks will be needed. But this is fancier
	GridSize.y = 1; GridSize.z = 1;

	// Allocate Memory for GPU
	cudaMalloc(&FP_GPU, sizeof(int)); // We are using FP_GPU as simply an int, not as an array! But it
	// still has to be a pointer because that is how we allocate memory on the GPU. Found that out several times
	// when I was coding late at night LOL! Don't pull a me and try to code on a laptop at 2:00am because
	// your little sister gets sick! That is not the time to be doing HW!
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(FP_GPU, 0, sizeof(int)); // This makes sure we zero everything out!
	cudaErrorCheck(__FILE__, __LINE__);

	// Allocate random state array
	cudaMalloc((void **)&devStates, N * sizeof(curandState));
	cudaErrorCheck(__FILE__, __LINE__);
}

// This will clean up any loose ends
void clean()
{
	// Clean up GPU
	cudaFree(FP_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(devStates);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char** argv)
{
	// Setup everything
	setup();

	for (int i = 0; i < 10; i++) // <-- Loop 10 times
    {
        // Get a new seed each time
        seed = time(NULL) + i;

        // Launch random walk kernel
        randomWalk<<<GridSize, BlockSize>>>(devStates, seed, FP_GPU);
        cudaErrorCheck(__FILE__, __LINE__);

        cudaDeviceSynchronize();
        cudaErrorCheck(__FILE__, __LINE__);
    
        // Copy final position back
        cudaMemcpy(&FP_CPU, FP_GPU, sizeof(int), cudaMemcpyDeviceToHost);
        cudaErrorCheck(__FILE__, __LINE__);
    
        // Print the final position
        printf("Final position after %d steps (Run %d): %d\n", N, i + 1, FP_CPU);

        // Optional: Small delay so time(NULL) is more likely to change
        usleep(200000); // 0.2 seconds delay (if using Linux/Mac)
    }
 
	// Clean up
	clean();
 
	 return 0;
}

