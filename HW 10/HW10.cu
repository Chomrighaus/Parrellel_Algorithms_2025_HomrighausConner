// Name: Conner Homrighaus
// Vector Dot product on many block 
// nvcc HW10.cu -o run
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
 4. In HW9 we had to do the final add "reduction" on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 1024 // Threads in a block

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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
/*
1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.
 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.
*/
// This will be the layout of the parallel space we will be using.
void setUpDevices() {
    // Ensure BLOCK_SIZE is a power of 2
    if ((BLOCK_SIZE & (BLOCK_SIZE - 1)) != 0) { // Use bitwise operators!
        printf(" BLOCK_SIZE must be a power of 2. Please correct this -_-. Have a good day! :D\n");
        exit(EXIT_FAILURE);
    }
	// The above works because a power of two will only have one bit active at a time! 0010 is a power of 2! But 0110 is not!

	// Get number of CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf(" No Cuda Devices... Really? Alrighty, don't know why you're trying to run Cuda then. Maybe switch devices? Have a good day.\n");
        exit(EXIT_FAILURE);
    }

    printf(" Number of CUDA devices: %d\n", deviceCount);

    // Get device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Ensure the block size is within the allowed limit
    if (BLOCK_SIZE > prop.maxThreadsPerBlock) {
        printf(" BLOCK_SIZE is bigger than the maximum allowed per block (%d). Sorry homie! Still have a good day! :D\n", prop.maxThreadsPerBlock);
        exit(EXIT_FAILURE);
    }

    // set up block and grid
    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Ensures correct division
    GridSize.y = 1;
    GridSize.z = 1;

    // Ensure the grid size is within device limits
    if (GridSize.x > prop.maxGridSize[0]) {
        printf(" Grid size is bigger than the maximum allowed (%d). My apologies, but I am exiting. :(\n", prop.maxGridSize[0]);
        exit(EXIT_FAILURE);
    }

	// Find the best GPU based on compute capability
	int bestDevice = 0;
    int maxComputeCapability = 0;
    for (int i = 0; i < deviceCount; i++) {
        int computeCapability = prop.major * 10 + prop.minor; // prop.major gives us overall architecture family, and prop.minor gives us additional feature details. You need both
		// to fully compute capability 
        if (computeCapability >= 30 && computeCapability > maxComputeCapability) {
            maxComputeCapability = computeCapability;
            bestDevice = i;
        }
    }
	// Because atomic floats being able to compute floats did not come until compute capability
	// 3.0, then I need to make sure any of the devices can do it!
	if (maxComputeCapability < 30) {
		printf(" Your GPU does not support atomic operations for floating-point numbers. Sorry! \n");
		exit(EXIT_FAILURE);
	}

    // Set the best available device
    cudaSetDevice(bestDevice);
    printf(" Using GPU %d with compute capability %d.%d\n", bestDevice, maxComputeCapability / 10, maxComputeCapability % 10);
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

/*
 3. Always checking to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values.
 4. In HW9 we had to do the final add "reduction" on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity.
*/
// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
    int tid = threadIdx.x;
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Shared memory for block-level reduction
    __shared__ float temp[BLOCK_SIZE];
    
    // Put data into shared memory
    if (id < n) {
        temp[tid] = a[id] * b[id];
    } else {
        temp[tid] = 0.0f; // Padding with zeros
    }
    __syncthreads();

    // Perform block-level reduction
	// For some reason if I don't define stride inside of this it does not work.
	// If you could figure out why or if we could talk about it I would appreciate it.
	// I genuinely don't know why this does not work.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Use atomicAdd to safely add everything and store results into final dot product
    if (tid == 0) {
        atomicAdd(c, temp[0]);
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// I wanted to try this! And it works!
	float DotGPU = 0.0f;
	cudaMemcpy(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
		// Just cause I can
		printf("The value of DotCPU:\n %f\n The value of DotGPU:\n %f", DotCPU, DotGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


