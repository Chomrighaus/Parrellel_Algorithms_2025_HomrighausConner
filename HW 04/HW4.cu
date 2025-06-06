// Name: Conner Homrighaus
// nvcc HW4.cu -o temp
/*
 What to do:
 This is the solution to HW3. It works well for adding vectors with fixed-size blocks. 
 Given the size of the vector it needs to add, it takes a set block size, determines how 
 many blocks are needed, and creates a grid large enough to complete the task. Cool, cool!
 
 But—and this is a big but—this can get you into trouble because there is a limited number 
 of blocks you can use. Though large, it is still finite. Therefore, we need to write the 
 code in such a way that we don't have to worry about this limit. Additionally, some block 
 and grid sizes work better than others, which we will explore when we look at the 
 streaming multiprocessors.
 
 Extend this code so that, given a block size and a grid size, it can handle any vector addition. 
 Start by hard-coding the block size to 256 and the grid size to 64. Then, experiment with different 
 block and grid sizes to see if you can achieve any speedup. Set the vector size to a very large value 
 for time testing.

 You’ve probably already noticed that the GPU doesn’t significantly outperform the CPU. This is because 
 we’re not asking the GPU to do much work, and the overhead of setting up the GPU eliminates much of the 
 potential speedup. 
 
 To address this, modify the computation so that:
 c = sqrt(cos(a)*cos(a) + a*a + sin(a)*sin(a) - 1.0) + sqrt(cos(b)*cos(b) + b*b + sin(b)*sin(b) - 1.0)
 Hopefully, this is just a convoluted and computationally expensive way to calculate a + b.
 If the compiler doesn't recognize the simplification and optimize away all the unnecessary work, 
 this should create enough computational workload for the GPU to outperform the CPU.

 Write the loop as a for loop rather than a while loop. This will allow you to also use #pragma unroll 
 to explore whether it provides any speedup. Make sure to include an if (id < n) condition in your code 
 to ensure safety. Finally, be prepared to discuss the impact of #pragma unroll and whether it helped 
 improve performance.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <math.h>

// Defines
#define N 11504 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.00000001;

// Function prototypes
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 64; 
	GridSize.y = 1;
	GridSize.z = 1;
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
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = sqrt(a[id]*a[id] + (sin(a[id])*sin(a[id]) + cos(a[id])*cos(a[id]) - 1)) + sqrt(b[id]*b[id] + (sin(b[id])*sin(b[id]) + cos(b[id])*cos(b[id]) - 1));
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	
	if(id < N)
		{
			#pragma unroll 2
			for (int i = id; i < n; i += stride)
        	{
				c[id] = sqrt(a[id]*a[id] + (sin(a[id])*sin(a[id]) + cos(a[id])*cos(a[id]) - 1)) + sqrt(b[id]*b[id] + (sin(b[id])*sin(b[id]) + cos(b[id])*cos(b[id]) - 1));
        	}
		}

}


// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(int id = 0; id < n; id++)
	{ 
		sum += c[id];
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) 
	{
		return(1);
	}
	else 
	{
		return(0);
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
	cudaFree(B_GPU); 
	cudaFree(C_GPU);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);

	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();

	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}

