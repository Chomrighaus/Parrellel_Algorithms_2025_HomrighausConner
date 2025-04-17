// Name:
// Vector addition on two GPUs.
// nvcc HW22.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 1:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 2:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU0, *B_GPU0, *C_GPU0; //Half of GPU Pointers
float *A_GPU1, *B_GPU1, *C_GPU1; //Second half of GPU Pointers
dim3 BlockSize, GridSize0, GridSize1; //This will hold the dimensions of our grid and blocks
float Tolerance = 0.01;
// Compute proper split
int halfN = N / 2; //First half of split
int secondHalfN = N - halfN;//Second half of split

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
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

// This will be the layout of the parallel space we will be using.
// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	//grab device count in order to confirm the pc has 2 devices
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting CUDA device count: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (deviceCount < 2) {
        fprintf(stderr, "At least two CUDA-capable GPUs are required.\n");
        exit(EXIT_FAILURE);
    }

    // Set desired block size (same for both GPUs)
    BlockSize.x = 256;
    BlockSize.y = 1;
    BlockSize.z = 1;

    // Compute per-GPU grid sizes
    GridSize0.x = (halfN + BlockSize.x - 1) / BlockSize.x;
    GridSize0.y = 1;
    GridSize0.z = 1;

    GridSize1.x = (secondHalfN + BlockSize.x - 1) / BlockSize.x;
    GridSize1.y = 1;
    GridSize1.z = 1;

    // Check if each GPU can support the grid/block config
    for (int i = 0; i < 2; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        if (BlockSize.x > prop.maxThreadsDim[0] ||
            BlockSize.y > prop.maxThreadsDim[1] ||
            BlockSize.z > prop.maxThreadsDim[2]) {
            fprintf(stderr, "Device %d cannot support block dimensions (%d, %d, %d).\n",
                    i, BlockSize.x, BlockSize.y, BlockSize.z);
            exit(EXIT_FAILURE);
        }

        dim3 checkGridSize = (i == 0) ? GridSize0 : GridSize1;
        if (checkGridSize.x > prop.maxGridSize[0] ||
            checkGridSize.y > prop.maxGridSize[1] ||
            checkGridSize.z > prop.maxGridSize[2]) {
            fprintf(stderr, "Device %d cannot support grid dimensions (%d, %d, %d).\n",
                    i, checkGridSize.x, checkGridSize.y, checkGridSize.z);
            exit(EXIT_FAILURE);
        }
    }

    printf("Both GPU 0 and GPU 1 validated for block and grid sizes.\n");
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
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
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

    // Freeing device memory on GPU 0
    cudaSetDevice(0);
    cudaFree(A_GPU0); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU0); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU0);
    cudaErrorCheck(__FILE__, __LINE__);

    // Freeing device memory on GPU 1
    cudaSetDevice(1);
    cudaFree(A_GPU1); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(B_GPU1); 
    cudaErrorCheck(__FILE__, __LINE__);
    cudaFree(C_GPU1);
    cudaErrorCheck(__FILE__, __LINE__);
}

void allocateMemory()
{
    // Allocate host (CPU) memory
    A_CPU = (float*)malloc(N * sizeof(float));
    B_CPU = (float*)malloc(N * sizeof(float));
    C_CPU = (float*)malloc(N * sizeof(float));

    if (!A_CPU || !B_CPU || !C_CPU) {
        fprintf(stderr, "Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory on GPU 0
    cudaSetDevice(0);
    cudaMalloc(&A_GPU0, halfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU0, halfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU0, halfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);

    // Allocate device memory on GPU 1
    cudaSetDevice(1);
    cudaMalloc(&A_GPU1, secondHalfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU1, secondHalfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU1, secondHalfN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
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
	
	// Copy data to GPU 0
    cudaSetDevice(0);
    cudaMemcpyAsync(A_GPU0, A_CPU, halfN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(B_GPU0, B_CPU, halfN * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data to GPU 1
    cudaSetDevice(1);
    cudaMemcpyAsync(A_GPU1, A_CPU + halfN, secondHalfN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(B_GPU1, B_CPU + halfN, secondHalfN * sizeof(float), cudaMemcpyHostToDevice);

	cudaSetDevice(0);
    addVectorsGPU<<<GridSize0, BlockSize>>>(A_GPU0, B_GPU0, C_GPU0, halfN);

    cudaSetDevice(1);
    addVectorsGPU<<<GridSize1, BlockSize>>>(A_GPU1, B_GPU1, C_GPU1, secondHalfN);

    // Copy results back to host
    cudaSetDevice(0);
    cudaMemcpyAsync(C_CPU, C_GPU0, halfN * sizeof(float), cudaMemcpyDeviceToHost);

    cudaSetDevice(1);
    cudaMemcpyAsync(C_CPU + halfN, C_GPU1, secondHalfN * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
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

