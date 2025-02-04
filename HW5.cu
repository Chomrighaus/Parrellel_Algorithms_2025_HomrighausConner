// Name: Conner Homrighaus
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		printf("\n--- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor); 
        printf("Clock rate: %d kHz\n", prop.clockRate);
        printf("Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
        printf("Kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        printf("\n--- Memory Information for device %d ---\n", i);
        printf("Total global memory: %ld bytes\n", prop.totalGlobalMem);
        printf("Total constant memory: %ld bytes\n", prop.totalConstMem);
        printf("Max memory pitch: %ld bytes\n", prop.memPitch); // Max integer that can be stored
        printf("Texture alignment: %ld bytes\n", prop.textureAlignment); // the minimum required alignment to create linear texture from a buffer
        printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
        printf("Shared memory per multiprocessor: %ld bytes\n", prop.sharedMemPerMultiprocessor);
        printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("ECC support: %s\n", prop.ECCEnabled ? "Enabled" : "Disabled"); // Error correcting code

        printf("\n--- Execution Configuration for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Threads per warp: %d\n", prop.warpSize);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        printf("\n--- Compute Features for device %d ---\n", i);
        printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("Managed memory: %s\n", prop.managedMemory ? "Yes" : "No");
        printf("Concurrent managed access: %s\n", prop.concurrentManagedAccess ? "Yes" : "No");
        printf("Direct Managed Memory Access from host: %s\n", prop.directManagedMemAccessFromHost ? "Yes" : "No");
        printf("Asynchronous Engine Count: %d\n", prop.asyncEngineCount);
        printf("PCI Bus ID: %d\n", prop.pciBusID);
        printf("PCI Device ID: %d\n", prop.pciDeviceID);
        printf("PCI Domain ID: %d\n", prop.pciDomainID);
        printf("\n");
	}	
	return(0);
}

