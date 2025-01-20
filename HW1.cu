// Name: Conner Homrighaus
// nvcc HW1.cu -o temp
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
I understand that this code is a vector adder that adds via array addition. I played with the pointers a bit in my 
visual studio, but I did not commit those changes in order to perserve the original code!
I did make a few of my own comments on things that I went through and double checked I understood! For example, what tv_sec did!
 2. Compile, run, and play around with the code.
I did play around with the code, just on the local machine. However, I am going to go ahead and run the command
git pull in order to merge this update with the cloned repository on my machine!
*/

// Include files
#include <sys/time.h>
// sys/time allows us to grab various time based functions! 
// tv_sec: Seconds since the Epoch (January 1, 1970).
#include <stdio.h>
// Standard Header!



// Defines
#define N 1000 // Length of the vector

// Global variables
// Float *A_CPU, *B_CPU, and *C_CPU are pointers! They are currently not pointing to anything
// at this moment in the code!
float *A_CPU, *B_CPU, *C_CPU; 
//This tolerance is important because of the way float values are stored! Without this tollerance we 
// would be unable to check if the math was done correctly later in the code!
float Tolerance = 0.00000001;

// Function prototypes
// These function prototypes are used so that when the functions are called we don't
// get errors! 
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

//Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.	
	// Malloc allows for dynamic memory allocation and is very important when we are
	// unsure of how many variables we need to store within a pointer!
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		// Vector addition through the use of arrays! 
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	int id;
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(id = 0; id < n; id++)
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

//Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
}

int main()
{
	timeval start, end;
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();

	// Starting the timer.	
	gettimeofday(&start, NULL);

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU");
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end));
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n");
	
	return(0);
}

