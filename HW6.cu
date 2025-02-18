// Name: Conner Homrighaus
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
unsigned int N = WindowWidth * WindowHeight; // Number of pixels
dim3 GridSize;
dim3 BlockSize;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

float *CPU_pixels, *GPU_pixels, *CPU_x, *GPU_x, *CPU_y, *GPU_y;

// Function prototypes
void cudaErrorCheck(const char*, int);
float escapeOrNotColor(float, float);

// Cuda Prototypes
__global__ void escapeOrNotColorGPU(float *, float *, float *, float, int, double, double);

// My Functions
void setUpDevice();
void allocateMemory();
void initailize();
void display(void);
void cleanUp();


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


// Set Up Device
void setUpDevice()
{
    // We were recommended to keep to the 1024 by 1024. That way every thread can handel 1 pixel!
    // Therefore, if I keep it with BlockSize.x and GridSize.x, then every block will have 1024 threads, and
    // there will be 1024 blocks. This matches the total number of pixels!!!

    BlockSize.x = WindowWidth;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = WindowHeight;
    GridSize.y = 1;
    GridSize.z = 1;
}

// Allocate Memory for Host and Device (CPU and GPU)
void allocateMemory()
{
    // CPU
    CPU_pixels = (float *)malloc(N*3*sizeof(float));
    CPU_x = (float *)malloc(WindowWidth*sizeof(float));
    CPU_y = (float *)malloc(WindowHeight*sizeof(float));

    // GPU
    cudaMalloc(&GPU_pixels, N*3*sizeof(float));
    cudaMalloc(&GPU_x, WindowWidth*sizeof(float));
    cudaMalloc(&GPU_y, WindowHeight*sizeof(float));
}


// This will be for any intailization I need to do. After my previous attempt, I have decided to try again.
// This time I would like my code to be a lot cleaner!
void initailize()
{
    // StepSize of X and Y
    float stepSizeX, stepSizeY;
    stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
    float j = XMin;
    float k = YMin;
    for(int i = 0; j > XMax; i += 1)
    {
        CPU_x[i] = j;
        j += stepSizeX;
    }
    for(int i = 0; k > XMax; i += 1)
    {
        CPU_y[i] = j;
        k += stepSizeY;
    }

}



__global__ void escapeOrNotColorGPU(float *pixels, float *x, float *y, float maxMag, int maxIt, double A, double B)
{
    // Okay, by using the x and y values I need to set the colors of every pixel!
    // In order to do that, I will more or less follow the code in display and the original
    // escapeornotcolor

    int pixel_id = (blockIdx.x*blockDim.x + threadIdx.x) * 3; // every thread handels one pixels colors!
	int id = blockIdx.x*blockDim.x + threadIdx.x; // every thread will handel one value of x and y

    // This is only because of the fact that there are 1024 x and y values. As well as 1024 pixels. However,
    // every pixel needs three spots for its colors!

    // Green and Blue off
    pixels[pixel_id+1] = 0.0; 	//Green off
	pixels[pixel_id+2] = 0.0;	//Blue off

    // Initailize Stats Needed
    int count = 0;
    int maxCount = maxIt;
    float mag = (x[id]*x[id] + y[id]*y[id]);
    float tempX;


    // Make sure we don't go out of bounds
    if(id < blockDim.x * gridDim.x)
    {
        while(mag < maxMag && count < maxCount) 
	    {
            tempX = x[id]; //We will be changing the x but we need its old value to find y.
            x[id] = x[id]*x[id] - y[id]*y[id] + A;
            y[id] = (2.0 * tempX * y[id]) + B;
            mag = sqrt(x[id]*x[id] + y[id]*y[id]);
            count++;
	    }
        if(count < maxCount) 
        {
            pixels[pixel_id] = 0.0;
        } else{
            pixels[pixel_id] = 1.0;
        }
    }
}

void display(void) 
{ 
    // Launch the Kernal! Store the values of the globals and pass them to the gpu since it does not have access
    // to them automagically.
	escapeOrNotColorGPU<<<GridSize,BlockSize>>>(GPU_pixels, GPU_x, GPU_y, MAXMAG, MAXITERATIONS, A, B);
    cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy GPU_pixels into CPU_pixels
    cudaMemcpy(CPU_pixels, GPU_pixels, N*3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, CPU_pixels); 
	glFlush(); 
}

void cleanUp()
{
    // CPU
    free(CPU_pixels);
    free(CPU_x);
    free(CPU_y);

    // GPU
    cudaFree(GPU_pixels);
    cudaFree(GPU_x);
    cudaFree(GPU_y);
}

int main(int argc, char** argv)
{ 

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	
    // When the program closes I need to clean up
    atexit(cleanUp);

    // Set Up my Devices
    setUpDevice();

    // Allocate Memory
    allocateMemory();
    
    // Initialize X and Y
    initailize();
    
    // Copy Memory from CPU to GPU
    cudaMemcpy(GPU_x, CPU_x, WindowWidth*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMemcpy(GPU_y, CPU_y, WindowHeight*sizeof(float), cudaMemcpyHostToDevice);
    cudaErrorCheck(__FILE__, __LINE__);

    glutMainLoop();
}
