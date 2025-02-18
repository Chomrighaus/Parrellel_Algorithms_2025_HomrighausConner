// Name: Conner Homrighaus
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
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
unsigned int WindowWidth = 1400;
unsigned int WindowHeight = 1250;
int N = WindowWidth * WindowHeight; // Number of Pixels

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float, int, int);

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

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int width, int height) 
{
	//Getting the offset into the pixel buffer. 
	//We need the 3 because each pixel has a red, green, and blue value.
	int pixelIndex = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(pixelIndex < width * height)
	{
		float x,y,mag,tempX;
		int count, id;
		
		int maxCount = MAXITERATIONS;
		float maxMag = MAXMAG;

		// Find the x and y units of where the pixel is
		int pixel_X = pixelIndex % width;
        int pixel_Y = pixelIndex / width;



		//Asigning each thread its x and y value using aspect ratio!
		x = xMin + dx*pixel_X;
		y = yMin + dy*pixel_Y;
		
		count = 0;
		mag = sqrt(x*x + y*y);
		while (mag < maxMag && count < maxCount) 
		{
			//We will be changing the x but we need its old value to find y.	
			tempX = x; 
			x = x*x - y*y + A;
			y = (2.0 * tempX * y) + B;
			mag = sqrt(x*x + y*y);
			count++;
		}
		
		id = 3 * pixelIndex;
		//Setting the red value
		if(count < maxCount) //It excaped, set it to no color!
		{
			pixels[id]     = 0.0;
			pixels[id + 1] = 0.0;
			pixels[id + 2] = 0.0;
		}
		else //It Stuck around! Do a gradiant!
		{
			float t = (float)count / MAXITERATIONS;
			pixels[id] = 0.0; // Red
			pixels[id + 1] = t * 1.0; // Green
			pixels[id + 2] = (1.0 - t) * 1.0; // Blue
		}
	}

}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	if(WindowWidth < 1024)
	{
		blockSize.x = WindowWidth;
	} else {
		blockSize.x = 1024; // maximum 
	}
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = WindowHeight;
	gridSize.y = 1;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
}


int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}



