// Name: Conner Homrighaus
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc HW18.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float3 *P_GPU, *V_GPU, *F_GPU;
float *M, *M_GPU; 
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize, GridSize;

// Function prototypes
void setupCUDA();
void allocateMemCud();
void freeAll();
__global__ void forcesCUDA(float3*, float3*, float3*, float*, int);
__global__ void positionsCUDA(float3*, float3*, float3*, float*, int, float, float, float);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
void cudaErrorCheck(const char*, int);
int main(int, char**);

// cuda error!!! Don't forget it! 
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

// Set up CUDA
void setupCUDA()
{
	BlockSize.x = N;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;

	if(N > 1024)
	{
		printf("\n The number of bodies is greater than 1024. Exiting.\n");
		exit(0);
	}
}

// Allocate Memory I need
void allocateMemCud()
{
    cudaMalloc(&P_GPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&V_GPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&F_GPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&M_GPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}
// Calculate Forces!!!
__global__ void forcesCUDA(float3* p, float3* v, float3* f, float* m, int n)
{
	// INDEX
	int id = threadIdx.x + blockIdx.x*blockDim.x;

    /* OG Nbody Force!!!
    for(int i=0; i<N; i++)
		{
			F[i].x = 0.0;
			F[i].y = 0.0;
			F[i].z = 0.0;
		}
		for(int i=0; i<N; i++)
		{
			for(int j=i+1; j<N; j++)
			{
				dx = P[j].x-P[i].x;
				dy = P[j].y-P[i].y;
				dz = P[j].z-P[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (G*M[i]*M[j])/(d2) - (H*M[i]*M[j])/(d2*d2);
				F[i].x += force_mag*dx/d;
				F[j].x -= force_mag*dx/d;
				F[i].y += force_mag*dy/d;
				F[j].y -= force_mag*dy/d;
				F[i].z += force_mag*dz/d;
				F[j].z -= force_mag*dz/d;
			}
		}
    */
	
	if(id < n)
	{
		// Force magnitude 
		float force_mag;
        // Distances
		float dx, dy, dz, d, d2;

		// Start with all forces initalized to 0
		f[id].x = 0.0;
		f[id].y = 0.0;
		f[id].z = 0.0;

		// Loop through all the other bodies and get the force on the body we are working on.
		for(int i = 0; i < n; i++)
		{
			// Make sure we don't grab the force of the body we are currently on!
			if(i != id)
			{
				// Find the distance between the two bodies
				dx = p[i].x - p[id].x;
				dy = p[i].y - p[id].y;
				dz = p[i].z - p[id].z;
                d2 = dx*dx + dy*dy + dz*dz;
				d = sqrt(d2);

				// Calculate the force magnitude.
				force_mag = (G*m[id]*m[i])/(d2) - (H*m[id]*m[i])/(d2*d2);
				// Add the force to the body we are working on.
				f[id].x += force_mag*dx/d;
				f[id].y += force_mag*dy/d;
				f[id].z += force_mag*dz/d;
			}
		}
	}
}

__global__ void positionsCUDA(float3* p, float3* v, float3* f, float* m, int n, float dt, float time, float damp)
{
	// This kernel will update the position of each body :D
	
    // Index!!!
	int id = threadIdx.x + blockIdx.x*blockDim.x;

    /* Position Math From OG Nbody
    for(int i=0; i<N; i++)
		{
			if(time == 0.0)
			{
				V[i].x += (F[i].x/M[i])*0.5*dt;
				V[i].y += (F[i].y/M[i])*0.5*dt;
				V[i].z += (F[i].z/M[i])*0.5*dt;
			}
			else
			{
				V[i].x += ((F[i].x-Damp*V[i].x)/M[i])*dt;
				V[i].y += ((F[i].y-Damp*V[i].y)/M[i])*dt;
				V[i].z += ((F[i].z-Damp*V[i].z)/M[i])*dt;
			}
			P[i].x += V[i].x*dt;
			P[i].y += V[i].y*dt;
			P[i].z += V[i].z*dt;
		}
    */
	
	if(id < n)
	{
		// Update velocity and position!
		if(time == 0.0)
		{
			v[id].x += (f[id].x/m[id])*0.5*dt;
			v[id].y += (f[id].y/m[id])*0.5*dt;
			v[id].z += (f[id].z/m[id])*0.5*dt;
		}
		else
		{
			v[id].x += ((f[id].x-damp*v[id].x)/m[id])*dt;
			v[id].y += ((f[id].y-damp*v[id].y)/m[id])*dt;
			v[id].z += ((f[id].z-damp*v[id].z)/m[id])*dt;
		}
		p[id].x += v[id].x*dt;
		p[id].y += v[id].y*dt;
		p[id].z += v[id].z*dt;
	}
    // I tried to follow along using the previous nbody... Might need to get some
    // explained here. I'm pretty confident I somewhat understand, but I would
    // really appreciate maybe going over this in the future.
    // I think I will work this out as just an nbody problem on paper... I think that would help.
    // That reminds me, I need to go back over my adjacency matrix.
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		// Free EVERYTHING! We need to clean up the room!!!
        freeAll();
		exit(0);
	}
}

void freeAll()
{
    free(M);
	free(P);
	free(V);
	free(F);
	cudaFree(P_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(V_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(F_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(M_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

    // I have some time so I'm gonna make it pretty! 
    // My favorite gradiant time!!!
	for(i = 0; i < N; i++)
	{
		// Calculate a gradient factor between 0 and 1
		float gradientFactor = (float)i / (float)(N - 1);
		
		// Color ranging from blue (0.0, 0.0, 1.0) to green (0.0, 1.0, 0.0)
		glColor3f(0.0, gradientFactor, 1.0 - gradientFactor);
		
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius, 20, 20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
    // The way this was not all on the same line really messed
    // with me so I fixed it.
	drawPicture();
	gettimeofday(&start, NULL);
    nBody();
    gettimeofday(&end, NULL);
    drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	setupCUDA();
    allocateMemCud();
	printf("\n To start timing type s.\n");
}

/* OG nBody 
void nBody()
{
	float force_mag; 
	float dx,dy,dz,d, d2;
	int    drawCount = 0; 
	float  time = 0.0;
	float dt = 0.0001;
	while(time < RUN_TIME)
	{
		for(int i=0; i<N; i++)
		{
			F[i].x = 0.0;
			F[i].y = 0.0;
			F[i].z = 0.0;
		}
		for(int i=0; i<N; i++)
		{
			for(int j=i+1; j<N; j++)
			{
				dx = P[j].x-P[i].x;
				dy = P[j].y-P[i].y;
				dz = P[j].z-P[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				force_mag  = (G*M[i]*M[j])/(d2) - (H*M[i]*M[j])/(d2*d2);
				F[i].x += force_mag*dx/d;
				F[j].x -= force_mag*dx/d;
				F[i].y += force_mag*dy/d;
				F[j].y -= force_mag*dy/d;
				F[i].z += force_mag*dz/d;
				F[j].z -= force_mag*dz/d;
			}
		}
		for(int i=0; i<N; i++)
		{
			if(time == 0.0)
			{
				V[i].x += (F[i].x/M[i])*0.5*dt;
				V[i].y += (F[i].y/M[i])*0.5*dt;
				V[i].z += (F[i].z/M[i])*0.5*dt;
			}
			else
			{
				V[i].x += ((F[i].x-Damp*V[i].x)/M[i])*dt;
				V[i].y += ((F[i].y-Damp*V[i].y)/M[i])*dt;
				V[i].z += ((F[i].z-Damp*V[i].z)/M[i])*dt;
			}
			P[i].x += V[i].x*dt;
			P[i].y += V[i].y*dt;
			P[i].z += V[i].z*dt;
		}
		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) drawPicture();
			drawCount = 0;
		}
		time += dt;
		drawCount++;
	}
}
*/
void nBody()
{
	int    drawCount = 0; 
	float  time = 0.0;
	float dt = 0.0001;

	// Copying the data to the GPU.
	cudaMemcpyAsync(P_GPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(V_GPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(F_GPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(M_GPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	while(time < RUN_TIME)
	{
		// Calculating the forces for each body.
		forcesCUDA<<<GridSize, BlockSize>>>(P_GPU, V_GPU, F_GPU, M_GPU, N);
		cudaErrorCheck(__FILE__, __LINE__);

		// Updating the position of each body.
		positionsCUDA<<<GridSize, BlockSize>>>(P_GPU, V_GPU, F_GPU, M_GPU, N, dt, time, Damp);
		cudaErrorCheck(__FILE__, __LINE__);

        /*
        if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) drawPicture();
			drawCount = 0;
		}
        */
		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag)
			{
				// Copying the data back to the CPU.
				// cudaMemcpy acts as a synchronization point.
				cudaMemcpy(P, P_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
				cudaErrorCheck(__FILE__, __LINE__);
				drawPicture();
			}
			drawCount = 0;
		}
		/*
        time += dt;
		drawCount++;
        */
		time += dt;
		drawCount++;
	}
	// Copying the data back to the CPU one last time.
	// cudaMemcpy acts as a synchronization point.
	cudaMemcpy(P, P_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}