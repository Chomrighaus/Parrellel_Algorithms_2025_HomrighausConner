// Name: Conner Homrighaus
// nBody run on all available GPUs. 
// nvcc HW25.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 It runs on two GPUs and two GPUs only. Rewrite it so it automatically detects the number of 
 available GPUs on the machine and runs using all of them.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0


// Globals
int N;
int *BGPU; // Bodies on each GPU
int NumberOfGpus; // This will hold the number of gpus
float3 *P, *V, *F;
float *M; 
// I need to use array of pointers. 
float3 **PGPU[25], **VGPU[25], **FGPU[25];
float **MGPU[25];
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 *GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup(int NumberOfGpus);
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
void nBody(int NumberOfGpus);
void cleanUpRoom(int NumberOfGpus);
int main(int, char**);

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

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaSetDevice(0);
	cudaMemcpyAsync(P, PGPU[0], N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	
	for(int i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int test;

	// This will set up the number of bodies 
	N = 101;
	
	// Get the number of GPUS
	cudaGetDeviceCount(&NumberOfGpus);
	if(NumberOfGpus == 0) // No GPUS
	{
		printf("\n Initiating cashapp, trust me you need it. Get a GPU broke human!\n");
		exit(0);
	}
	else // uncountable? negative? matter subtracted from the univers?
	{ 
		printf("\n Dude... How? Just how? I don't even think Elon Musk with all of his money could invent negative gpus...\n");
		printf("\n I won't share my code or play with liars!\n");
		exit(0);
	}
	printf("\n The number of GPUs is: %d\n", NumberOfGpus);

	// Allocate memory for arrays dependent on the number of GPUs
	BGPU = (int*)malloc(NumberOfGpus*sizeof(int));
	PGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));
	VGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));
	FGPU = (float3**)malloc(NumberOfGpus*sizeof(float3*));
	MGPU = (float**)malloc(NumberOfGpus*sizeof(float*));
	GridSizes = (dim3*)malloc(NumberOfGpus*sizeof(dim3));

	// Now we set up the memory on CPU.
    M = (float*)malloc(N*sizeof(float));
    P = (float3*)malloc(N*sizeof(float3));
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));
	
	// Now lets set up the number of bodies on each and every gpu
	for(int i = 0; i < NumberOfGpus; i++)
	{
		BGPU[i] = N/NumberOfGpus;
	}
	BGPU[NumberOfGpus-1] += N % NumberOfGpus; // This is to make sure we have all the bodies accounted for.
	
	// Set up the blocks
	BlockSize.x = 128;
	BlockSize.y = 1;
	BlockSize.z = 1;
		
	// I really liked Kyle's multiple number of gpus gridsize! So I am going to make sure that I get this!
	for(int i = 0; i < NumberOfGpus; i++)
	{
		GridSizes[i].x = (NGPUs[i] - 1)/BlockSize.x + 1;
		GridSizes[i].y = 1;
		GridSizes[i].z = 1;
	}
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;

	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
    Damp = 0.5;
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
			
			// Making sure the bodies' centers are at least a diameter apart.
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


	// Because this is a double pointer, or an array of pointers, I think I need the address of each pointer in order
	// to make this work... Might as well try!
	for(int i = 0; i < NumberOfGpus; i++) 
	{
		cudaSetDevice(i);
		cudaMemcpyAsync(&PGPU[i], P, N*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(&VGPU[i], V, N*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(&FGPU[i], F, N*sizeof(float3), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(&MGPU[i], M, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	printf("\n Setup finished.\n");
}

// Position, Velocity, Force, Mass, Gravity, Time Step (sigma thing), Bodies On Gpu, Total Number of Bodies, Shift Over
__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int gpuB, int N, int shift)
{
    float dx, dy, dz, d, d2;
    float force_mag;
    
    // Global Index
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(i < gpuB)
	{
		i += shift; // Make sure I shift over

		// Zero out forces
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < N; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x; // Find the change in the X position
				dy = p[j].y-p[i].y; // Find the change in the Y position
				dz = p[j].z-p[i].z; // Find the change in the Z position
				d2 = dx*dx + dy*dy + dz*dz; // Find distance squared
				d  = sqrt(d2); // Find distance
				
				// Calculate force 
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2); 
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

// Position, Velocity, Force, Mass, dampening, change in time, time, Bodies On Gpu, Total Number of Bodies, Shift Over
__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int gpuB, int N, int shift)
{
    // Global Index 
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(i < gpuB)
	{
		i += shift;
		
		// I thought about optimizing this, but I don't want to accidentally confuse myself...
		if(t == 0.0f) // Initial Time Step
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else // Everything else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		// Update position
		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}


void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;
	int shift = BGPU[0];

	while(t < RUN_TIME)
	{
		// Run through all gpus, calculate forces and move bodies!
		for(int device = 0; device < NumberOfGpus; device++)
		{
			cudaSetDevice(device);
			getForces<<<GridSizes[device],BlockSize>>>(PGPU[device], VGPU[device], FGPU[device], MGPU[device], G, H, BGPU[device], N, i*shift);
			cudaErrorCheck(__FILE__, __LINE__);
			moveBodies<<<GridSizes[device],BlockSize>>>(PGPU[device], VGPU[device], FGPU[device], MGPU[device], Damp, dt, t, BGPU[device], N, i*shift);
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Sync all devices!
		for(int device = 0; device < NumberOfGpus; device++)
		{
			cudaSetDevice(device);
			cudaDeviceSynchronize();
			cudaErrorCheck(__FILE__, __LINE__);
		}
		
		// Copying memory between GPUs. You do not need to copy velocities because velocities are only updated with 
		// information from it's own thread.
		for(int device = 0; device < NumberOfGpus; device++)
		{
			cudaSetDevice(device);
			for(int otherDevice = 0; otherDevice < NumberOfGpus; otherDevice++)
			{
				if(i != j)
				{
					cudaMemcpyAsync(PGPU[otherDevice] + shift*device, PGPU[device] + shift*device, BGPU[device]*sizeof(float3), cudaMemcpyDeviceToDevice);
					cudaErrorCheck(__FILE__, __LINE__);
				}
			}	
		}
		
		// Sync all devices!
		for(int device = 0; device < NumberOfGpus; device++)
		{
			cudaSetDevice(device);
			cudaDeviceSynchronize();
			cudaErrorCheck(__FILE__, __LINE__);
		}

		if(drawCount == DRAW_RATE) 
		{	
			drawPicture();
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}


// Free memory
void cleanUpRoom()
{
	// Free memory on CPU
	free(P);
	free(V);
	free(F);
	free(M);

	// Free the bodies
	free(BGPU);
	
	// free GPU pointers first!
	for (int i = 0; i < NumberOfGpus; i++) {
		cudaSetDevice(i);
		if(PGPU[i]) cudaFree(PGPU[i]);
		if(VGPU[i]) cudaFree(VGPU[i]);
		if(FGPU[i]) cudaFree(FGPU[i]);
		if(MGPU[i]) cudaFree(MGPU[i]);
	}
	
	// Free the host side
	free(PGPU);
	free(VGPU);
	free(FGPU);
	free(MGPU);
}



int main(int argc, char** argv)
{
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
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
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
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

	// Clean up the room
	cleanUpRoom();
	return 0;
}
