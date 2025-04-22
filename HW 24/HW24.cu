// Name: Conner Homrighaus
// nBody code on multiple GPUs. 
// nvcc HW24.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code with all the bells and whistles removed. 
 Modify it so it runs on two GPUs.
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
#define RUN_TIME 10.0

// Globals
int N;
float3 *P, *V, *F;
float *M; 
float3 *PGPU0, *VGPU0, *FGPU0;
float *MGPU0;
float3 *PGPU1, *VGPU1, *FGPU1;
float *MGPU1;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *, float3 *, float3 *, float *, float, float, int, int, int);
__global__ void moveBodies(float3 *, float3 *, float3 *, float *, float, float, float, int, int, int);
void nBody();
void cleanUpRoom();
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

// Visualization
void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaSetDevice(0);
	cudaMemcpyAsync(P, PGPU0, N*sizeof(float3), cudaMemcpyDeviceToHost);
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


// Setup scene
void setup()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__, __LINE__);
	if (deviceCount < 2) {
		fprintf(stderr, "How Dare You, You Scrub! How Dare You Think You Could Run Me Without Two GPUs!\n");
		fprintf(stderr, "Send me ur paypal and I'll CONSIDER buying you a second GPU you scrub. __(ツ)_/¯\n");
		exit(EXIT_FAILURE);
	}
	

	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;
	
	// Number of bodies
	N = 1000;

	BlockSize.x = BLOCK_SIZE; 
	BlockSize.y = 1; 
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; GridSize.y = 1; GridSize.z = 1;
	GridSize.y = 1;
	GridSize.z = 1;

	Damp = 0.5;

	M = (float*)malloc(N*sizeof(float));
	P = (float3*)malloc(N*sizeof(float3));
	V = (float3*)malloc(N*sizeof(float3));
	F = (float3*)malloc(N*sizeof(float3));

	// I keep getting a memory dump. After looking around, I know the problem is here.
	// Because of this, I have looked to chatgpt and it was useless. Then I went to
	// copilot and it said I am not allocating enough memory when I try to use
	// N and halfOfN. It suggested I try n*2, so that is what I am going to try.

	// I need half the number of bodies
	int halfOfN = N/2;
	// n*2
	int nTimesTwo = 2*halfOfN;

	cudaSetDevice(0);
	cudaMalloc(&MGPU0, nTimesTwo*sizeof(float));
	cudaMalloc(&PGPU0, nTimesTwo*sizeof(float3));
	cudaMalloc(&VGPU0, nTimesTwo*sizeof(float3));
	cudaMalloc(&FGPU0, nTimesTwo*sizeof(float3));

	cudaSetDevice(1);
	cudaMalloc(&MGPU1, nTimesTwo*sizeof(float));
	cudaMalloc(&PGPU1, nTimesTwo*sizeof(float3));
	cudaMalloc(&VGPU1, nTimesTwo*sizeof(float3));
	cudaMalloc(&FGPU1, nTimesTwo*sizeof(float3));

	Diameter = pow(H/G, 1.0/(LJQ - LJP));
	Radius = Diameter/2.0;
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;

	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0; break;
				}
			}
		}
		V[i] = F[i] = make_float3(0.0, 0.0, 0.0);
		M[i] = 1.0;
	}

	// I am going to need you to explain why I don't have to change this? Is
	// it because we are allocating extra memory on the gpu to make sure it can
	// hold the total. Okay, so after typing that I realized...
	cudaSetDevice(0);
	cudaMemcpyAsync(PGPU0, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(VGPU0, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(FGPU0, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(MGPU0, M, N*sizeof(float), cudaMemcpyHostToDevice);

	cudaSetDevice(1);
	cudaMemcpyAsync(PGPU1, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(VGPU1, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(FGPU1, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(MGPU1, M, N*sizeof(float), cudaMemcpyHostToDevice);
}

// Compute forces
__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int halfN, int n, int device)
{
	// Alright, change in x, change in y, change in z, distance, and distance before being squared
	float dx, dy, dz,d,d2;
	// Calculate forces
	float force_mag;

	// I will need some way to shift somehow in order for the second device to move over
	// to calculate the other half of the forces!
	int shift;
	
	// I only want to do this on the other gpu, so the first one I don't want to move!
	if(device == 0)
	{
		shift = 0;
	}
	else
	{
		shift = halfN;
	}
	// Global Index
	int i = threadIdx.x + blockDim.x*blockIdx.x + shift;
	
	// DON'T CHANGE THE MATH AND PRAY TO GOD THAT I HAVE THIS SET UP RIGHT PLS PLS PLS PLS PLS	
	if(i < n)
	{
		f[i].x = 0.0f;
		f[i].y = 0.0f;
		f[i].z = 0.0f;
		
		for(int j = 0; j < n; j++)
		{
			if(i != j)
			{
				dx = p[j].x-p[i].x;
				dy = p[j].y-p[i].y;
				dz = p[j].z-p[i].z;
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
				f[i].x += force_mag*dx/d;
				f[i].y += force_mag*dy/d;
				f[i].z += force_mag*dz/d;
			}
		}
	}
}

// Move bodies
__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int halfN, int n, int device)
{
	// Still need to shift
	int shift;
		
	if(device == 0)
	{
		shift = 0;
	}
	else
	{
		shift = halfN;
	}
	
	int i = threadIdx.x + blockDim.x*blockIdx.x + shift;
	
	if(i < n)
	{
		if(t == 0.0f)
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt/2.0f;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt/2.0f;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt/2.0f;
		}
		else
		{
			v[i].x += ((f[i].x-damp*v[i].x)/m[i])*dt;
			v[i].y += ((f[i].y-damp*v[i].y)/m[i])*dt;
			v[i].z += ((f[i].z-damp*v[i].z)/m[i])*dt;
		}

		p[i].x += v[i].x*dt;
		p[i].y += v[i].y*dt;
		p[i].z += v[i].z*dt;
	}
}

// Simulation loop
void nBody()
{
	int drawCount = 0; 
	float t = 0.0;
	float dt = 0.0001;
	int halfOfN = N/2;

	while(t < RUN_TIME)
	{
		// Update first half of bodies! Then move the first half around!
		cudaSetDevice(0);
		getForces<<<GridSize,BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, G, H, halfOfN, N, 0);
		moveBodies<<<GridSize,BlockSize>>>(PGPU0, VGPU0, FGPU0, MGPU0, Damp, dt, t, halfOfN, N, 0);

		// Update second half of bodies! Then move the second half around
		cudaSetDevice(1);
		getForces<<<GridSize,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, G, H, halfOfN, N, 1);
		moveBodies<<<GridSize,BlockSize>>>(PGPU1, VGPU1, FGPU1, MGPU1, Damp, dt, t, halfOfN, N, 1);

		// Before I can go any further, I am going to need to make sure both GPU 
		// 0 and 1 are ready to move on! Therefore, I will sync up both devices!
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);

		// I will need to copy all of the info around now! That way I can move everything around!
		cudaSetDevice(0);	
		cudaMemcpyAsync(PGPU1, PGPU0, halfOfN*sizeof(float3), cudaMemcpyDeviceToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Copying 2nd half of body positions updated on device 1 to devive 0.
		cudaSetDevice(1);	
		cudaMemcpyAsync(&PGPU0[halfOfN], &PGPU1[halfOfN], halfOfN*sizeof(float3), cudaMemcpyDeviceToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		
		// Sync up everything again so we don't break!
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);

		if(drawCount == DRAW_RATE) 
		{
			drawPicture();
			drawCount = 0;
		}
		t += dt;
		drawCount++;

		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaSetDevice(0);
		cudaDeviceSynchronize();
	}
}

// Free memory
void cleanUpRoom()
{
	free(P);
	free(V);
	free(F);
	free(M);
	
	cudaSetDevice(0);
	cudaFree(PGPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(VGPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(FGPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(MGPU0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaSetDevice(1);
	cudaFree(PGPU1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(VGPU1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(FGPU1);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(MGPU1);
	cudaErrorCheck(__FILE__, __LINE__);
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
	cleanUpRoom();
	return 0;
}