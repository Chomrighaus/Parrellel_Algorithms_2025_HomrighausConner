// Name: Conner Homrighaus
// Optimizing nBody GPU code. 
// nvcc -use_fast_math HW21.cu -o temp -lglut -lm -lGLU -lGL

/*
	10,752 bodies
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 256
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
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float4 *PGPU; // instead of float3 *PGPU;
float3 *VGPU, *FGPU;
float *MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void freeAll();
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForces(float4*, float3*, float3*, float*, float, float, int);
__global__ void moveBodies(float4*, float3*, float3*, float*, float, float, float, int);
__global__ void initialBodyMover(float4*, float3*, float3*, float*, float, float, float, int);
void nBody();
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

void freeAll()
{
    free(M);
	free(P);
	free(V);
	free(F);
	cudaFree(PGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(VGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(FGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(MGPU);
	cudaErrorCheck(__FILE__, __LINE__);
}


void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		freeAll();
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    nBody();
    cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
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

    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
		
    Damp = 0.5;
    	
    M = (float*)malloc(N*sizeof(float));
    P = (float3*)malloc(N*sizeof(float3));
    V = (float3*)malloc(N*sizeof(float3));
    F = (float3*)malloc(N*sizeof(float3));
    	
    cudaMalloc(&MGPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&PGPU, N * sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
    	
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
	
	cudaMemcpyAsync(PGPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(VGPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(FGPU, F, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(MGPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing type s.\n");
}

__global__ void getForces(float4 *p, float3 *v, float3 *f, float *m, float g, float h, int n)
{
	extern __shared__ char shared[];
	float4* tile_p = (float4*)shared;
	float* tile_m = (float*)(tile_p + blockDim.x);

	float dx, dy, dz, d2;
	float force_mag;

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n)
	{
		float4 pi4 = p[i];
		float3 pi = make_float3(pi4.x, pi4.y, pi4.z);
		float mi = m[i];
		float3 fi = make_float3(0.0f, 0.0f, 0.0f);

		for (int tile = 0; tile < gridDim.x; tile++)
		{
			int j = tile * blockDim.x + threadIdx.x;

			if (j < n)
			{
				tile_p[threadIdx.x] = p[j];
				tile_m[threadIdx.x] = m[j];
			}
			else
			{
				tile_p[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				tile_m[threadIdx.x] = 0.0f;
			}

			__syncthreads();

			for (int k = 0; k < blockDim.x; k++)
			{
				int idx = tile * blockDim.x + k;
				if (idx >= n || idx == i) continue;

				float4 pj4 = tile_p[k];
				float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
				float mj = tile_m[k];

				dx = pj.x - pi.x;
				dy = pj.y - pi.y;
				dz = pj.z - pi.z;

				d2 = dx * dx + dy * dy + dz * dz + 1e-10f;
				float inv_d = rsqrtf(d2);
				float inv_d2 = inv_d * inv_d;
				float inv_d4 = inv_d2 * inv_d2;

				force_mag = (g * mi * mj) * inv_d2 - (h * mi * mj) * inv_d4;

				fi.x += force_mag * dx * inv_d;
				fi.y += force_mag * dy * inv_d;
				fi.z += force_mag * dz * inv_d;
			}

			__syncthreads();
		}

		f[i] = fi;
	}
}

__global__ void moveBodies(float4 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{
	extern __shared__ char shared[];
	float* tile_m = (float*)shared;
	float3* tile_f = (float3*)(tile_m + blockDim.x);
	float3* tile_v = (float3*)(tile_f + blockDim.x);

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n)
	{
		tile_m[threadIdx.x] = m[i];
		tile_f[threadIdx.x] = f[i];
		tile_v[threadIdx.x] = v[i];
	}
	__syncthreads();

	if (i < n)
	{
		float inv_m = 1.0f / tile_m[threadIdx.x];

		tile_v[threadIdx.x].x += ((tile_f[threadIdx.x].x - damp * tile_v[threadIdx.x].x) * inv_m) * dt;
		tile_v[threadIdx.x].y += ((tile_f[threadIdx.x].y - damp * tile_v[threadIdx.x].y) * inv_m) * dt;
		tile_v[threadIdx.x].z += ((tile_f[threadIdx.x].z - damp * tile_v[threadIdx.x].z) * inv_m) * dt;

		float4 pi4 = p[i];
		pi4.x += tile_v[threadIdx.x].x * dt;
		pi4.y += tile_v[threadIdx.x].y * dt;
		pi4.z += tile_v[threadIdx.x].z * dt;
		p[i] = pi4;
		v[i] = tile_v[threadIdx.x];
	}
}

__global__ void initialBodyMover(float4 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int n)
{
	extern __shared__ char shared[];
	float* tile_m = (float*)shared;
	float3* tile_f = (float3*)(tile_m + blockDim.x);
	float3* tile_v = (float3*)(tile_f + blockDim.x);

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n)
	{
		tile_m[threadIdx.x] = m[i];
		tile_f[threadIdx.x] = f[i];
		tile_v[threadIdx.x] = v[i];
	}
	__syncthreads();

	if (i < n)
	{
		float inv_m = 1.0f / tile_m[threadIdx.x];
		float half_dt = 0.5f * dt;

		tile_v[threadIdx.x].x += ((tile_f[threadIdx.x].x - damp * tile_v[threadIdx.x].x) * inv_m) * half_dt;
		tile_v[threadIdx.x].y += ((tile_f[threadIdx.x].y - damp * tile_v[threadIdx.x].y) * inv_m) * half_dt;
		tile_v[threadIdx.x].z += ((tile_f[threadIdx.x].z - damp * tile_v[threadIdx.x].z) * inv_m) * half_dt;

		float4 pi4 = p[i];
		pi4.x += tile_v[threadIdx.x].x * dt;
		pi4.y += tile_v[threadIdx.x].y * dt;
		pi4.z += tile_v[threadIdx.x].z * dt;
		p[i] = pi4;
		v[i] = tile_v[threadIdx.x];
	}
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;
	
	size_t sharedMemSize = BLOCK_SIZE * (sizeof(float3) + sizeof(float));
	size_t sharedMemSizeBodyMovers = BLOCK_SIZE * (sizeof(float) + 2 * sizeof(float3));
	
	getForces<<<GridSize,BlockSize, sharedMemSize>>>(PGPU, VGPU, FGPU, MGPU, G, H, N);
	cudaErrorCheck(__FILE__, __LINE__);
	initialBodyMover<<<GridSize, BlockSize, sharedMemSizeBodyMovers>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, t, N);
	cudaErrorCheck(__FILE__, __LINE__);

	t += dt;
	drawCount++;


	while(t < RUN_TIME)
	{
		getForces<<<GridSize,BlockSize, sharedMemSize>>>(PGPU, VGPU, FGPU, MGPU, G, H, N);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize,BlockSize, sharedMemSizeBodyMovers>>>(PGPU, VGPU, FGPU, MGPU, Damp, dt, t, N);
		cudaErrorCheck(__FILE__, __LINE__);

		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{	
				drawPicture();
			}
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
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




