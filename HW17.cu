// Name: Conner Homrighaus
// Two body problem
// nvcc HW17.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user-friendly.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define NUMBER_OF_SPHERES 10

// Globals
typedef struct 
{
    float px, py, pz, vx, vy, vz, fx, fy, fz, mass;
    float r, g, b;  // Color components
} Sphere;

Sphere spheres[NUMBER_OF_SPHERES];

// Function prototypes
void set_initial_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initial_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		spheres[i].px = (LENGTH_OF_BOX - DIAMETER) * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].py = (LENGTH_OF_BOX - DIAMETER) * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].pz = (LENGTH_OF_BOX - DIAMETER) * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].vx = 2.0 * MAX_VELOCITY * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].vy = 2.0 * MAX_VELOCITY * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].vz = 2.0 * MAX_VELOCITY * ((float)rand()/RAND_MAX - 0.5);
		spheres[i].mass = 1.0;
        	spheres[i].r = (float)rand() / RAND_MAX;
        	spheres[i].g = (float)rand() / RAND_MAX;
        	spheres[i].b = (float)rand() / RAND_MAX;
	}
}

void keep_in_box(Sphere *s)
{
	float halfBox = (LENGTH_OF_BOX - DIAMETER) / 2.0;

	if(s->px > halfBox || s->px < -halfBox) s->vx = -s->vx;
	if(s->py > halfBox || s->py < -halfBox) s->vy = -s->vy;
	if(s->pz > halfBox || s->pz < -halfBox) s->vz = -s->vz;
}

void get_forces()
{
	float dx, dy, dz, dist, dist2, forceMag;
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		spheres[i].fx = spheres[i].fy = spheres[i].fz = 0.0;
		
		for (int j = 0; j < NUMBER_OF_SPHERES; j++)
		{
			if (i == j) continue;

			dx = spheres[j].px - spheres[i].px;
			dy = spheres[j].py - spheres[i].py;
			dz = spheres[j].pz - spheres[i].pz;

			dist2 = dx*dx + dy*dy + dz*dz;
			dist = sqrt(dist2);

			if (dist < DIAMETER) dist = DIAMETER;

			forceMag = GRAVITY * spheres[i].mass * spheres[j].mass / (dist2);
			spheres[i].fx += forceMag * dx / dist;
			spheres[i].fy += forceMag * dy / dist;
			spheres[i].fz += forceMag * dz / dist;
		}
	}
}

void move_bodies(float time)
{
	for (int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		spheres[i].vx += DT * spheres[i].fx / spheres[i].mass;
		spheres[i].vy += DT * spheres[i].fy / spheres[i].mass;
		spheres[i].vz += DT * spheres[i].fz / spheres[i].mass;
		
		spheres[i].px += DT * spheres[i].vx;
		spheres[i].py += DT * spheres[i].vy;
		spheres[i].pz += DT * spheres[i].vz;

		keep_in_box(&spheres[i]);
	}
}

void nbody()
{	
	set_initial_conditions();
	draw_picture();
	
	float time = 0.0;
	while (time < STOP_TIME)
	{
		get_forces();
		move_bodies(time);
		draw_picture();
		time += DT;
	}
}

void Drawwirebox()
{
    GLfloat white[] = {1.0, 1.0, 1.0, 1.0};  // White color for the box
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white);
    
    glColor3f(1.0, 1.0, 1.0); // In case lighting is off
    glutWireCube(LENGTH_OF_BOX);
}

void draw_picture()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Drawwirebox();
    
    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        glPushMatrix();
        GLfloat sphere_color[] = { spheres[i].r, spheres[i].g, spheres[i].b, 1.0 };
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, sphere_color);
        glTranslatef(spheres[i].px, spheres[i].py, spheres[i].pz);
        glutSolidSphere(DIAMETER/2.0, 20, 20);
        glPopMatrix();
    }
    
    glutSwapBuffers();
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLfloat light_pos[] = {0.0, 0.0, 10.0, 1.0};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	nbody();
    draw_picture();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) w / (GLfloat) h, 1.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(XWindowSize, YWindowSize);
	glutCreateWindow("N-Body Simulation");
	
	glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
    
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	
	glutMainLoop();
	return 0;
}
