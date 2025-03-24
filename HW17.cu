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
#include <string.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT 0.0001
#define GRAVITY 0.1
#define DIAMETER 1.0
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define NUMBER_OF_SPHERES 10

// Global state
int paused = 0;
float simulationSpeed = 1.0;

// Prototypes 
void set_initial_conditions();
void get_forces();
void move_bodies(float);
void Display(void);
void reshape(int, int);
void handleKeypress(unsigned char key, int x, int y);
void drawText(const char* text, float x, float y);
void printMenu();
void nbody();


// Body structure
struct Body {
    float px, py, pz;
    float vx, vy, vz;
    float fx, fy, fz;
    float mass;
};
Body bodies[NUMBER_OF_SPHERES];


void set_initial_conditions() 
{
    srand(time(NULL));
    for (int i = 0; i < NUMBER_OF_SPHERES; ++i) {
        bodies[i].px = (LENGTH_OF_BOX - DIAMETER) * ((float)rand() / RAND_MAX) - LENGTH_OF_BOX / 2;
        bodies[i].py = (LENGTH_OF_BOX - DIAMETER) * ((float)rand() / RAND_MAX) - LENGTH_OF_BOX / 2;
        bodies[i].pz = (LENGTH_OF_BOX - DIAMETER) * ((float)rand() / RAND_MAX) - LENGTH_OF_BOX / 2;
        bodies[i].vx = ((float)rand() / RAND_MAX - 0.5) * 2;
        bodies[i].vy = ((float)rand() / RAND_MAX - 0.5) * 2;
        bodies[i].vz = ((float)rand() / RAND_MAX - 0.5) * 2;
        bodies[i].mass = 1.0;
    }
}

void get_forces() 
{
    for (int i = 0; i < NUMBER_OF_SPHERES; ++i) {
        bodies[i].fx = bodies[i].fy = bodies[i].fz = 0.0;
        for (int j = 0; j < NUMBER_OF_SPHERES; ++j) {
            if (i != j) {
                float dx = bodies[j].px - bodies[i].px;
                float dy = bodies[j].py - bodies[i].py;
                float dz = bodies[j].pz - bodies[i].pz;
                float r2 = dx * dx + dy * dy + dz * dz;
                float r = sqrt(r2);

                if (r > DIAMETER) {
                    float force = GRAVITY * bodies[i].mass * bodies[j].mass / r2;
                    bodies[i].fx += force * dx / r;
                    bodies[i].fy += force * dy / r;
                    bodies[i].fz += force * dz / r;
                }
            }
        }
    }
}

void move_bodies(float time) 
{
    for (int i = 0; i < NUMBER_OF_SPHERES; ++i) {
        bodies[i].vx += DT * simulationSpeed * bodies[i].fx / bodies[i].mass;
        bodies[i].vy += DT * simulationSpeed * bodies[i].fy / bodies[i].mass;
        bodies[i].vz += DT * simulationSpeed * bodies[i].fz / bodies[i].mass;

        bodies[i].px += DT * simulationSpeed * bodies[i].vx;
        bodies[i].py += DT * simulationSpeed * bodies[i].vy;
        bodies[i].pz += DT * simulationSpeed * bodies[i].vz;

        // Keep bodies within the box
        if (bodies[i].px > LENGTH_OF_BOX / 2 || bodies[i].px < -LENGTH_OF_BOX / 2) bodies[i].vx *= -1;
        if (bodies[i].py > LENGTH_OF_BOX / 2 || bodies[i].py < -LENGTH_OF_BOX / 2) bodies[i].vy *= -1;
        if (bodies[i].pz > LENGTH_OF_BOX / 2 || bodies[i].pz < -LENGTH_OF_BOX / 2) bodies[i].vz *= -1;
    }
}

void nbody() 
{
    // Calculate Forces
    for (int i = 0; i < NUMBER_OF_SPHERES; ++i) 
	{
        bodies[i].fx = bodies[i].fy = bodies[i].fz = 0.0;
        for (int j = 0; j < NUMBER_OF_SPHERES; ++j) 
		{
            if (i != j) 
			{
                float dx = bodies[j].px - bodies[i].px;
                float dy = bodies[j].py - bodies[i].py;
                float dz = bodies[j].pz - bodies[i].pz;
                float r2 = dx * dx + dy * dy + dz * dz;
                float r = sqrt(r2);

                if (r > DIAMETER) 
				{
                    float force = GRAVITY * bodies[i].mass * bodies[j].mass / r2;
                    bodies[i].fx += force * dx / r;
                    bodies[i].fy += force * dy / r;
                    bodies[i].fz += force * dz / r;
                }
            }
        }
    }
}

void Display() {
    if (!paused) nbody();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);

    for (int i = 0; i < NUMBER_OF_SPHERES; ++i) {
        glPushMatrix();
        glTranslatef(bodies[i].px, bodies[i].py, bodies[i].pz);
        glutSolidSphere(DIAMETER / 2.0, 20, 20);
        glPopMatrix();
    }

    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape(int w, int h) 
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (double)w / (double)h, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

// This should drawText to show the options
void drawText(const char* text, float x, float y) 
{
    glRasterPos2f(x, y);
    for (int i = 0; text[i] != '\0'; i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, text[i]);
}

// This will handle all key presses
void handleKeypress(unsigned char key, int x, int y) {
    switch (key) {
		case 'p':
        case 'P':
            paused = !paused;
            break;
        case '+':
            simulationSpeed *= 1.5;
            break;
        case '-':
            simulationSpeed *= 0.75;
            break;
		case 'r':
        case 'R':
            set_initial_conditions();
            break;
        case 27: // Escape key
            exit(0);
    }
}

void printMenu() {
    printf("\n=============== N-Body Simulation Controls ===============\n");
    printf("P / p - Pause/Resume the simulation\n");
    printf("+     - Increase simulation speed\n");
    printf("-     - Decrease simulation speed\n");
    printf("R / r - Reset simulation\n");
    printf("Esc   - Quit simulation\n");
    printf("Ctrl+C - Forcefully exit simulation (if needed, and must be used on terminal)\n");
    printf("==========================================================\n\n");
}

int main(int argc, char** argv) 
{
	printMenu();
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize,YWindowSize);
    glutInitWindowPosition(0,0);
    glutCreateWindow("N-Body Simulation");
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnable(GL_DEPTH_TEST);
    glutDisplayFunc(Display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(handleKeypress);
    set_initial_conditions();
    glutMainLoop();
    return 0;
}
