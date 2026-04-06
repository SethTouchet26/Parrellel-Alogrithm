// Name: Seth Touchet
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
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
#define GRAVITY 2.0
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

#define NUMBER_OF_SPHERES 3

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);

float r[NUMBER_OF_SPHERES], g[NUMBER_OF_SPHERES], b[NUMBER_OF_SPHERES];

float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES]; //The changes made to convert to an N-Body
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES];
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES];
float mass [NUMBER_OF_SPHERES];

// Function prototypes
void set_initial_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies();
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initial_conditions() //needing to just rewrite the code here to be clearer with the addition to the n-body.
{ 
	srand(time(NULL));
	float dx, dy, dz, separation;

	for(int i = 0; i <NUMBER_OF_SPHERES; i++)
	{
		int valid = 0;

		while(!valid)
		{
			valid = 1;

			px[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			py[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			pz[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	
			for(int j = 0; j < i; j++)
			{
		
				dx = px[i] - px[j];
				dy = py[i] - py[j];
				dz = pz[i] - pz[j];
				separation = sqrt(dx*dx + dy*dy + dz*dz);

				if(separation < DIAMETER)
				{
					valid = 0;
					break;
				}
			}
		}
		vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
		mass[i] = MASS;

		r[i] = (float)rand() / RAND_MAX;
		g[i] = (float)rand() / RAND_MAX;
		b[i] = (float)rand() / RAND_MAX;
	}
}

void Drawwirebox()
{		
	glColor3f (1.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glColor3f(r[i], g[i], b[i]);
		glutSolidSphere(radius, 20, 20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(px[i] > halfBoxLength)
		{
			px[i] = 2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		else if(px[i] < -halfBoxLength)
		{
			px[i] = -2.0*halfBoxLength - px[i];
			vx[i] = - vx[i];
		}
		
		if(py[i] > halfBoxLength)
		{
			py[i] = 2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
		else if(py[i] < -halfBoxLength)
		{
			py[i] = -2.0*halfBoxLength - py[i];
			vy[i] = - vy[i];
		}
				
		if(pz[i] > halfBoxLength)
		{
			pz[i] = 2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
		else if(pz[i] < -halfBoxLength)
		{
			pz[i] = -2.0*halfBoxLength - pz[i];
			vz[i] = - vz[i];
		}
	}
}

void get_forces()
{
	float dx,dy,dz,r,r2,forceMag;
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		fx[i] = fy[i] = fz[i] = 0.0;
	}
	for(int i = 0; i <NUMBER_OF_SPHERES; i++)
	{
		for(int j = i+1; j < NUMBER_OF_SPHERES; j++)
		{
			dx = px[j] - px[i];
			dy = py[j] - py[i];
			dz = pz[j] - pz[i];
						
			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);

			forceMag = mass[i] * mass[j] * GRAVITY/r2;
			
			if (r < DIAMETER) //Collision pushback
			{
				forceMag +=  SPHERE_PUSH_BACK_STRENGTH * PUSH_BACK_REDUCTION * (r - DIAMETER);
			}
		
			float fxij = forceMag * dx/r;
			float fyij = forceMag * dy/r;
			float fzij = forceMag * dz/r;
			fx[i] += fxij;
			fy[i] += fyij;
			fz[i] += fzij;

			fx[j] -= fxij;
			fy[j] -= fyij;
			fz[j] -= fzij;
		}
	}
}

void move_bodies()
{
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		vx[i] += 0.5*DT*(fx[i] - DAMP * vx[i]) /mass[i];
		vy[i] += 0.5*DT*(fy[i] - DAMP * vy[i]) /mass[i];
		vz[i] += 0.5*DT*(fz[i] - DAMP * vz[i]) /mass[i];

		px[i] += DT*vx[i];
		py[i] += DT*vy[i];
		pz[i] += DT*vz[i];
	
	}
	keep_in_box();
}

void nbody()
{	
	static int intitalized = 0;
	static int tdraw = 0;
	float  time = 0.0;
	if (!intitalized)
	{
		set_initial_conditions();
		intitalized = 1;
	}
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
		move_bodies();
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body Screen Saver");
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

	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutIdleFunc(nbody);

	glutMainLoop();
	return 0;
}/*Observation: */
