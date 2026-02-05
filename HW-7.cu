
// Name: Seth Touchet
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
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
dim3 BlockSize(16,16,1);
dim3 GridSize((WindowWidth + BlockSize.x-1)/ BlockSize.x, (WindowHeight + BlockSize.y - 1)/ BlockSize.y,1);
// add the blocksize and gridsize

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;



float *gpuPixels = NULL;
int gpuIndex = 0;

// Function prototypes
void cudaErrorCheck(const char*, int);


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
#define CUDA_CHECK() cudaErrorCheck(__FILE__, __LINE__)

float escapeOrNotColor(float x, float y)

{
	float value = gpuPixels[gpuIndex];
	gpuIndex++;
	return value;
}

__device__ float escapeDevice(float x, float y) // just to be sure its for CUDA 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0f * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0f);
	}
	else
	{
		return(1.0f);
	}
}
//CUDA kernel
__global__ void juliaKernel(float *pixels,int width, int height,float xmin, float ymin,float stepX, float stepY) //Just so CUDA does not freak out (had some problems with the escapeorNotColor part of the code)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) return;

	int index = (iy * width + ix) * 3;
    float x = xmin + ix * stepX;
    float y = ymin + iy * stepY;

    float color = escapeDevice(x, y);
    
    pixels[index]     = color; // Red
    pixels[index + 1] = 0.0f;  // Green
    pixels[index + 2] = 0.0f;  // Blue
}


void display(void) 
{ 
	float *pixels; 
	float *d_pixels;
	float x, y, stepSizeX, stepSizeY;
	int k;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	int size = WindowWidth*WindowHeight*3*sizeof(float);
	pixels = (float *)malloc(size);
	gpuPixels = (float *)malloc(size);

	cudaMalloc((void **)&d_pixels, size); //to check with the malloc and make sure nothing goes wrong when intregrating with the CUDA additions
    CUDA_CHECK();

	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	juliaKernel<<<GridSize, BlockSize>>>(d_pixels, WindowWidth, WindowHeight, XMin, YMin, stepSizeX, stepSizeY);

	cudaDeviceSynchronize();
	cudaMemcpy(gpuPixels, d_pixels, size, cudaMemcpyDeviceToHost);
	gpuIndex = 0;

	k=0;
	y = YMin;
	while(y < YMax) 
	{
		x = XMin;
		while(x < XMax) 
		{
			pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
			pixels[k+1] = 0.0f; 	//Green off
			pixels[k+2] = 0.0f;	//Blue off
			k=k+3;			//Skip to next pixel (3 float jump)
			x += stepSizeX;
		}
		y += stepSizeY;
	}
	
	cudaDeviceSynchronize();
    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK();

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 

	cudaFree(d_pixels); //to clear the device pixels and the pixels on screen
    free(pixels);
	free(gpuPixels);
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--with--Julia");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
