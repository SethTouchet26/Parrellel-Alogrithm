// Name: Seth Touchet
// Device query
// nvcc E_DeviceQuery.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count); //Shows how many GPUs that the computer has on standby, usually just one.
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);

		printf(" ---General Information for device %d ---\n", i);

		printf("Name: %s\n", prop.name); // Would display the GPU model.
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); // This determines what CUDA features the GPU supports.
		printf("Clock rate: %d\n", prop.clockRate); // Core frequency of the GPU.

		printf("Device copy overlap: "); // This can copy memory and execute the kernels at the same time.
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");

		printf("Kernel execution timeout : "); // Show to the consumer GPUs kill long kernels
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");

		printf(" ---Memory Information for device %d ---\n", i);

		printf("Total global mem: %ld\n", prop.totalGlobalMem); // This would show the Total GPU DRAM that is available.
		printf("Total constant Mem: %ld\n", prop.totalConstMem); // Showing the small cached memory for read-only constants.
		printf("Max mem pitch: %ld\n", prop.memPitch); // Displays the maximum width of 2D allocations.
		printf("Texture Alignment: %ld\n", prop.textureAlignment); // This would be alignment requirement for texture memory.

		printf(" ---MP Information for device %d ---\n", i);

		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU

		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock); // This would show the amount of shared memory available per multiprocessor on the GPU.
		printf("Registers per mp: %d\n", prop.regsPerBlock); // This register the file size available per block.
		printf("Threads in warp: %d\n", prop.warpSize); // Highlights the number of threads that is being executed together.
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); // Would display the upper limit (mainly 1024).
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // Shows the maximum number of blocks in the grid dimensions.
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // Would display the maximum size of the grid within the three dimensions.

		printf("\n");
	}	
	return(0);
}
