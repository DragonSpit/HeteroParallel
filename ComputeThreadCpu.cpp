#include <windows.h>
#include <iostream>
#include <fstream>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

#include "mkl_vsl.h"

#include "asyncNodeGenerator.h"
#include "CudaSupport.h"
#include "TimerCycleAccurateArray.h"

using namespace std;

extern int mklRandomFloatParallel_SkipAhead(float  * RngArray, size_t NumValues, unsigned int seed, int RngType, int NumCores);
extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];
extern bool gRunComputeWorkers;

WorkItemType workCPU;				// work item for CPU to do. This is to be setup before ghEventHaveWorkItemForCpu gets set to notify the CPU thread to start working on it
HANDLE ghEventHaveWorkItemForCpu;		// when asserted, work item for CPU is ready

// Thread waits to receive a single event for work to be performed, does the work, and sets a compute-done event when that work has been completed
DWORD WINAPI ThreadMultiCoreCpuCompute(LPVOID lpParam)
{
	UNREFERENCED_PARAMETER(lpParam);	// lpParam not used in this example
	DWORD dwWaitResult;

	while (gRunComputeWorkers)
	{
		//printf("ThreadMultiCoreCpuCompute %d waiting for MultiCore CPU work item...\n", GetCurrentThreadId());

		dwWaitResult = WaitForSingleObject(ghEventHaveWorkItemForCpu, INFINITE);	// forever wait for CPU work item

		switch (dwWaitResult)
		{
			// Event object was signaled
		case WAIT_OBJECT_0:
			//printf("ThreadMultiCoreCpuCompute %d received work item to do\n", GetCurrentThreadId());
			break;
			// An error occurred
		default:
			printf("ThreadMultiCoreCpuCompute Wait error (%d)\n", GetLastError());
			return 1;
		}

		// TODO: Do the CPU work requested in the work item
		unsigned int rngSeed = 2;
		int rngType = VSL_BRNG_PHILOX4X32X10;	// was VSL_BRNG_MCG59, which doesn't scale as well, seeming to have memory contension past 2 cores
		int numCores = 1;
		int rngResult = mklRandomFloatParallel_SkipAhead((float *)workCPU.HostResultPtr, workCPU.AmountOfWork, rngSeed, rngType, numCores);
		// Signal the associated event to indicate work item has been finished
		//printf("ThreadMultiCoreCpuCompute %d done with work item. Signaling dispatcher\n", GetCurrentThreadId());
		if (!SetEvent(ghEventsComputeDone[ComputeEngine::CPU]))	// Set one event to the signaled state
		{
			printf("SetEvent[%d] failed (%d)\n", ComputeEngine::CPU, GetLastError());
			return 2;
		}
	}
	return 0;
}
