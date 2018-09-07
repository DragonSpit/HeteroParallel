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

extern void GenerateRandFloatCuda(float *devMemPtr, float *sysMemPtr, curandGenerator_t& prngGPU, size_t numRandoms, bool verify, curandGenerator_t& prngCPU);
extern void copyCudaToSystemMemory(void *systemMemPtr, void *cudaMemPtr, size_t numBytes);
extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - MultiCoreCpu, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
extern bool gRunComputeWorkers;

WorkItemType workCudaGPU;			// work item for Cuda GPU to do. This is to be setup before ghEventHaveWorkItemForCudaGpu gets set to notify the CPU thread to start working on it
HANDLE ghEventHaveWorkItemForCudaGpu;	// when asserted, work item for Cuda GPU is ready

CudaRngEncapsulation    * gCudaRngSupport    = NULL;	// TODO: Make sure to delete it once done
CudaMemoryEncapsulation * gCudaMemorySupport = NULL;	// TODO: Make sure to delete it once done

DWORD WINAPI ThreadCudaGpuCompute(LPVOID lpParam)
{
	UNREFERENCED_PARAMETER(lpParam);	// lpParam not used in this example
	DWORD dwWaitResult;

	while (gRunComputeWorkers)
	{
		//printf("ThreadCudaGpuCompute %d waiting for CUDA GPU work item...\n", GetCurrentThreadId());

		dwWaitResult = WaitForSingleObject(ghEventHaveWorkItemForCudaGpu, INFINITE);	// forever wait for GPU work item

		switch (dwWaitResult)
		{
			// Event object was signaled
		case WAIT_OBJECT_0:
			//printf("ThreadCudaGpuCompute %d received work item to do\n", GetCurrentThreadId());
			break;
			// An error occurred
		default:
			printf("ThreadCudaGpuCompute Wait error (%d)\n", GetLastError());
			return 1;
		}

		//Sleep(2000);	// for debug, to slow GPU down artificially for each work item
		bool verify = false;
		GenerateRandFloatCuda((float *)workCudaGPU.DeviceResultPtr, (float *)workCudaGPU.HostResultPtr, gCudaRngSupport->prngGPU, workCudaGPU.AmountOfWork, verify, gCudaRngSupport->prngCPU);

		// Signal the associated event to indicate work item has been finished
		//printf("ThreadCudaGpuCompute %d done with work item. Signaling dispatcher\n", GetCurrentThreadId());
		if (!SetEvent(ghEventsComputeDone[ComputeEngine::CUDA_GPU]))	// Set done event for CudaGpu to signaled state
		{
			printf("SetEvent[%d] failed (%d)\n", ComputeEngine::CUDA_GPU, GetLastError());
			return 2;
		}
	}
	return 0;
}

