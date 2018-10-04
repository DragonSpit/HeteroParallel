#include <windows.h>
#include <iostream>
#include <fstream>
#include <arrayfire.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

#include "mkl_vsl.h"

#include "asyncNodeGenerator.h"
#include "ArrayFireSupport.h"
#include "TimerCycleAccurateArray.h"

using namespace std;

//extern void GenerateRandFloatCuda(float *devMemPtr, float *sysMemPtr, curandGenerator_t& prngGPU, size_t numRandoms, bool verify, curandGenerator_t& prngCPU);
//extern void copyCudaToSystemMemory(void *systemMemPtr, void *cudaMemPtr, size_t numBytes);
extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - MultiCoreCpu, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
extern void generateRandomFloatArray(af::array& randomArrayFloat, size_t numRandoms, float* hostDestArray);
extern void sortArray(unsigned * inHostArray, size_t numItems, unsigned * outHostArray);
extern bool gRunComputeWorkers;

WorkItemType workOpenclGPU;			// work item for Cuda GPU to do. This is to be setup before ghEventHaveWorkItemForCudaGpu gets set to notify the CPU thread to start working on it
HANDLE ghEventHaveWorkItemForOpenclGpu;	// when asserted, work item for Cuda GPU is ready

OpenClGpuRngEncapsulation    * gOpenClRngSupport   = NULL;	// TODO: Make sure to delete it once done
OpenClGpuMemoryEncapsulation * gOpenClSourceMemory = NULL;	// TODO: Make sure to delete it once done
OpenClGpuMemoryEncapsulation * gOpenClResultMemory = NULL;	// TODO: Make sure to delete it once done

DWORD WINAPI ThreadOpenclGpuCompute(LPVOID lpParam)
{
	UNREFERENCED_PARAMETER(lpParam);	// lpParam not used in this example
	DWORD dwWaitResult;

	while (gRunComputeWorkers)
	{
		//printf("ThreadOpenclGpuCompute %d waiting for OpenCL GPU work item...\n", GetCurrentThreadId());

		dwWaitResult = WaitForSingleObject(ghEventHaveWorkItemForOpenclGpu, INFINITE);	// forever wait for GPU work item

		switch (dwWaitResult)
		{
			// Event object was signaled
		case WAIT_OBJECT_0:
			//printf("ThreadOpenclGpuCompute %d received work item to do\n", GetCurrentThreadId());
			break;
			// An error occurred
		default:
			printf("ThreadOpenclGpuCompute: Wait error (%d)\n", GetLastError());
			return 1;
		}

		//Sleep(2000);	// for debug, to slow GPU down artificially for each work item
		bool verify = false;
		switch (workOpenclGPU.TypeOfWork)
		{
		case GenerateRandoms:
			generateRandomFloatArray(gOpenClResultMemory->m_gpu_memory, workOpenclGPU.AmountOfWork, (float *)workOpenclGPU.HostResultPtr);
			break;
		case Sort:
			sortArray((unsigned *)workOpenclGPU.HostSourcePtr, workOpenclGPU.AmountOfWork, (unsigned *)workOpenclGPU.HostResultPtr);
			break;
		default:
			printf("ThreadOpenclGpuCompute: Don't know how to do this kind of work (%d)\n", workOpenclGPU.TypeOfWork);
			return 1;
		}

		// Signal the associated event to indicate work item has been finished
		//printf("ThreadOpenclGpuCompute %d done with work item. Signaling dispatcher\n", GetCurrentThreadId());
		if (!SetEvent(ghEventsComputeDone[ComputeEngine::OPENCL_GPU]))	// Set done event for OpenclGpu to signaled state
		{
			printf("SetEvent[%d] failed (%d)\n", ComputeEngine::OPENCL_GPU, GetLastError());
			return 2;
		}
	}
	return 0;
}

