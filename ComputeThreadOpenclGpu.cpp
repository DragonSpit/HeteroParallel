#include <windows.h>
#include <iostream>
#include <fstream>
#include <arrayfire.h>

#include "mkl_vsl.h"

#include "asyncNodeGenerator.h"
#include "ArrayFireSupport.h"
#include "TimerCycleAccurateArray.h"

using namespace std;

extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - MultiCoreCpu, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
extern void generateRandomFloatArray(af::array& randomArrayFloat, size_t numRandoms, float* hostDestArray);
extern void sortArray_ArrayFire(unsigned * inHostArray, size_t numItems, unsigned * outHostArray);
extern bool gRunComputeWorkers;

WorkItemType workOpenclGPU;			// work item for Cuda GPU to do. This is to be setup before ghEventHaveWorkItemForCudaGpu gets set to notify the CPU thread to start working on it
HANDLE ghEventHaveWorkItemForOpenclGpu;	// when asserted, work item for Cuda GPU is ready

OpenClGpuRngEncapsulation    * gOpenClRngSupport   = NULL;	// TODO: Make sure to delete it once done
OpenClGpuMemoryEncapsulation * gOpenClSourceMemory = NULL;	// TODO: Make sure to delete it once done
OpenClGpuMemoryEncapsulation * gOpenClResultMemory = NULL;	// TODO: Make sure to delete it once done

bool IsCompletedWorkItemSorted(unsigned * resultSortedArray_CPU, size_t resultArrayIndex_CPU, size_t NumOfItemsInWorkQuanta)
{
	for (size_t i = 0; i < NumOfItemsInWorkQuanta - 1; i++)
	{
		if (i < 2000)
			printf("%u\n", resultSortedArray_CPU[i]);
		if (resultSortedArray_CPU[i] > resultSortedArray_CPU[i + 1])
			return false;
	}
	return true;
}

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
			//printf("Executing generation of random work item\n");
// TODO: Switch away from using gOpenClResultMemory->m_gpu_memory and use workOpenclGPU.HostSourcePtr instead (needs tested)
			generateRandomFloatArray(gOpenClResultMemory->m_gpu_memory, workOpenclGPU.AmountOfWork, (float *)workOpenclGPU.HostResultPtr);
			break;
		case Sort:
			//printf("Executing sorting work item\n");
			sortArray_ArrayFire((unsigned *)workOpenclGPU.HostSourcePtr, workOpenclGPU.AmountOfWork, (unsigned *)workOpenclGPU.HostResultPtr);
			//if (!IsCompletedWorkItemSorted((unsigned *)workOpenclGPU.HostResultPtr, 0, workOpenclGPU.AmountOfWork))
			//	exit(-1);
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

