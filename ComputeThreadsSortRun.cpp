#if 1
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
#include "ArrayFireSupport.h"
#include "TimerCycleAccurateArray.h"
#include <arrayfire.h>

using namespace std;

extern void copyCudaToSystemMemory(void *systemMemPtr, void *cudaMemPtr, size_t numBytes);

extern WorkItemType workCPU;					// work item for CPU to do. This is to be setup before ghEventHaveWorkItemForCpu gets set to notify the CPU thread to start working on it
extern HANDLE ghEventHaveWorkItemForCpu;		// when asserted, work item for CPU      is ready
extern WorkItemType workCudaGPU;				// work item for Cuda GPU to do. This is to be setup before ghEventHaveWorkItemForCudaGpu gets set to notify the CPU thread to start working on it
extern HANDLE ghEventHaveWorkItemForCudaGpu;	// when asserted, work item for Cuda GPU is ready
extern WorkItemType workOpenclGPU;
extern HANDLE ghEventHaveWorkItemForOpenclGpu;

extern CudaRngEncapsulation			* gCudaRngSupport;			// TODO: Make sure to delete it once done
extern OpenClGpuRngEncapsulation    * gOpenClRngSupport;		// TODO: Make sure to delete it once done
extern CudaMemoryEncapsulation		* gCudaSourceMemory;		// TODO: Make sure to delete it once done
extern CudaMemoryEncapsulation		* gCudaResultMemory;		// TODO: Make sure to delete it once done
extern OpenClGpuMemoryEncapsulation	* gOpenClSourceMemory;		// TODO: Make sure to delete it once done
extern OpenClGpuMemoryEncapsulation	* gOpenClResultMemory;		// TODO: Make sure to delete it once done

extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - CPU, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
extern bool   gRunComputeWorkers;

// TODO: Make generic to handle any data type to be sorted
// TODO: Handle in-place sorting and stable sorting
int CpuGenerateSortWork(SortToDo & sortSpec, const size_t & NumOfItemsInWorkQuanta,
						unsigned long * sourceArray_CPU, size_t & sourceArrayIndex_CPU,
						unsigned long * resultArray_CPU, size_t & resultArrayIndex_CPU, size_t & inputWorkIndex)
{
	if (sortSpec.CPU.allowedToWork &&
		(sortSpec.resultDestination == ResultInCpuMemory     || sortSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CPU is allowed to do work, it should do work
		 sortSpec.resultDestination == ResultInCudaGpuMemory || sortSpec.resultDestination == ResultInOpenclGpuMemory)) {
		printf("CPU work item being generated\n");
		workCPU.TypeOfWork = Sort;
		workCPU.ForWhichWorker = ComputeEngine::CPU;
		workCPU.AmountOfWork  = NumOfItemsInWorkQuanta;
		workCPU.HostSourcePtr = (char *)(&(sourceArray_CPU[sourceArrayIndex_CPU]));
		workCPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
		workCudaGPU.DeviceResultPtr = NULL;
		printf("CPU work item: amountOfWork = %zd at CPU source memory address %p and destination address %p\n", workCPU.AmountOfWork, workCPU.HostSourcePtr, workCPU.HostResultPtr);
		printf("Event set for work item for MultiCore CPU\n");
		if (!SetEvent(ghEventHaveWorkItemForCpu))	// signal that CPU has a work item to work on
		{
			printf("SetEvent ghEventWorkForCpu failed (%d)\n", GetLastError());
			return -5;
		}
		sourceArrayIndex_CPU += NumOfItemsInWorkQuanta;
		resultArrayIndex_CPU += NumOfItemsInWorkQuanta;
		inputWorkIndex++;
	}
	return 0;
}

int CudaGpuGenerateSortWork(SortToDo & sortSpec, const size_t & NumOfItemsInWorkQuanta,
							unsigned long * sourceArray_GPU, size_t & sourceArrayIndex_GPU, unsigned long * sourceArray_CPU, size_t & sourceArrayIndex_CPU,
							unsigned long * resultArray_GPU, size_t & resultArrayIndex_GPU, unsigned long * resultArray_CPU, size_t & resultArrayIndex_CPU,
							size_t & inputWorkIndex)
{
	if (sortSpec.CudaGPU.allowedToWork &&
		(sortSpec.resultDestination == ResultInCpuMemory     || sortSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
		 sortSpec.resultDestination == ResultInCudaGpuMemory || sortSpec.resultDestination == ResultInOpenclGpuMemory)) {
		if ((resultArrayIndex_GPU + NumOfItemsInWorkQuanta) < sortSpec.CudaGPU.maxElements) {
			printf("CudaGPU work item being generated\n");
			workCPU.TypeOfWork = Sort;
			workCudaGPU.ForWhichWorker  = ComputeEngine::CUDA_GPU;
			workCudaGPU.AmountOfWork    = NumOfItemsInWorkQuanta;
			workCudaGPU.DeviceSourcePtr = (char *)(&(sourceArray_GPU[sourceArrayIndex_GPU]));	// device memory is always used
			workCudaGPU.DeviceResultPtr = (char *)(&(resultArray_GPU[resultArrayIndex_GPU]));
			if (sourceArray_CPU != NULL) {
				workCudaGPU.HostSourcePtr = (char *)(&(sourceArray_CPU[sourceArrayIndex_CPU]));
				sourceArrayIndex_CPU += NumOfItemsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
			}
			if (sortSpec.resultDestination == ResultInCpuMemory) {
				workCudaGPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
				resultArrayIndex_CPU += NumOfItemsInWorkQuanta;
				// don't advance GPU array index, since we reuse the same result GPU array
			}
			else if (sortSpec.resultDestination == ResultInEachDevicesMemory || sortSpec.resultDestination == ResultInCudaGpuMemory) {
				workCudaGPU.HostResultPtr = NULL;
				resultArrayIndex_GPU += NumOfItemsInWorkQuanta;
				// don't advance CPU array index
			}
			printf("Cuda GPU work item: amountOfWork = %zd at GPU memory address %p\n", workCudaGPU.AmountOfWork, workCudaGPU.DeviceResultPtr);
			printf("Event set for work item for CUDA GPU\n");
			if (!SetEvent(ghEventHaveWorkItemForCudaGpu))		// signal that CudaGpu has a work item to work on
			{
				printf("SetEvent ghEventHaveWorkItemForCudaGpu failed (%d)\n", GetLastError());
				return -6;
			}
			inputWorkIndex++;
		}
	}
	return 0;
}

int OpenclGpuGenerateSortWork(SortToDo & sortSpec, const size_t & NumOfItemsInWorkQuanta,
								unsigned long * sourceArray_GPU, size_t & sourceArrayIndex_GPU, unsigned long * sourceArray_CPU, size_t & sourceArrayIndex_CPU,
								unsigned long * resultArray_GPU, size_t & resultArrayIndex_GPU, unsigned long * resultArray_CPU, size_t & resultArrayIndex_CPU,
								size_t & inputWorkIndex)
{
	if (sortSpec.OpenclGPU.allowedToWork &&
		(sortSpec.resultDestination == ResultInCpuMemory     || sortSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
		 sortSpec.resultDestination == ResultInCudaGpuMemory || sortSpec.resultDestination == ResultInOpenclGpuMemory)) {
		if ((resultArrayIndex_GPU + NumOfItemsInWorkQuanta) < sortSpec.OpenclGPU.maxElements) {
			printf("OpenclGPU work item being generated\n");
			workCPU.TypeOfWork = Sort;
			workOpenclGPU.ForWhichWorker = ComputeEngine::OPENCL_GPU;
			workOpenclGPU.AmountOfWork  = NumOfItemsInWorkQuanta;
			//workOpenclGPU.DeviceSourcePtr = (char *)(&(sourceArray_GPU[sourceArrayIndex_GPU]));	// device memory is always used
			//workOpenclGPU.DeviceResultPtr = (char *)(&(resultArray_GPU[resultArrayIndex_GPU]));
			if (sourceArray_CPU != NULL) {
				workOpenclGPU.HostSourcePtr = (char *)(&(sourceArray_CPU[sourceArrayIndex_CPU]));
				sourceArrayIndex_CPU += NumOfItemsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
			}
			if (sortSpec.resultDestination == ResultInCpuMemory) {
				workOpenclGPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
				resultArrayIndex_CPU += NumOfItemsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
				// don't advance GPU array index, since we reuse the same result array
			}
			else if (sortSpec.resultDestination == ResultInEachDevicesMemory || sortSpec.resultDestination == ResultInOpenclGpuMemory) {
				workOpenclGPU.HostResultPtr = NULL;
				resultArrayIndex_GPU += NumOfItemsInWorkQuanta;
				// don't advance CPU array index
			}
			printf("OpenCL GPU work item: amountOfWork = %zd at GPU memory address %p\n", workOpenclGPU.AmountOfWork, workOpenclGPU.DeviceResultPtr);
			printf("Event set for work item for OpenCL GPU\n");
			if (!SetEvent(ghEventHaveWorkItemForOpenclGpu))		// signal that OpenclGpu has a work item to work on
			{
				printf("SetEvent ghEventHaveWorkItemForOpenclGpu failed (%d)\n", GetLastError());
				return -6;
			}
			inputWorkIndex++;
		}
	}
	return 0;
}

bool IsCompletedWorkItemSorted(unsigned long * resultSortedArray_CPU, size_t resultArrayIndex_CPU, size_t NumOfItemsInWorkQuanta)
{
	for (size_t i = 0; i < NumOfItemsInWorkQuanta - 1; i++)
	{
		//printf("%u\n", resultSortedArray_CPU[i]);
		if (resultSortedArray_CPU[i] > resultSortedArray_CPU[i + 1])
			return false;
	}
	return true;
}
// TODO: Capture the pattern of load balancing any algorithm, possibly using a template to load balance anything.
// numTimes is needed to see if running the first time is slower than running subsequent times, as this is commonly the case due to OS paging and CPU caching
int runLoadBalancerSortThread(SortToDo& sortSpec, ofstream& benchmarkFile, unsigned numTimes)
{
	printf("runLoadBalancerSortThread entering\n");
	size_t NumOfItemsToSort = sortSpec.totalItemsToSort;
	size_t NumOfItemsInWorkQuanta = sortSpec.CPU.workQuanta;		// TODO: Need to separate CPU and GPU workQuanta, and handle them being different
																	// TODO: Fix the problem with the case of asking the CudaGPU to generate more randoms that can fit into it's memory, but no other computational units are helping to generate more
																	// TODO: One possible way to do this is to pre-determine the NumOfWorkItems and shrink it in case there is not enough memory between all of the generators
																	// TODO: Another way is to create a method that takes genSpec as input and outputs all of the needed setup variables with their values for the rest of the code to use
	// Figure out how many work items to generate
	// TODO: Currently, this is static, but eventually should be dynamic (within the work generation loop) as work quanta will be different for each computational unit and may even be dynamically sized
	size_t NumOfWorkItems = NumOfWorkItems = NumOfItemsToSort / NumOfItemsInWorkQuanta;
	if (sortSpec.resultDestination == ResultInCudaGpuMemory && !sortSpec.CPU.allowedToWork && !sortSpec.OpenclGPU.allowedToWork && !sortSpec.FpgaGPU.allowedToWork)	// only CudaGPU is working
		NumOfWorkItems = __min(sortSpec.CudaGPU.maxElements, NumOfItemsToSort) / sortSpec.CudaGPU.workQuanta;
	else if (sortSpec.resultDestination == ResultInCpuMemory)
		NumOfWorkItems = NumOfWorkItems = NumOfItemsToSort / NumOfItemsInWorkQuanta;
	printf("runLoadBalancerSortThread #1\n");

	//TODO: need source device memory!
	unsigned long *sourceUnsortedArray_CudaGPU = (unsigned long *)(gCudaSourceMemory != NULL ? gCudaSourceMemory->m_gpu_memory : NULL);
	unsigned long *resultSortedArray_CudaGPU   = (unsigned long *)(gCudaSourceMemory != NULL ? gCudaResultMemory->m_gpu_memory : NULL);
	printf("runLoadBalancerSortThread #2\n");
	// TODO: OpenCL memory for now needs to be host memory, as we are getting sorting to work from host to host memory first. Plus, ArrayFire has it's own array data type in GPU memory instead of using host mapped memory
	unsigned long *sourceUnsortedArray_OpenClGPU = NULL;
	unsigned long *resultSortedArray_OpenClGPU   = NULL;
	unsigned long *sourceUnsortedArray_CPU     = (unsigned long *)sortSpec.Unsorted.CPU.Buffer;
	unsigned long *resultSortedArray_CPU       = (unsigned long *)sortSpec.Sorted.CPU.Buffer;

	for (unsigned numRuns = 0; numRuns < numTimes; numRuns++)
	{
		printf("runLoadBalancerSortThread: run %d\n", numRuns);
		TimerCycleAccurateArray	timer;
		timer.reset();
		timer.timeStamp();
		// Start each worker in the graph, once the graph has been constructructed
		// TODO: Need to handle less work than enough for each of the worker type (e.g. 1.5xWorkQuanta randoms, 0.2xWorkQuanta randoms, with two available workers)
		size_t sourceArrayIndex_CPU = 0;
		size_t resultArrayIndex_CPU = 0;
		size_t sourceArrayIndex_CudaGPU = 0;
		size_t resultArrayIndex_CudaGPU = 0;
		size_t sourceArrayIndex_OpenClGPU = 0;
		size_t resultArrayIndex_OpenClGPU = 0;
		size_t inputWorkIndex = 0;
		if (NumOfWorkItems > 0) {	// CPU work item
			// TODO: Consider all combinations of where the randoms end up and who is allowed to help generate them. Is there a way to handle them in a general way (flags)?
			int returnValue = CpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta, sourceUnsortedArray_CPU, sourceArrayIndex_CPU, resultSortedArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 1) {	// CudaGpu work item
			int returnValue = CudaGpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta,
													  sourceUnsortedArray_CudaGPU, sourceArrayIndex_CudaGPU, sourceUnsortedArray_CPU, sourceArrayIndex_CPU,
													  resultSortedArray_CudaGPU,   resultArrayIndex_CudaGPU, resultSortedArray_CPU,   resultArrayIndex_CPU,
													  inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 2) {	// OpenclGpu work item
			int returnValue = OpenclGpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta,
														sourceUnsortedArray_OpenClGPU, sourceArrayIndex_OpenClGPU, sourceUnsortedArray_CPU, sourceArrayIndex_CPU,
														resultSortedArray_OpenClGPU,   resultArrayIndex_OpenClGPU, resultSortedArray_CPU,   resultArrayIndex_CPU,
														inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}

		DWORD dwEvent;
		unsigned numCpuWorkItemsDone = 0;
		unsigned numCudaGpuWorkItemsDone = 0;
		unsigned numOpenclGpuWorkItemsDone = 0;

		for (size_t outputWorkIndex = 0; outputWorkIndex < NumOfWorkItems; )
		{
			// Wait for the threads to signal one of the done event objects - returns an index of the event triggered
			dwEvent = WaitForMultipleObjects(
				NumComputeDoneEvents,   // number of objects in array
				ghEventsComputeDone,    // array of objects
				FALSE,                  // wait for any object
				10000);                 // 10-second wait       (TODO! What should this timeout be? Or, should we wait forever?)

			// The return value indicates which event is signaled
			switch (dwEvent)
			{
				// ghEventsComputeDone[0] (CPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + CPU:
				printf("ghEventsComputeDone CPU event was signaled.\n");
				if (!IsCompletedWorkItemSorted(resultSortedArray_CPU, resultArrayIndex_CPU - NumOfItemsInWorkQuanta, NumOfItemsInWorkQuanta))
					exit(-1);
				if (inputWorkIndex < NumOfWorkItems)	// Create new work item for CPU
				{
					int returnValue = CpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta, sourceUnsortedArray_CPU, sourceArrayIndex_CPU, resultSortedArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
					if (returnValue != 0) return returnValue;
				}
				numCpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[1] (CUDA GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + CUDA_GPU:
				printf("ghEventsComputeDone CUDA GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					int returnValue = CudaGpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta,
															  sourceUnsortedArray_CudaGPU, sourceArrayIndex_CudaGPU, sourceUnsortedArray_CPU, sourceArrayIndex_CPU,
															  resultSortedArray_CudaGPU, resultArrayIndex_CudaGPU, resultSortedArray_CPU, resultArrayIndex_CPU,
															  inputWorkIndex);
					if (returnValue != 0) return returnValue;
				}
				numCudaGpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[2] (OPENCL GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + OPENCL_GPU:
				printf("ghEventsComputeDone OpenCL GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					int returnValue = OpenclGpuGenerateSortWork(sortSpec, NumOfItemsInWorkQuanta,
																sourceUnsortedArray_OpenClGPU, sourceArrayIndex_OpenClGPU, sourceUnsortedArray_CPU, sourceArrayIndex_CPU,
																resultSortedArray_OpenClGPU, resultArrayIndex_OpenClGPU, resultSortedArray_CPU, resultArrayIndex_CPU,
																inputWorkIndex);
					if (returnValue != 0) return returnValue;
				}
				numOpenclGpuWorkItemsDone++;
				break;

			case WAIT_TIMEOUT:
				printf("Wait timed out.\n");
				break;

				// Return value is invalid.
			default:
				printf("Wait error: %d\n", GetLastError());
				ExitProcess(0);
			}
			outputWorkIndex++;
		}
		timer.timeStamp();

		printf("CPU        completed %d work items\nCUDA   GPU completed %d work items\nOpenCL GPU completed %d work items\n", numCpuWorkItemsDone, numCudaGpuWorkItemsDone, numOpenclGpuWorkItemsDone);

		if (sortSpec.resultDestination == ResultInCpuMemory && !sortSpec.CudaGPU.allowedToWork && !sortSpec.OpenclGPU.allowedToWork)
		{
			printf("To sort randoms by CPU only, ran at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU sorted %0.0f%% randoms\n", (double)numCpuWorkItemsDone / NumOfWorkItems * 100);
			benchmarkFile << NumOfWorkItems * NumOfItemsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
		}
		else if (sortSpec.resultDestination == ResultInCpuMemory && sortSpec.CudaGPU.allowedToWork && sortSpec.OpenclGPU.allowedToWork)
		{
			printf("Just sorting of randoms runs at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU sorted %0.0f%% randoms, CUDA GPU generated %0.0f%% randoms, OpenCL GPU generated %0.0f%% randoms\n",
				(double)numCpuWorkItemsDone / NumOfWorkItems * 100, (double)numCudaGpuWorkItemsDone / NumOfWorkItems * 100, (double)numOpenclGpuWorkItemsDone / NumOfWorkItems * 100);
			benchmarkFile << NumOfWorkItems * NumOfItemsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
		}
		else if ((sortSpec.resultDestination == ResultInEachDevicesMemory && sortSpec.CudaGPU.allowedToWork) ||
			     (sortSpec.resultDestination == ResultInCudaGpuMemory     && sortSpec.CudaGPU.allowedToWork))
		{
			printf("Just generation of randoms runs at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU generated %zd randoms, CudaGPU generated %zd randoms.\nAsked to generate %zd, generated %zd\n",
				resultArrayIndex_CPU, resultArrayIndex_CudaGPU, NumOfItemsToSort, resultArrayIndex_CPU + resultArrayIndex_CudaGPU);
			benchmarkFile << NumOfWorkItems * NumOfItemsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfItemsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
			timer.reset();
			timer.timeStamp();
			// Copy all of the GPU generated randoms for verification of correctness by some rudamentary statistics
			copyCudaToSystemMemory(&resultSortedArray_CPU[resultArrayIndex_CPU], resultSortedArray_CudaGPU, resultArrayIndex_CudaGPU * sizeof(float));
			timer.timeStamp();
			//printf("Copy from CudaGPU to CPU runs at %zd bytes/second\n", (size_t)((double)(resultArrayIndex_GPU * sizeof(float)) / timer.getAverageDeltaInSeconds()));
		}

		// TODO: Turn this into a statistics checking function
		double average = 0.0;
		size_t totalRandomsGenerated = 0;
		if (sortSpec.resultDestination == ResultInCpuMemory)
			totalRandomsGenerated = resultArrayIndex_CPU;
		else
			totalRandomsGenerated = resultArrayIndex_CPU + resultArrayIndex_CudaGPU;
		for (size_t i = 0; i < totalRandomsGenerated; i++)
			average += resultSortedArray_CPU[i];
		average /= totalRandomsGenerated;
		printf("Mean = %f of %zd random values. Random array size is %zd\n\n", average, totalRandomsGenerated, sortSpec.totalItemsToSort);
	}
	return 0;
}
#endif