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

extern DWORD WINAPI ThreadMultiCoreCpuCompute(LPVOID);
extern DWORD WINAPI ThreadCudaGpuCompute(LPVOID);
extern DWORD WINAPI ThreadOpenclGpuCompute(LPVOID);

extern CudaRngEncapsulation			* gCudaRngSupport;			// TODO: Make sure to delete it once done
extern CudaMemoryEncapsulation		* gCudaMemorySupport;		// TODO: Make sure to delete it once done
extern OpenClGpuRngEncapsulation    * gOpenClRngSupport;		// TODO: Make sure to delete it once done
extern OpenClGpuMemoryEncapsulation	* gOpenClMemorySupport;		// TODO: Make sure to delete it once done

HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - CPU, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
bool   gRunComputeWorkers = true;

int loadBalancerCreateEventsAndThreads(void)
{
	HANDLE hThread;
	DWORD i, dwThreadID;

	ghEventHaveWorkItemForCpu = CreateEvent(
		NULL,   // default security attributes
		FALSE,  // auto-reset event object
		FALSE,  // initial state is nonsignaled
		NULL);  // unnamed object

	if (ghEventHaveWorkItemForCpu == NULL)
	{
		printf("CreateEvent error: %d\n", GetLastError());
		return -2;
	}
	ghEventHaveWorkItemForCudaGpu = CreateEvent(
		NULL,   // default security attributes
		FALSE,  // auto-reset event object
		FALSE,  // initial state is nonsignaled
		NULL);  // unnamed object

	if (ghEventHaveWorkItemForCudaGpu == NULL)
	{
		printf("CreateEvent error: %d\n", GetLastError());
		return -2;
	}
	ghEventHaveWorkItemForOpenclGpu = CreateEvent(
		NULL,   // default security attributes
		FALSE,  // auto-reset event object
		FALSE,  // initial state is nonsignaled
		NULL);  // unnamed object

	if (ghEventHaveWorkItemForOpenclGpu == NULL)
	{
		printf("CreateEvent error: %d\n", GetLastError());
		return -2;
	}

	// Create all compute done event objects
	for (i = 0; i < NumComputeDoneEvents; i++)
	{
		ghEventsComputeDone[i] = CreateEvent(
			NULL,   // default security attributes
			FALSE,  // auto-reset event object
			FALSE,  // initial state is nonsignaled
			NULL);  // unnamed object

		if (ghEventsComputeDone[i] == NULL)
		{
			printf("CreateEvent error: %d\n", GetLastError());
			return -1;
		}
	}

	gRunComputeWorkers = true;
	// TODO: Automate with an array of function pointers and an array of thread ID's that would be assigned by CreteThread
	// Create MultiCoreCpuCompute thread (e.g. for MKL)
	hThread = CreateThread(
		NULL,         // default security attributes
		0,            // default stack size
		(LPTHREAD_START_ROUTINE)ThreadMultiCoreCpuCompute,
		NULL,         // no thread function arguments
		0,            // default creation flags
		&dwThreadID); // receive thread identifier

	if (hThread == NULL)
	{
		printf("CreateThread error: %d\n", GetLastError());
		return -3;
	}

	// Create CudaGpu thread (e.g. for cuRAND)
	hThread = CreateThread(
		NULL,         // default security attributes
		0,            // default stack size
		(LPTHREAD_START_ROUTINE)ThreadCudaGpuCompute,
		NULL,         // no thread function arguments
		0,            // default creation flags
		&dwThreadID); // receive thread identifier

	if (hThread == NULL)
	{
		printf("CreateThread error: %d\n", GetLastError());
		return -4;
	}

	// Create OpenclGpu thread for ArrayFire on Intel GPU
	hThread = CreateThread(
		NULL,         // default security attributes
		0,            // default stack size
		(LPTHREAD_START_ROUTINE)ThreadOpenclGpuCompute,
		NULL,         // no thread function arguments
		0,            // default creation flags
		&dwThreadID); // receive thread identifier

	if (hThread == NULL)
	{
		printf("CreateThread error: %d\n", GetLastError());
		return -4;
	}

	return 0;
}
int loadBalancerDestroyEventsAndThreads(void)
{
	// TODO: Shut down all compute threads and wait for them to be done
	gRunComputeWorkers = false;
	// TODO: How do we shut down threads that are blocked waiting on work events? Maybe by creating null/zero/shutdown work item to let them know it's time to shutdown.

	// Close event handles
	CloseHandle(ghEventHaveWorkItemForCpu);
	CloseHandle(ghEventHaveWorkItemForCudaGpu);
	for (size_t i = 0; i < NumComputeDoneEvents; i++)
		CloseHandle(ghEventsComputeDone[i]);

	return 0;
}

int InitialCpuGenWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta, float * resultArray_CPU, size_t & resultArrayIndex_CPU, size_t & inputWorkIndex)
{
	if (genSpec.CPU.allowedToWork &&
		(genSpec.resultDestination == ResultInCpuMemory || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CPU is allowed to do work, it should do work
			genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
		//printf("First CPU work item\n");
		workCPU.WorkerType = ComputeEngine::CPU;
		workCPU.AmountOfWork = NumOfRandomsInWorkQuanta;
		workCPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
		//printf("Event set for work item for MultiCore CPU\n");
		if (!SetEvent(ghEventHaveWorkItemForCpu))	// signal that CPU has a work item to work on
		{
			printf("SetEvent ghEventWorkForCpu failed (%d)\n", GetLastError());
			return -5;
		}
		resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;
		inputWorkIndex++;
	}
	return 0;
}

int InitialGpuGenWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta,
						float * resultArray_GPU, size_t & resultArrayIndex_GPU, float * resultArray_CPU, size_t & resultArrayIndex_CPU,
						size_t & inputWorkIndex)
{
	if (genSpec.CudaGPU.allowedToWork &&
		(genSpec.resultDestination == ResultInCpuMemory || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
			genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
		if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.CudaGPU.maxRandoms) {
			//printf("First CudaGPU work item\n");
			workCudaGPU.WorkerType = ComputeEngine::CUDA_GPU;
			workCudaGPU.AmountOfWork = NumOfRandomsInWorkQuanta;
			workCudaGPU.DeviceResultPtr = (char *)(&(resultArray_GPU[resultArrayIndex_GPU]));
			if (genSpec.resultDestination == ResultInCpuMemory) {
				workCudaGPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
				resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
																		// don't advance GPU array index, since we reuse the same result array
			}
			else if (genSpec.resultDestination == ResultInEachDevicesMemory || genSpec.resultDestination == ResultInCudaGpuMemory) {
				workCudaGPU.HostResultPtr = NULL;
				resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
				// don't advance CPU array index
			}
			//printf("Cuda GPU work item: amountOfWork = %d at GPU memory address %p\n", workCudaGPU.AmountOfWork, workCudaGPU.DeviceResultPtr);
			//printf("Event set for work item for CUDA GPU\n");
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

int InitialOpenclGpuGenWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta,
							float * resultArray_GPU, size_t & resultArrayIndex_GPU, float * resultArray_CPU, size_t & resultArrayIndex_CPU,
							size_t & inputWorkIndex)
{
	if (genSpec.OpenclGPU.allowedToWork &&
		(genSpec.resultDestination == ResultInCpuMemory     || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
		 genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
		if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.OpenclGPU.maxRandoms) {
			//printf("First OpenclGPU work item\n");
			workOpenclGPU.WorkerType = ComputeEngine::OPENCL_GPU;
			workOpenclGPU.AmountOfWork = NumOfRandomsInWorkQuanta;
			workOpenclGPU.DeviceResultPtr = NULL;
			if (genSpec.resultDestination == ResultInCpuMemory) {
				workOpenclGPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
				resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
																		// don't advance GPU array index, since we reuse the same result array
			}
			else if (genSpec.resultDestination == ResultInEachDevicesMemory || genSpec.resultDestination == ResultInOpenclGpuMemory) {
				workOpenclGPU.HostResultPtr = NULL;
				resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
				// don't advance CPU array index
			}
			//printf("OpenCL GPU work item: amountOfWork = %d at GPU memory address %p\n", workOpenclGPU.AmountOfWork, workOpenclGPU.DeviceResultPtr);
			//printf("Event set for work item for OpenCL GPU\n");
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

// TODO: Capture the pattern of load balancing any algorithm, possibly using a template to load balance anything.
// numTimes is needed to see if running the first time is slower than running subsequent times, as this is commonly the case due to OS paging and CPU caching
int runLoadBalancerThread(RandomsToGenerate& genSpec, ofstream& benchmarkFile, unsigned numTimes)
{
	size_t NumOfRandomsToGenerate = genSpec.randomsToGenerate;
	size_t NumOfRandomsInWorkQuanta = genSpec.CPU.workQuanta;		// TODO: Need to separate CPU and GPU workQuanta, and handle them being different
																	// TODO: Fix the problem with the case of asking the CudaGPU to generate more randoms that can fit into it's memory, but no other computational units are helping to generate more
																	// TODO: One possible way to do this is to pre-determine the NumOfWorkItems and shrink it in case there is not enough memory between all of the generators
																	// TODO: Another way is to create a method that takes genSpec as input and outputs all of the needed setup variables with their values for the rest of the code to use
	// Figure out how many work items to generate
	// TODO: Currently, this is static, but eventually should be dynamic (within the work generation loop) as work quanta will be different for each computational unit and may even be dynamically sized
	size_t NumOfWorkItems = NumOfWorkItems = NumOfRandomsToGenerate / NumOfRandomsInWorkQuanta;
	if (genSpec.resultDestination == ResultInCudaGpuMemory && !genSpec.CPU.allowedToWork && !genSpec.OpenclGPU.allowedToWork && !genSpec.FpgaGPU.allowedToWork)	// only CudaGPU is working
		NumOfWorkItems = __min(genSpec.CudaGPU.maxRandoms, NumOfRandomsToGenerate) / genSpec.CudaGPU.workQuanta;
	else if (genSpec.resultDestination == ResultInCpuMemory)
		NumOfWorkItems = NumOfWorkItems = NumOfRandomsToGenerate / NumOfRandomsInWorkQuanta;

	float *randomFloatArray_GPU = (float *)gCudaMemorySupport->m_gpu_memory;
	float *randomFloatArray_CPU = (float *)genSpec.generated.CPU.Buffer;

	for (unsigned numRuns = 0; numRuns < numTimes; numRuns++)
	{
		TimerCycleAccurateArray	timer;
		timer.reset();
		timer.timeStamp();
		// Start each worker in the graph, once the graph has been constructructed
		// TODO: Need to handle less work than enough for each of the worker type (e.g. 1.5xWorkQuanta randoms, 0.2xWorkQuanta randoms, with two available workers)
		size_t resultArrayIndex_CPU = 0;
		size_t resultArrayIndex_GPU = 0;
		size_t inputWorkIndex = 0;
		if (NumOfWorkItems > 0) {	// CPU work item
			// TODO: Consider all combinations of where the randoms end up and who is allowed to help generate them. Is there a way to handle them in a general way (flags)?
			int returnValue = InitialCpuGenWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 1) {	// CudaGpu work item
			int returnValue = InitialGpuGenWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_GPU, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 2) {	// OpenclGpu work item
			int returnValue = InitialOpenclGpuGenWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_GPU, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
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
				//printf("ghEventsComputeDone CPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems)	// Create new work item for CPU
				{
					if (genSpec.CPU.allowedToWork &&
					   (genSpec.resultDestination == ResultInCpuMemory || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CPU is allowed to do work, it should do work
						genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
						//printf("Another CPU work item\n");
						workCPU.WorkerType = ComputeEngine::CPU;
						workCPU.AmountOfWork = NumOfRandomsInWorkQuanta;
						workCPU.HostResultPtr = (char *)(&(randomFloatArray_CPU[resultArrayIndex_CPU]));
						if (!SetEvent(ghEventHaveWorkItemForCpu))	// Set one event to the signaled state
						{
							printf("SetEvent ghEventWorkForCpu failed (%d)\n", GetLastError());
							return -5;
						}
						inputWorkIndex++;
						//printf("Gave new work item to CPU. resultArrayIndex = %zd. Completed %zd work items\n", resultArrayIndex, workCompletedByCPU);
						resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;
					}
				}
				numCpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[1] (CUDA GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + CUDA_GPU:
				//printf("ghEventsComputeDone CUDA GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					if (genSpec.CudaGPU.allowedToWork &&
					   (genSpec.resultDestination == ResultInCpuMemory     || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
						genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
						if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.CudaGPU.maxRandoms) {
							//printf("Another CudaGPU work item\n");
							workCudaGPU.WorkerType   = ComputeEngine::CUDA_GPU;
							workCudaGPU.AmountOfWork = NumOfRandomsInWorkQuanta;
							workCudaGPU.DeviceResultPtr    = (char *)(&(randomFloatArray_GPU[resultArrayIndex_GPU]));
							//printf("resultArrayIndex_GPU = %zd\n", resultArrayIndex_GPU);
							if (genSpec.resultDestination == ResultInCpuMemory) {
								workCudaGPU.HostResultPtr = (char *)(&(randomFloatArray_CPU[resultArrayIndex_CPU]));
								resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
																						// don't advance GPU array index, since we reuse the same result array
							}
							else if (genSpec.resultDestination == ResultInEachDevicesMemory || genSpec.resultDestination == ResultInCudaGpuMemory) {
								workCudaGPU.HostResultPtr = NULL;
								resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
								// don't advance CPU array index
							}
							//printf("Created work items for CUDA GPU\n");
							if (!SetEvent(ghEventHaveWorkItemForCudaGpu))	// Set one event to the signaled state
							{
								printf("SetEvent ghEventHaveWorkItemForCudaGpu failed (%d)\n", GetLastError());
								return -6;
							}
							inputWorkIndex++;
							//printf("Gave new work item to GPU. resultArrayIndex = %zd. Completed %zd work items\n", resultArrayIndex_GPU, numCudaGpuWorkItemsDone);
						}
					}
				}
				numCudaGpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[2] (OPENCL GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + OPENCL_GPU:
				//printf("ghEventsComputeDone OpenCL GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					if (genSpec.OpenclGPU.allowedToWork &&
						(genSpec.resultDestination == ResultInCpuMemory     || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CudaGpu is allowed to do work, it should do work
						 genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
						if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.OpenclGPU.maxRandoms) {
							//printf("Another OpenClGPU work item\n");
							workOpenclGPU.WorkerType = ComputeEngine::CUDA_GPU;
							workOpenclGPU.AmountOfWork = NumOfRandomsInWorkQuanta;
							workOpenclGPU.DeviceResultPtr = NULL;
							//printf("resultArrayIndex_GPU = %zd\n", resultArrayIndex_GPU);
							if (genSpec.resultDestination == ResultInCpuMemory) {
								workOpenclGPU.HostResultPtr = (char *)(&(randomFloatArray_CPU[resultArrayIndex_CPU]));
								resultArrayIndex_CPU += NumOfRandomsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
																						// don't advance GPU array index, since we reuse the same result array
							}
							else if (genSpec.resultDestination == ResultInEachDevicesMemory || genSpec.resultDestination == ResultInCudaGpuMemory) {
								workOpenclGPU.HostResultPtr = NULL;
								resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
								// don't advance CPU array index
							}
							//printf("Created work items for CUDA GPU\n");
							if (!SetEvent(ghEventHaveWorkItemForOpenclGpu))	// Set one event to the signaled state
							{
								printf("SetEvent ghEventHaveWorkItemForOpenclGpu failed (%d)\n", GetLastError());
								return -6;
							}
							inputWorkIndex++;
							//printf("Gave new work item to OpenCL GPU. resultArrayIndex = %zd. Completed %zd work items\n", resultArrayIndex_GPU, numCudaGpuWorkItemsDone);
						}
					}
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

		if (genSpec.resultDestination == ResultInCpuMemory && !genSpec.CudaGPU.allowedToWork && !genSpec.OpenclGPU.allowedToWork)
		{
			printf("To generate randoms by CPU only, ran at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU generated %0.0f%% randoms\n", (double)numCpuWorkItemsDone / NumOfWorkItems * 100);
			benchmarkFile << NumOfWorkItems * NumOfRandomsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
		}
		else if (genSpec.resultDestination == ResultInCpuMemory && genSpec.CudaGPU.allowedToWork && genSpec.OpenclGPU.allowedToWork)
		{
			printf("Just generation of randoms runs at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU generated %0.0f%% randoms, CUDA GPU generated %0.0f%% randoms, OpenCL GPU generated %0.0f%% randoms\n",
				(double)numCpuWorkItemsDone / NumOfWorkItems * 100, (double)numCudaGpuWorkItemsDone / NumOfWorkItems * 100, (double)numOpenclGpuWorkItemsDone / NumOfWorkItems * 100);
			benchmarkFile << NumOfWorkItems * NumOfRandomsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
		}
		else if ((genSpec.resultDestination == ResultInEachDevicesMemory && genSpec.CudaGPU.allowedToWork) ||
			     (genSpec.resultDestination == ResultInCudaGpuMemory     && genSpec.CudaGPU.allowedToWork))
		{
			printf("Just generation of randoms runs at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("CPU generated %zd randoms, CudaGPU generated %zd randoms.\nAsked to generate %zd, generated %zd\n",
				resultArrayIndex_CPU, resultArrayIndex_GPU, NumOfRandomsToGenerate, resultArrayIndex_CPU + resultArrayIndex_GPU);
			benchmarkFile << NumOfWorkItems * NumOfRandomsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
			timer.reset();
			timer.timeStamp();
			// Copy all of the GPU generated randoms for verification of correctness by some rudamentary statistics
			copyCudaToSystemMemory(&randomFloatArray_CPU[resultArrayIndex_CPU], randomFloatArray_GPU, resultArrayIndex_GPU * sizeof(float));
			timer.timeStamp();
			//printf("Copy from CudaGPU to CPU runs at %zd bytes/second\n", (size_t)((double)(resultArrayIndex_GPU * sizeof(float)) / timer.getAverageDeltaInSeconds()));
		}

		// TODO: Turn this into a statistics checking function
		double average = 0.0;
		size_t totalRandomsGenerated = 0;
		if (genSpec.resultDestination == ResultInCpuMemory)
			totalRandomsGenerated = resultArrayIndex_CPU;
		else
			totalRandomsGenerated = resultArrayIndex_CPU + resultArrayIndex_GPU;
		for (size_t i = 0; i < totalRandomsGenerated; i++)
			average += randomFloatArray_CPU[i];
		average /= totalRandomsGenerated;
		printf("Mean = %f of %zd random values. Random array size is %zd\n\n", average, totalRandomsGenerated, genSpec.randomsToGenerate);
	}
	return 0;
}

// TODO: The last argument should be enum specifying whether to leave randoms in their respective memories for the next processing step to use them from there
// TODO: or to copy them to system memory or to GPU memory (i.e. which memory should the result end up in)
// TODO: Figure out what to do in each case, as GPU memory may be smaller in some cases than system memory. What if what the user asks for is not possible? Need to
// TODO: return error codes that are meaningful.
// TODO: It also doesn't seem like the user should specify WorkChunkSize, since the user doesn't really know what to set that to. That's something we would need to determine somehow
// TODO: Experiment with pinned shared memory cudaMallocHost(), since it may be faster to transfer between that memory and system memory. Maybe it's possible to generate straight in that memory and may be faster
// You should be able to ask a generator to generate 2 billion floats. One way is to generate as fast as it can and potentially use all of the GPU memory, if the GPU
// is the fastest generator. So, it should generate and leave it in the fastest devices memory, returning the structure that tells us where the data is and how much is
// in each device.
// Do memory allocators allow you to trim the memory you've allocated without moving the data - that would be cool. As we could allocate lots of memory and then
// trim it down in-place to only be as large as what we've actually used. Or, we need to return an array of workItem sized elements and grow the array as we go, but
// I'm not sure if GPU array can be grown automagically and efficiently like C++ ones can be.
// Maybe one option is to provide an array of pointers to each workItem, if GPU supports shared memory concept and we can get at this shared memory efficiently from the CPU
// and the GPU generator is not much slower than generating into GPU memory. Another option is to copy into system memory or GPU memory. Wtih GeForce 1070 coming with 8GBytes of
// memory, this is 1/2 of what my system has, even if you have 32 GB of system memory, 8 GB of graphics memory is substantial percentage of overall memory.
// Step #1: Let's just put the best benchmarks out of my blog and website and go from there.
// Step #2: Work on other output options, such a single array in any memory, and an array of workItems in shared memory (i.e. no copy, but less convenient to use)
int SelectAndRunLoadBalancerThread(RandomsToGenerate& genSpec, ofstream& benchmarkFile, unsigned NumTimes)
{
	if (genSpec.resultDestination == ResultInEachDevicesMemory)
	{
		printf("SelectAndRunLoadBalancerThread with ResultInEachDevicesMemory\n");
		// TODO: Need to return an address, number of randoms returned and the size of memory allocated.
		// TODO: Need to NOT free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
		return runLoadBalancerThread(genSpec, benchmarkFile, NumTimes);
	}
	else if (genSpec.resultDestination == ResultInCudaGpuMemory)
	{
		printf("SelectAndRunLoadBalancerThread with ResultInCudaGPUMemory\n");
		// TODO: Need to return an address, number of randoms returned and the size of memory allocated.
		// TODO: Need to NOT free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
		runLoadBalancerThread(genSpec, benchmarkFile, NumTimes);
		genSpec.generated.CPU.Length = 0;
	}
	else if (genSpec.resultDestination == ResultInCpuMemory)
	{
		// TODO: Need to not free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
		// TODO: GPU generated memory allocation needs to be different than for non-copy case, to handle each workItem and to copy the result of each workItem
		// TODO: In this way we can handle way bigger array and need to fail it if workItem's worth of randoms don't fit into GPU memory
		if (genSpec.CudaGPU.allowedToWork && genSpec.CudaGPU.workQuanta > genSpec.CudaGPU.memoryCapacity)
			return -3;		// TODO: Define error return codes in the .h file we provide to the users

		runLoadBalancerThread(genSpec, benchmarkFile, NumTimes);
		genSpec.generated.CudaGPU.Buffer = NULL;
		genSpec.generated.CudaGPU.Length = 0;
	}

	return 0;
}

void MemoryAllocatorCpu_GenRng(RandomsToGenerate& genSpec)
{
	// Determine how much CPU memory to allocate.
	// TODO: We should only allocate as much CPU/System memory as will be actually used (as not all result may go into system memory)
	// TODO: Only allocate system memory when we are going to put randoms into it
	// TODO: Create CpuMemoryEncapsulation class, just like there is one for the GPU
	if (genSpec.generated.CPU.Buffer == NULL) {
		printf("Before allocation of NumOfBytesForRandomArray = %zd\n", genSpec.randomsToGenerate * sizeof(float));
		float * randomFloatArray = new float[genSpec.randomsToGenerate];
		// Clearing the arrays also pages them in (warming them up), which improves performance by 3X for the first generator due to first use
		// TODO: reading one byte from each page is a faster way to warm up (page in) the array. I already have code for this.
		memset((void *)randomFloatArray, 0, genSpec.randomsToGenerate * sizeof(float));
		genSpec.generated.CPU.Buffer = (char *)randomFloatArray;
	}
	printf("After allocation of NumOfBytesForRandomArray = %zd at CPU memory location = %p\n", genSpec.randomsToGenerate * sizeof(float), genSpec.generated.CPU.Buffer);

	if (genSpec.CPU.workQuanta == 0)
		genSpec.CPU.workQuanta = 20 * 1024 * 1024;	// TODO: Need to define a global constant for this

	printf("NumOfRandomsToGenerate = %zd, CPU.workQuanta = %zd\n", genSpec.randomsToGenerate, genSpec.CPU.workQuanta);
}

int MemoryAllocatorCudaGpu_GenRng(RandomsToGenerate& genSpec)
{
	// Determine how much GPU memory to allocate
	if (genSpec.CudaGPU.workQuanta == 0)
		genSpec.CudaGPU.workQuanta = 20 * 1024 * 1024;	// TODO: Need to define a global constant for this
														// TODO: Develop a way to determine optimal work chunk size, possibly dynamically or during install on that machine, or over many runs get to better and better performance
	//size_t NumOfRandomsInWorkQuanta = genSpec.CudaGPU.workQuanta;	// TODO: Need to separate CPU and GPU workQuanta, and handle them being different
																	// TODO: Fix the problem with the case of asking the CudaGPU to generate more randoms that can fit into it's memory, but no other computational units are helping to generate more
																	// TODO: One possible way to do this is to pre-determine the NumOfWorkItems and shrink it in case there is not enough memory between all of the generators
																	// TODO: Another way is to create a method that takes genSpec as input and outputs all of the needed setup variables with their values for the rest of the code to use
	size_t preallocateGPUmemorySize;
	if (genSpec.resultDestination == ResultInCpuMemory && genSpec.CudaGPU.allowedToWork)
		preallocateGPUmemorySize = genSpec.CudaGPU.workQuanta * genSpec.CudaGPU.sizeOfItem;	// when GPU is a helper, only pre-allocate workQuanta size in GPU memory and use the same memory buffer for each work item
	else if (!genSpec.CudaGPU.allowedToWork)
		preallocateGPUmemorySize = 0;
	else if (genSpec.CudaGPU.allowedToWork)
		preallocateGPUmemorySize = genSpec.CudaGPU.maxRandoms * genSpec.CudaGPU.sizeOfItem;
	else {
		printf("Error: Unsupported configuration\n");
		return -1;
	}
	printf("Allocating CudaGPU memory of %zd bytes\n", preallocateGPUmemorySize);

	// CUDA memory allocation is extremely slow (seconds)!
	gCudaMemorySupport = new CudaMemoryEncapsulation(preallocateGPUmemorySize);
	genSpec.CudaGPU.itemsAllocated = preallocateGPUmemorySize / genSpec.CudaGPU.sizeOfItem;

	printf("NumOfRandomsToGenerate = %zd, GPU.workQuanta = %zd\n", genSpec.randomsToGenerate, genSpec.CudaGPU.workQuanta);
	return 0;
}

int MemoryAllocatorOpenClGpu_GenRng(RandomsToGenerate& genSpec)
{
	// Determine how much GPU memory to allocate
	if (genSpec.OpenclGPU.workQuanta == 0)
		genSpec.OpenclGPU.workQuanta = 20 * 1024 * 1024;	// TODO: Need to define a global constant for this
														// TODO: Develop a way to determine optimal work chunk size, possibly dynamically or during install on that machine, or over many runs get to better and better performance
														//size_t NumOfRandomsInWorkQuanta = genSpec.CudaGPU.workQuanta;	// TODO: Need to separate CPU and GPU workQuanta, and handle them being different
														// TODO: Fix the problem with the case of asking the CudaGPU to generate more randoms that can fit into it's memory, but no other computational units are helping to generate more
														// TODO: One possible way to do this is to pre-determine the NumOfWorkItems and shrink it in case there is not enough memory between all of the generators
														// TODO: Another way is to create a method that takes genSpec as input and outputs all of the needed setup variables with their values for the rest of the code to use
	size_t preallocateGPUmemorySize;
	if (genSpec.resultDestination == ResultInCpuMemory && genSpec.OpenclGPU.allowedToWork)
		preallocateGPUmemorySize = genSpec.OpenclGPU.workQuanta * genSpec.OpenclGPU.sizeOfItem;	// when GPU is a helper, only pre-allocate workQuanta size in GPU memory and use the same memory buffer for each work item
	else if (!genSpec.OpenclGPU.allowedToWork)
		preallocateGPUmemorySize = 0;
	else if (genSpec.OpenclGPU.allowedToWork)
		preallocateGPUmemorySize = genSpec.OpenclGPU.maxRandoms * genSpec.OpenclGPU.sizeOfItem;
	else {
		printf("Error: Unsupported configuration\n");
		return -1;
	}
	printf("Allocating OpenclGPU memory of %zd bytes\n", preallocateGPUmemorySize);

	// CUDA memory allocation is extremely slow (seconds)!
	gOpenClMemorySupport = new OpenClGpuMemoryEncapsulation(preallocateGPUmemorySize);
	genSpec.OpenclGPU.itemsAllocated = preallocateGPUmemorySize / genSpec.OpenclGPU.sizeOfItem;

	printf("NumOfRandomsToGenerate = %zd, OpenclGPU.workQuanta = %zd\n", genSpec.randomsToGenerate, genSpec.OpenclGPU.workQuanta);
	return 0;
}

int Validator(RandomsToGenerate& genSpec)
{
	if (genSpec.CPU.memoryCapacity < (genSpec.CPU.maxRandoms * sizeof(float))) {
		printf("Error: Maximum number of randoms for CPU memory exceeds memory capacity.\n");
		return -1;
	}
	if (genSpec.CudaGPU.memoryCapacity < (genSpec.CudaGPU.maxRandoms * sizeof(float))) {
		printf("Error: Maximum number of randoms for GPU memory exceeds memory capacity.\n");
		return -2;
	}

	return 0;
}

int benchmarkLoadBalancer()
{
	// TODO: Clean-up and split up cudaRand code into setup the PRNG function, memory allocation function and run several times function, to see how much CPU time the run function uses
	// TODO: Then implement the idea of the global timer that everyone can use and start putting it everywhere to see where time is being spent, and order of operations
	// TODO: Add the ability to specify all faster computational units to help all slower ones to generate more randoms, and the ability to be explicit on who helps whom
	// TODO: To debug high CPU utilization with graph and just cudaGPU running, reduce the graph to just the async_nod and figure out if it's the one or some other node is using lots of CPU
	string benchmarkFilename = "benchmarkResultsRng.txt";
	ofstream benchmarkFile;
	benchmarkFile.open(benchmarkFilename);
	benchmarkFile << "Number of Randoms\t" << "Randoms per second" << endl;		// header for columns

	//broadcastNodeExample();

	// TODO: Needs to be part of CUDA GPU setup, once at the beginning
	int argc = 0;
	const char **argv = NULL;
	int m_CudaDeviceID = findCudaDevice(argc, (const char **)argv);	// TODO: need to do this operation only once

	// TODO: Needs to be part of CUDA GPU algorithms setup
	unsigned long long prngSeed = 2;
	gCudaRngSupport = new CudaRngEncapsulation(prngSeed);

	// TODO: Needs to be part of OpenCL GPU setup, once at the beginning
	af::setDevice(1);	// device 1 on my laptop is Intel 530 GPU
	af::info();

	gOpenClRngSupport = new OpenClGpuRngEncapsulation(prngSeed);

	bool copyGPUresultsToSystemMemory = false;
	RandomsToGenerate genSpec;

	size_t maxRandomsToGenerate = (size_t)400 * 1024 * 1024;
	size_t minRandomsToGenerate = (size_t)380 * 1024 * 1024;
	unsigned NumTimes = 10;
	// TODO: Seems to be a bug going down to 20M randoms - runs forever
	// TODO: We also need to not be limited to the increment being of the same size as workQuanta
	size_t randomsToGenerateIncrement = (size_t)20 * 1024 * 1024;	// workQuanta increment to make sure total work divides evenly until we can support it not

																	// !! TODO: Create a structure of time stamp and string, to be able to create an array of time stamps and their identification with additional information
																	// !! TODO: This will help debug where the delays are and help determine if the issue is in my code or in TBB itself, as we expect no iterference between cudaGPU and MKL
																	// !! TODO: when the storage of randoms is in their respective local memories. The timestamp structure may have to be a global to avoid passing it into all layers of hierarchy.
	genSpec.randomsToGenerate = maxRandomsToGenerate;
	//genSpec.resultDestination = ResultInEachDevicesMemory;		// TODO: Eventually, we want this to be dynamic per work item for each worker to be adaptive to current conditions
	genSpec.resultDestination = ResultInCpuMemory;					// TODO: Eventually, we want this to be dynamic per work item for each worker to be adaptive to current conditions

	genSpec.CPU.workQuanta = 0;		// indicates user is ok with automatic determination
	genSpec.CPU.memoryCapacity = (size_t)16 * 1024 * 1024 * 1024;
	// TODO: That's all that should be specified - i.e. max percentage of device memory to be used
	genSpec.CPU.sizeOfItem = sizeof(float);
	genSpec.CPU.maxRandoms = (size_t)(genSpec.CPU.memoryCapacity * 0.50) / genSpec.CPU.sizeOfItem;	// use up to 50% of CPU memory for randoms
	genSpec.CPU.allowedToWork = true;
	genSpec.CPU.prngSeed = std::time(0);

	genSpec.CudaGPU.workQuanta = 0;		// indicates user is ok with automatic determination 
	genSpec.CudaGPU.memoryCapacity = (size_t)2  * 1024 * 1024 * 1024;
	// TODO: That's all that should be specified - i.e. max percentage of device memory to be used
	genSpec.CudaGPU.sizeOfItem = sizeof(float);
	genSpec.CudaGPU.maxRandoms = (size_t)(genSpec.CudaGPU.memoryCapacity * 0.75) / genSpec.CudaGPU.sizeOfItem;	// use up to 75% of GPU memory for randoms
	genSpec.CudaGPU.allowedToWork = true;
	genSpec.CudaGPU.prngSeed = std::time(0) + 10;

	genSpec.OpenclGPU.workQuanta = 0;		// indicates user is ok with automatic determination 
	genSpec.OpenclGPU.memoryCapacity = (size_t)6 * 1024 * 1024 * 1024;
	// TODO: That's all that should be specified - i.e. max percentage of device memory to be used
	genSpec.OpenclGPU.sizeOfItem = sizeof(float);
	genSpec.OpenclGPU.maxRandoms = (size_t)(genSpec.OpenclGPU.memoryCapacity * 0.75) / genSpec.OpenclGPU.sizeOfItem;	// use up to 75% of GPU memory for randoms
	genSpec.OpenclGPU.allowedToWork = true;
	genSpec.OpenclGPU.prngSeed = std::time(0) + 10;

	genSpec.generated.CPU.Buffer = NULL;		// NULL implies the generator is to allocate memory. non-NULL implies reuse the buffer provided
	genSpec.generated.CPU.Length = 0;
	genSpec.generated.CudaGPU.Buffer = NULL;	// NULL implies the generator is to allocate memory. non-NULL implies reuse the buffer provided
	genSpec.generated.CudaGPU.Length = 0;
	printf("genSpec set\n");

	if (Validator(genSpec) != 0)
		return -1;

	MemoryAllocatorCpu_GenRng(genSpec);

	MemoryAllocatorCudaGpu_GenRng(genSpec);

	MemoryAllocatorOpenClGpu_GenRng(genSpec);

	loadBalancerCreateEventsAndThreads();

	// TODO: Need to set the number of randoms generated in CPU memory and GPU memory at the end of all generation once it's known

	// TODO: Run multiple times (e.g. 1K times) for each size of array, and show performance of each time
	// TODO: Run over all cudaRand type of generators to provide performance numbers for all of them

	// Start at maximum size of array of randoms to generate, so that we can re-use the allocated array on the first iteration, since the rest of iteration will generate fewer randoms and will fit into the largest array
	for (size_t randomsToGenerate = maxRandomsToGenerate; randomsToGenerate >= minRandomsToGenerate; randomsToGenerate -= randomsToGenerateIncrement)
	{
		genSpec.randomsToGenerate = randomsToGenerate;
		// TODO: The two use cases I'm going to work on are:
		// TODO: 1. Generate a requested number of random numbers in system memory with the help of all computational units (multi-core CPU, CUDA GPU(s), OpenCL GPU(s), OpenCL FPGA(s)
		// TODO: 2. Generate a requested number of random numbers in variety of memories. The user will need to specify memory capacity of each computational unit (for now)
		// TODO:    and specify how much memory we are allowed to use. It seems like for each unit we need to its own workQuantaSize. Once each computational unit finishes filling
		// TODO:    up its own memory space, it could stop (for now). Later, we could improve this by having the units help each other.
		// TODO: Another item that's a must, is to configure the pipeline to have only a single computational unit do the work. This serves as a good comparison baseline,
		//       and gives us a way to see how much the additional computational units help.
		// TODO: Figure out what to do in each case, as GPU memory may be smaller in some cases than system memory. What if what the user asks for is not possible? Need to
		// TODO: return error codes that are meaningful.
		// TODO: It also doesn't seem like the user should specify WorkChunkSize, since the user doesn't really know what to set that to. That's something we would need to determine somehow
		// TODO: Experiment with pinned shared memory cudaMallocHost(), since it may be faster to transfer between that memory and system memory. Maybe it's possible to generate straight in that memory and may be faster

		// TODO: I need to setup a way to preallocate all memory (CPU and GPU) and reuse this memory for RNG over and over again, as this will provide the highest level of performance!
		// TODO: Of course, with the graph already pre-built and ready to go, along with memory that is pre-allocated
		// TODO: Also need to generate using the graph, but only using the GPU and have no help from CPU.

		// TODO: Separate facilities into pre-allocate memory in each computational unit, construct the graph, and then re-use the graph and memory over and over for RNG.
		// TODO: This usage should have the highest performance and makes sense for generators: provide them with memory, pre-construct them and then use them over and over again.
		// TODO: Remove workQuanta from users settings and create a wrapper structure for internal use that adds workQuanta to it, or pass it as a separate structure.

		// TODO: Consider Power savings as an awesome side effect from this and ride the coat-tails of Microsoft Edge on this (provide a link to their awesome blog of power savings
		// TODO: for video playback, which is an almost 2X power savings by using dedicated h/w) https://blogs.windows.com/windowsexperience/2016/07/13/get-better-quality-video-with-microsoft-edge/#6dMdIce30yUt9bSJ.97
		// TODO: To test it, we could fully charge my computer's battery and run C++ random generator, then run ours MKL/cudaRand and run it until the battery dies and see how many randoms each one generates
		// TODO: and compute random/time for each. We could then do it for MKL and cudaRand each alone and publish these findings to see which is most efficient, or maybe when using all
		// TODO: Create a UWP application service for high performance and accelerated algorithms, providing LSD Radix Sort, CudaRand algorithms, MKL algorithms, to make them available to C# and other UWP applications. Sadly, transfer of data runs at about 30MBytes/sec, which is way too slow. In-process UWP application service should perform well enough

		auto elapsed = time_call([&] {
			SelectAndRunLoadBalancerThread(genSpec, benchmarkFile, NumTimes);
		});
		printf("SelectAndRunLoadBalancerThread ran at an overall rate of %zd floats/second\n", (size_t)((double)genSpec.randomsToGenerate / (elapsed / 1000.0)));
	}

	loadBalancerDestroyEventsAndThreads();

	delete[] genSpec.generated.CPU.Buffer;
	delete gCudaMemorySupport;

	benchmarkFile.close();

	return 0;
}
