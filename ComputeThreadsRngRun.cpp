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
extern CudaMemoryEncapsulation		* gCudaMemorySupport;		// TODO: Make sure to delete it once done
extern OpenClGpuMemoryEncapsulation	* gOpenClMemorySupport;		// TODO: Make sure to delete it once done

extern HANDLE ghEventsComputeDone[NumComputeDoneEvents];	// 0 - CPU, 1 - CudaGpu, 2 - OpenClGpu, 3 - OpenClFpga
extern bool   gRunComputeWorkers;

int CpuGenerateWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta, float * resultArray_CPU, size_t & resultArrayIndex_CPU, size_t & inputWorkIndex)
{
	if (genSpec.CPU.allowedToWork &&
		(genSpec.resultDestination == ResultInCpuMemory || genSpec.resultDestination == ResultInEachDevicesMemory ||	// TODO: Where the result is going should not even matter, as long as CPU is allowed to do work, it should do work
			genSpec.resultDestination == ResultInCudaGpuMemory || genSpec.resultDestination == ResultInOpenclGpuMemory)) {
		//printf("First CPU work item\n");
		workCPU.WorkerType = ComputeEngine::CPU;
		workCPU.AmountOfWork = NumOfRandomsInWorkQuanta;
		workCPU.HostResultPtr = (char *)(&(resultArray_CPU[resultArrayIndex_CPU]));
		workCudaGPU.DeviceResultPtr = NULL;
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

int GpuGenerateWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta,
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

int OpenclGpuGenerateWork(RandomsToGenerate & genSpec, const size_t & NumOfRandomsInWorkQuanta,
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
			int returnValue = CpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 1) {	// CudaGpu work item
			int returnValue = GpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_GPU, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
			if (returnValue != 0) return returnValue;
		}
		if (NumOfWorkItems > 2) {	// OpenclGpu work item
			int returnValue = OpenclGpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, NULL, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
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
					int returnValue = CpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
					if (returnValue != 0) return returnValue;
				}
				numCpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[1] (CUDA GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + CUDA_GPU:
				//printf("ghEventsComputeDone CUDA GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					int returnValue = GpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_GPU, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
					if (returnValue != 0) return returnValue;
				}
				numCudaGpuWorkItemsDone++;
				break;
				// ghEventsComputeDone[2] (OPENCL GPU) was signaled => done with its work item
			case WAIT_OBJECT_0 + OPENCL_GPU:
				//printf("ghEventsComputeDone OpenCL GPU event was signaled.\n");
				if (inputWorkIndex < NumOfWorkItems) {
					int returnValue = OpenclGpuGenerateWork(genSpec, NumOfRandomsInWorkQuanta, randomFloatArray_GPU, resultArrayIndex_GPU, randomFloatArray_CPU, resultArrayIndex_CPU, inputWorkIndex);
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
