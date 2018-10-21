#pragma once
#include <iostream>
#include <fstream>
#include <ctime>

struct BufferType
{
	char*  Buffer;	// pointer to the generated results in the memory of the particular computational unit
	size_t Length;
};

struct RandomsGenerated
{
	BufferType CPU;			// spec for generated randoms in CPU/system memory
	BufferType CudaGPU;		// spec for generated randoms in graphics/GPU memory
	BufferType OpenclGPU;	// spec for generated randoms in graphics/GPU memory
};

struct SortBuffers
{
	BufferType CPU;			// spec for results in CPU/system memory
	bool allocatedByMeCpuBuffer;
	BufferType CudaGPU;		// spec for results in graphics/GPU memory
	bool allocatedByMeCudaGpuBuffer;
	BufferType OpenclGPU;	// spec for results in graphics/GPU memory
	bool allocatedByMeOpenclGpuBuffer;
};

enum ComputeEngine {
	CPU = 0, CUDA_GPU, OPENCL_GPU, FPGA
};

enum WorkType {
	GenerateRandoms = 0, Sort
};

const size_t NumComputeDoneEvents = 4;

enum ResultDestination {
	ResultInEachDevicesMemory = 0, ResultInCpuMemory, ResultInCudaGpuMemory, ResultInOpenclGpuMemory, ResultInFpgaMemory
};

// TODO: In the future, the user could specify 0 for each capacity and we discover it for them, but to start with we'll put the burden on the user
// TODO: In the future, the user could specify 0 for each work quanta and we discover the optimal value for them.
struct RandomsControlSpec
{
	size_t memoryCapacity;	// size of available memory in bytes
	size_t maxRandoms;		// up to this number of randoms to generate in memory (must divide evenly by workQuanta)
	size_t workQuanta;      // work quanta that will be done at a time (smaller is less efficient, but better load balance)
	size_t sizeOfItem;		// size of each data type item to be generated in bytes
	size_t itemsAllocated;	// number of items, each one of sizeOfItem, that have been allocated
	bool   allowedToWork;	// true => allowed to do the work
	unsigned long long prngSeed;
};

struct RandomsToGenerate
{
	enum ResultDestination resultDestination;
	size_t randomsToGenerate;	// total number of randoms to generate, possibly spread out across memories within various computational units, depending on speed of each RNG and memory capacity of each computational unit
	RandomsControlSpec CPU;
	RandomsControlSpec CudaGPU;
	RandomsControlSpec OpenclGPU;
	RandomsControlSpec FpgaGPU;
	RandomsGenerated   generated;
};

struct SortControlSpec
{
	size_t memoryCapacity;	// size of available memory in bytes
	size_t maxElements;		// up to this number of elements to sort in memory (must divide evenly by workQuanta)
	size_t workQuanta;      // work quanta that will be done at a time (smaller is less efficient, but better load balance)
	size_t sizeOfItem;		// size of each data type item to be sorted in bytes
	size_t itemsAllocated;	// number of items, each one of sizeOfItem, that have been allocated
	bool   allowedToWork;	// true => allowed to do the work
};

struct SortToDo
{
	enum ResultDestination resultDestination;
	size_t totalItemsToSort;	// total number of items to sort, possibly spread out across memories within various computational units, depending on speed and memory capacity of each computational unit
	SortControlSpec CPU;
	SortControlSpec CudaGPU;
	SortControlSpec OpenclGPU;
	SortControlSpec FpgaGPU;
	SortBuffers   Unsorted;
	SortBuffers   Sorted;
};
extern int GenerateHetero(RandomsToGenerate& genSpec, std::ofstream& benchmarkFile, unsigned NumTimes);
extern int freeCudaMemory(void *devPtr);

// Calls the provided work function and returns the number of milliseconds that it takes to call that function.
template <class Function>
__int64 time_call(Function&& f)
{
	__int64 begin = GetTickCount();
	f();
	return GetTickCount() - begin;
}

struct WorkItemType
{
	ComputeEngine ForWhichWorker;	// which type of worker this work item is intended for
	WorkType TypeOfWork;			// what type of work item this is
	size_t AmountOfWork;
	char*  HostSourcePtr;		// host   memory pointer where the source data comes from
	char*  HostResultPtr;		// host   memory pointer where the results will go
	char*  DeviceSourcePtr;		// device memory pointer where the source data comes from
	char*  DeviceResultPtr;		// device memory pointer where the results will go

};
