#pragma once
#include <iostream>
#include <fstream>
#include <ctime>

// TODO: use the generic buffer type here with a pointer and length, since this has nothing to do with randoms
struct RandomsGenSpec
{
	char*  buffer;				// pointer to generated randoms in memory of a computational unit
	size_t numberOfRandoms;
};

struct RandomsGenerated
{
	RandomsGenSpec CPU;			// spec for generated randoms in CPU/system memory
	RandomsGenSpec CudaGPU;		// spec for generated randoms in graphics/GPU memory
};

enum ComputeEngine {
	CPU = 0, CUDA_GPU, OPENCL_GPU, FPGA
};

const size_t NumComputeDoneEvents = 4;

enum ResultDestination {
	ResultInEachDevicesMemory = 0, ResultInSystemMemory, ResultInCudaGpuMemory, ResultInOpenclGpuMemory, ResultInFpgaMemory
};

// TODO: In the future, the user could specify 0 for each capacity and we discover it for them, but to start with we'll put the burden on the user
// TODO: In the future, the user could specify 0 for each work quanta and we discover the optimal value for them.
struct RandomsControlSpec
{
	size_t memoryCapacity;	// size of available memory in bytes
	size_t maxRandoms;		// up to this number of randoms to generate in memory (must divide evenly by workQuanta)
	size_t workQuanta;      // work quanta that will be done at a time (smaller is less efficient, but better load balance)
	bool   helpOthers;		// true => allowed to help other computational units with their work
	unsigned long long prngSeed;
};

struct RandomsToGenerate
{
	enum ResultDestination resultDestination;
	size_t randomsToGenerate;	// total number of randoms to generate, possibly spread out across memories within various computational units, depending on speed of each RNG and memory capacity of each computational unit
	RandomsControlSpec CPU;
	RandomsControlSpec CudaGPU;
	RandomsGenerated   generated;
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

struct WorkType
{
	ComputeEngine WorkerType;	// what kind of worker this is
	size_t AmountOfWork;
	char*  HostPtr;				// host memory pointer
	char*  DevicePtr;			// device memory pointer

	void SetWorkType(ComputeEngine workerType, size_t amountOfWord, char* hostPtr, char* devicePtr)
	{
		workerType = workerType;
		AmountOfWork = amountOfWord;
		HostPtr = hostPtr;
		DevicePtr = devicePtr;
	}
};
