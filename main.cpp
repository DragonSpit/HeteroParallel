/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "tbb/tbb_config.h"
//#include "../../common/utility/utility.h"

#if __TBB_PREVIEW_ASYNC_MSG && __TBB_CPP11_LAMBDAS_PRESENT

// includes, system
#include <iostream>
#include <stdlib.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>

// basic-parallel-transform.cpp
// compile with: /EHsc
#include "ipp.h"
#include "ipps.h"
#include "mkl_vsl.h"
#include <ppl.h>
#include <random>
#include <windows.h>
#include <algorithm>
#include <ctime>

#include "tbb/tbb.h"
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_queue.h"
#include "tbb/compat/thread"
#include "bzlib.h"
#include <thread>
#include <iostream>
#include <fstream>
#include "asyncNodeGenerator.h"

//#define __TBB_PREVIEW_OPENCL_NODE 1
//#include "tbb/flow_graph_opencl_node.h"

//https://software.intel.com/en-us/videos/cpus-gpus-fpgas-managing-the-alphabet-soup-with-intel-threading-building-blocks
// https://software.intel.com/en-us/articles/tbb-flowgraph-using-streaming-node
//https://books.google.com/books?id=CqkCCwAAQBAJ&pg=PA363&lpg=PA363&dq=tbb+sending+and+receiving+data+from+flow+graph&source=bl&ots=bIuwBLu3jn&sig=ItaXoP_ogeCELsHjcldmd3qWNE0&hl=en&sa=X&ved=0ahUKEwippO6bh5bXAhVD_WMKHaUuATkQ6AEIRjAF#v=onepage&q=tbb%20sending%20and%20receiving%20data%20from%20flow%20graph&f=false
//https://stackoverflow.com/tags/tbb-flow-graph/hot?filter=all

extern int tbbAsyncNodeExample();
extern int tbbGraphExample();
extern int tbb_join_node_example();
extern void runHeteroRandomGenerator();
extern void RngHetero(size_t numRandomsToGenerate, size_t workChunkSize, bool copyGPUresultsToSystemMemory);
extern void broadcastNodeExample();
extern int asyncNodeExample();
extern int indexerNodeExample();
extern int indexerNodeExampleWithOutputAndQueue();
extern int mainThread(void);
extern int benchmarkLoadBalancer(void);
extern int openClHelloWorld();
extern int secondOnenClExample(void);
extern int clRngExample(void);
extern int ArrayFireTest(int device);
extern int ArrayFireIntegerExample(int device);
extern void generateRandomArrayInChunks(int device, size_t numChunks, size_t chunkSize, unsigned long long seed = 2);
extern int myMainSort(int argc, char **argv);


using namespace concurrency;
using namespace std;

int GenerateRandomsCUDA(int argc, char **argv);
int BenchmarkMKLparallel_SkipAhead_Double(int NumRandomValues, int seed);

int mklRandomDouble(int NumRandomValues, unsigned int seed, unsigned int numTimes)
{
	// TODO: Victor: Figure out why IPP memory allocation has trouble with 500M doubles
	//Ipp64f * pRandArray_64f = (Ipp64f *)ippsMalloc_64f(NumRandomValues);
	double * pRandArray_64f = new double[NumRandomValues];
	if (!pRandArray_64f)
	{
		wcout << L"mklRandomDouble: unable to allocate memory" << endl;
		return -1;
	}

	double average;
	VSLStreamStatePtr stream;

	/* Initializing */
	average = 0.0;
	vslNewStream(&stream, VSL_BRNG_MT19937, seed);

	/* Generating */
	for (unsigned int i = 0; i < numTimes; i++)
	{
		auto elapsed_m = time_call([&NumRandomValues, &pRandArray_64f] {
			memset(pRandArray_64f, 0, NumRandomValues * sizeof(Ipp64f));	// try clearing array between iterations to see if not starting with the same initial data makes a difference
		});
		wcout << L"memset took " << (elapsed_m / 1000.0) << " second" << endl;
		auto elapsed = time_call([&stream, &NumRandomValues, &pRandArray_64f] {
			vdRngUniform(VSL_RNG_METHOD_UNIFORMBITS64_STD, stream, NumRandomValues, pRandArray_64f, 0.0, 1.0);
		});
		wcout << L"MKL::vdRngUniform runs at " << (long long)((double)NumRandomValues / (elapsed / 1000.0)) << " doubles/second" << endl;
		std::wcout << "The range of values in array is " << *std::min_element(pRandArray_64f, pRandArray_64f + NumRandomValues) << " to " << *std::max_element(pRandArray_64f, pRandArray_64f + NumRandomValues) << endl;

		//vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, NumRandomValues, pRandArray_64f, 5.0, 2.0);

		for (int j = 0; j < NumRandomValues; j++) {
			average += pRandArray_64f[j];
		}
	}
	average /= NumRandomValues * numTimes;

	vslDeleteStream(&stream);

	std::wcout << "Sample mean of distribution = " << average << endl;

	//ippsFree(pRandArray_64f);
	free(pRandArray_64f);

	return 0;
}

int benchmarkGraph()
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
																				// What do we do when we have OpenCL device and FPGA device. This will create more combinations, but this is a good start for now
	bool copyGPUresultsToSystemMemory = false;
	RandomsToGenerate genSpec;

	size_t maxRandomsToGenerate = (size_t)500 * 1024 * 1024;
	size_t minRandomsToGenerate = (size_t)480 * 1024 * 1024;
	unsigned NumTimes = 10;
	// TODO: Seems to be a bug going down to 20M randoms - runs forever
	// TODO: We also need to not be limited to the increment being of the same size as workQuanta
	size_t randomsToGenerateIncrement = (size_t)20 * 1024 * 1024;	// workQuanta increment to make sure total work divides evenly until we can support it not

																	// !! TODO: Create a structure of time stamp and string, to be able to create an array of time stamps and their identification with additional information
																	// !! TODO: This will help debug where the delays are and help determine if the issue is in my code or in TBB itself, as we expect no iterference between cudaGPU and MKL
																	// !! TODO: when the storage of randoms is in their respective local memories. The timestamp structure may have to be a global to avoid passing it into all layers of hierarchy.
	genSpec.CPU.workQuanta = 0;		// indicates user doesn't know what to set 
	genSpec.CudaGPU.workQuanta = 0;
	genSpec.randomsToGenerate = maxRandomsToGenerate;
	genSpec.CPU.memoryCapacity = (size_t)16 * 1024 * 1024 * 1024;
	genSpec.CudaGPU.memoryCapacity = (size_t)2 * 1024 * 1024 * 1024;
	genSpec.CPU.maxRandoms = (size_t)(genSpec.CPU.memoryCapacity     * 0.50) / sizeof(float);	// use up to 50% of CPU memory for randoms
	genSpec.CudaGPU.maxRandoms = (size_t)(genSpec.CudaGPU.memoryCapacity * 0.75) / sizeof(float);	// use up to 75% of GPU memory for randoms
	genSpec.resultDestination = ResultInCudaGpuMemory;
	genSpec.CudaGPU.helpOthers = false;
	genSpec.CPU.helpOthers = false;
	genSpec.CPU.prngSeed = std::time(0);
	genSpec.CudaGPU.prngSeed = std::time(0) + 10;
	genSpec.generated.CPU.Buffer = NULL;		// NULL implies allocate memory. non-NULL implies reuse the buffer provided
	genSpec.generated.CPU.Length = 0;
	genSpec.generated.CudaGPU.Buffer = NULL;	// NULL implies allocate memory. non-NULL implies reuse the buffer provided
	genSpec.generated.CudaGPU.Length = 0;
	//printf("genSpec set\n");

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
			GenerateHetero(genSpec, benchmarkFile, NumTimes);
		});
		printf("GenerateHetero ran at an overall rate of %zd floats/second\n", (size_t)((double)genSpec.randomsToGenerate / (elapsed / 1000.0)));
	}
	delete[] genSpec.generated.CPU.Buffer;
	if ((genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers) ||
		 genSpec.resultDestination == ResultInCudaGpuMemory)
		freeCudaMemory(genSpec.generated.CudaGPU.Buffer);

	benchmarkFile.close();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" bool runTest(const int argc, const char **argv,
                        char *data, int2 *data_int2, unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
wmain(int argc, char **argv)
{
	//benchmarkGraph();

	//mainThread();
	benchmarkLoadBalancer();	// CPU and Cuda GPU RNG
	//myMainSort(argc, argv);

	//openClHelloWorld();
	//secondOnenClExample();

	//ArrayFireTest(0);
	//ArrayFireIntegerExample(1);
	//generateRandomArrayInChunks(1, 4, 4 * 1024 * 1024);
	//generateRandomArrayInChunks(1, 4, 4 * 1024 * 1024);

	//clRngExample();

	////tbbAsyncNodeExample();
	////indexerNodeExample();
	////indexerNodeExampleWithOutputAndQueue();

	//return 0;

	////tbb_join_node_example();

	//return 0;

	//tbbGraphExample();

	//// Generate random numbers on CPU
	//const int NumberOfRandomValues = 1000000000;
	//const unsigned int seed = 42;
	//const unsigned int numberOfTimes = 1;
	////mklRandomDouble(NumberOfRandomValues, seed, numberOfTimes);
	////cout << endl;

	//cout << "Generate Randoms on CudaGPU using CUDA" << endl;
	//// Generate random numbers on GPU
	//int returnValue = GenerateRandomsCUDA(argc, argv);

	//cout << "Generate Randoms on multi-core CPU using MKL and OpenMP" << endl;
	//BenchmarkMKLparallel_SkipAhead_Double(NumberOfRandomValues, seed);

	//cout << "The rest is more CudaGPU CUDA stuff" << endl;
    // input data
    //int len = 16;
    //// the data has some zero padding at the end so that the size is a multiple of
    //// four, this simplifies the processing as each thread can process four
    //// elements (which is necessary to avoid bank conflicts) but no branching is
    //// necessary to avoid out of bounds reads
    //char str[] = { 82, 111, 118, 118, 121, 42, 97, 121, 124, 118, 110, 56,
    //               10, 10, 10, 10
    //             };

    //// Use int2 showing that CUDA vector types can be used in cpp code
    //int2 i2[16];

    //for (int i = 0; i < len; i++)
    //{
    //    i2[i].x = str[i];
    //    i2[i].y = 10;
    //}

    //bool bTestResult;

    //// run the device part of the program
    //bTestResult = runTest(argc, (const char **)argv, str, i2, len);

    //std::cout << str << std::endl;

    //char str_device[16];

    //for (int i = 0; i < len; i++)
    //{
    //    str_device[i] = (char)(i2[i].x);
    //}

    //std::cout << str_device << std::endl;

    //exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
	return 0;
}

#endif