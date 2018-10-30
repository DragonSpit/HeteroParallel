// TODO: Separate the load balancer itself into a separate piece and abstract it well to have a simple interface where the user provides work items into a queue
//       as there is room in the queue and the queue depth can be specified. At first, we may need a queue for each type of worker, and as we mature aspire to have
//       a single queue. It's ok to start the interface with a single item for each compute engine (worker) as we have it now, and progress toward better interface.
//       Single queue may be problematic, as a worker can get stuck behind others not completing their work fast enough and the faster worker not being able to get
//       new work items out of the queue. It may be better to have a queue for each worker type, or be able to find a work item for a particular worker, unless all
//       work items are such that any worker can complete them.

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
//extern void RngHetero(size_t numRandomsToGenerate, size_t workChunkSize, bool copyGPUresultsToSystemMemory);
extern void broadcastNodeExample();
//extern int asyncNodeExample();
extern int indexerNodeExample();
extern int indexerNodeExampleWithOutputAndQueue();
extern int mainThread(void);
extern int benchmarkRngLoadBalancer(void);
extern int benchmarkSortLoadBalancer();
extern int openClHelloWorld();
extern int secondOnenClExample(void);
extern int clRngExample(void);
extern int ArrayFireTest(int device);
extern int ArrayFireIntegerExample(int device);
extern void generateRandomArrayInChunks(int device, size_t numChunks, size_t chunkSize, unsigned long long seed = 2);
extern int CudaThrustSort(int argc, char **argv);
extern int CudaThrustSort(unsigned *hostSourcePrt, unsigned *hostResultPrt, size_t length);


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
	//mainThread();
	//benchmarkRngLoadBalancer();		// RNG     by Multi-core CPU, Cuda GPU, and OpenCL GPU
	benchmarkSortLoadBalancer();	// Sorting by Multi-core CPU, Cuda GPU, and OpenCL GPU
#if 0
	size_t length = 20 * 1024 * 1024;
	unsigned * thrustTestSourceBuffer = new unsigned[length];
	unsigned * thrustTestResultBuffer = new unsigned[length];

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	for (size_t i = 0; i < length; i++)
		thrustTestSourceBuffer[i] = dist(mt) * UINT_MAX;

	CudaThrustSort(thrustTestSourceBuffer, thrustTestResultBuffer, length);

	delete[] thrustTestSourceBuffer;
	delete[] thrustTestResultBuffer;
#endif
	//CudaThrustSort(argc, argv);
	//openClHelloWorld();
	//secondOnenClExample();

	//ArrayFireTest(0);	// CUDA GPU (950M)
	//ArrayFireTest(1);	// Intel OpenCL GPU (530)
	//ArrayFireTest(2);	// Intel OpenCL quad-core CPU (i5-6300HQ)

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