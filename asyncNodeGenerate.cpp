// Architecture of Heterogeneous Computing Node (CHN)
// Work queue, where multiple different type of workers come to pick up work items
// Work generators puts new work items for workers to pick up. Workers pick up new work items when they nothing to work on.
// Each type of worker implements a method to pick up new work and a method to do the work.
// There may need to be a way to return results, and in some cases to copy them to the destination memory type.
// Currently async_node is our only option, since TBB has no CUDA streaming_node implemented yet, but TBB does support OpenCL streaming_node
// Intel flowGraph webinar mentioned to advocate or contribute other streaming_node factories
// TBB has multifunction_node and multifunction_node, which have multiple outputs.
// Nodes needed are: async_cpu_rng, async_gpu_rng, opencl_gpu_rng, and opencl_fpga_rng (in the future)

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
#include <iostream>
#include <windows.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

#include "tbb/tbb.h"
#include "tbb/flow_graph.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_queue.h"
#include "tbb/compat/thread"
#include "bzlib.h"
#include <thread>
#include "CudaSupport.h"
#include "TimerCycleAccurateArray.h"
#include "asyncNodeGenerator.h"

//https://software.intel.com/en-us/videos/cpus-gpus-fpgas-managing-the-alphabet-soup-with-intel-threading-building-blocks

using namespace concurrency;
using namespace std;

using namespace tbb::flow;

extern int mklRandomDoubleParallel_SkipAhead(double * RngArray, int NumValues, unsigned int seed, int RngType, int NumCores);
extern int mklRandomFloatParallel_SkipAhead( float  * RngArray, size_t NumValues, unsigned int seed, int RngType, int NumCores);
extern void GenerateRandFloatCuda(float *devMemPtr, float *sysMemPtr, curandGenerator_t& prngGPU, size_t numRandoms, bool verify, curandGenerator_t& prngCPU);
extern int GenerateRandCuda(              int numRandoms, int seed);
extern int generateRandomFloat(float *f_Rand, int rngAlgorithm, int numRandoms, int seed, int numIterations);
extern void copyCudaToSystemMemory(void *systemMemPtr, void *cudaMemPtr, size_t numBytes);
extern void CudaThreadSynchronize();

struct Buffer {
	size_t len;		// length of buffer
	char* b;		// buffer storage
};

struct BufferResultMsg {

	BufferResultMsg() {}
	BufferResultMsg(Buffer& resultBuffer, size_t seqId, bool isLast = false)
		: resultBuffer(resultBuffer), seqId(seqId), isLast(isLast) {}

	// TODO: Do we need this method, since no one seems to be using it?
	static BufferResultMsg createBufferResultMsg(size_t seqId, size_t resultChunkSize) {
		Buffer resultBuffer;
		resultBuffer.b = new char[resultChunkSize];
		resultBuffer.len = resultChunkSize;

		return BufferResultMsg(resultBuffer, seqId);
	}

	static void destroyBufferMsg(const BufferResultMsg& destroyMsg) {
		delete[] destroyMsg.resultBuffer.b;
	}

	void markLast(size_t lastId) {
		isLast = true;
		seqId = lastId;
	}

	size_t seqId;
	Buffer resultBuffer;
	bool isLast;
};

typedef WorkItemType        gen_input_type;		// number of random floats to generate
typedef BufferResultMsg gen_output_type;	// TODO: needs to be output_type, since it's expected by gateway.try_put()

typedef tbb::flow::async_node< WorkItemType, BufferResultMsg > async_cpu_rng_node;
typedef tbb::flow::async_node< WorkItemType, BufferResultMsg > async_gpu_rng_node;
//typedef tbb::flow::async_node< BufferMsg, tbb::flow::continue_msg > async_file_writer_node;

class AsyncGenerateNodeActivity {
public:
	typedef async_cpu_rng_node::gateway_type gateway_type;
	enum ComputationalUnitType { CPU, CudaGPU };

	struct work_type {
		gen_input_type numRandFloats;
		gateway_type*  gateway;
	};
	// TODO: Turn this into a getter instead of making it public
	CudaRngEncapsulation *m_CudaRngSupport;
	string   m_cudaTimestamps[200];	// hard array to avoid dynamic allocation of std::vector at unpredictable times
	unsigned m_cudaTimestampsIndex;
	string   m_cpuTimestamps[200];
	unsigned m_cpuTimestampsIndex;

	AsyncGenerateNodeActivity(enum ComputationalUnitType compType, size_t cudaNumBytesToPreallocate, bool freeCudaMemory, unsigned long long prngSeed) :
		m_endWork(false), m_seqId(0), m_computionalUnitType(compType), m_CudaDeviceID(0), m_freeCudaMemory(freeCudaMemory),
		m_workThread(&AsyncGenerateNodeActivity::workLoop, this)
	{
		m_timer.reset();
		if (m_computionalUnitType == CudaGPU)
		{
			//printf("AsyncGenerateNodeActivity constructor (GPU)\n");
			// initialize the GPU, either identified by --device
			// or by picking the device with highest flop rate.
			int argc = 0;
			const char **argv = NULL;
			m_CudaDeviceID = findCudaDevice(argc, (const char **)argv);	// TODO: need to do this operation only once
			m_CudaRngSupport = new CudaRngEncapsulation(prngSeed, cudaNumBytesToPreallocate, m_freeCudaMemory);
		}
		printf("AsyncGenerateNodeActivity constructor has completed\n");
	}

	~AsyncGenerateNodeActivity() {
		//printf("AsyncGenerateNodeActivity destructor (GPU)\n");
		endWork();
		if (m_workThread.joinable())
			m_workThread.join();
		if (m_computionalUnitType == CudaGPU)
			delete(m_CudaRngSupport);
	}

	// Submits a new work item to the external worker to be performed asynchronously with respect to the TBB graph
	void submitWorkItem(gen_input_type i, gateway_type* gateway) {
		work_type w = { i, gateway };
		gateway->reserve_wait();
		m_workQueue.push(w);
	}

	void submitWorkToBeDone(gen_input_type i, gateway_type* gateway) {
		work_type w = { i, gateway };
		gateway->reserve_wait();
		m_workQueue.push(w);
	}

	bool endWork() {
		m_endWork = true;
		return m_endWork;
	}

private:

	void workLoop() {
		//cout << "AsyncGenerateNodeActivity.workLoop()\n";
		while (!m_endWork) {
			work_type w;
			while (m_workQueue.try_pop(w)) {
				gen_output_type result = doWork(w.numRandFloats);
				//send the result back to the graph
				w.gateway->try_put(result);
				// signal that work is done
				w.gateway->release_wait();
			}
		}
	}

	gen_output_type doWork(gen_input_type& numRandFloats)
	{
		if (m_computionalUnitType == CPU)
		{
			// TODO: Need to not allocate any memory within the result, but just result information
			gen_output_type result = BufferResultMsg::createBufferResultMsg(m_seqId, 1 * sizeof(float));	// CPU generates an array of floats
			m_seqId++;
			unsigned int rngSeed = 2;
			int rngType = VSL_BRNG_MCG59;
			int numCores = 4;
			//printf("Starting doWork() CPU: numOfRandFloats to generate is %d of floats\n", numRandFloats.amountOfWork);
			int rngResult;
			m_timer.reset();
			m_timer.timeStamp();
			rngResult = mklRandomFloatParallel_SkipAhead((float *)numRandFloats.HostResultPtr, numRandFloats.AmountOfWork, rngSeed, rngType, numCores);
			m_timer.timeStamp();
			std::stringstream outString;
			outString << "doWork: mklRandomFloatParallel_SkipAhead " << m_timer.getCycleCount(1) << " took " << m_timer.getAverageDeltaInSeconds() << " seconds and ran at " << (size_t)((double)numRandFloats.AmountOfWork / m_timer.getAverageDeltaInSeconds()) << " floats/second" << endl;
			m_cpuTimestamps[m_cpuTimestampsIndex++] = outString.str();
			//printf("doWork: mklRandomFloatParallel_SkipAhead took %f seconds and ran at %zd floats/second\n", m_timer.getAverageDeltaInSeconds(), (size_t)((double)numRandFloats.amountOfWork / m_timer.getAverageDeltaInSeconds()));
			//printf("doWork: mklRandomFloatParallel_SkipAhead took %f seconds and ran at %f floats/sec\n", elapsed / 1000.0, (double)numRandFloats.amountOfWork / (elapsed / 1000.0));
			//printf("Done generating randoms on CPU\n");
			return result;
		}
		else if (m_computionalUnitType == CudaGPU)
		{
			m_timer.reset();
			m_timer.timeStamp();
			// TODO: Need to not allocate any memory within the result, but just result information
			gen_output_type result = BufferResultMsg::createBufferResultMsg(m_seqId, 1 * sizeof(float));	// GPU generates an array of floats
			m_seqId++;
			printf("Starting doWork() GPU: numOfRandFloats to generate is %zd of floats\n", numRandFloats.AmountOfWork);
			bool verify = false;
			GenerateRandFloatCuda((float *)numRandFloats.DeviceResultPtr, (float *)numRandFloats.HostResultPtr, m_CudaRngSupport->prngGPU, numRandFloats.AmountOfWork, verify, m_CudaRngSupport->prngCPU);
			m_timer.timeStamp();
			printf("doWork: GenerateRandFloatCuda took %f seconds and ran at %zd floats/second\n", m_timer.getAverageDeltaInSeconds(), (size_t)((double)numRandFloats.AmountOfWork / m_timer.getAverageDeltaInSeconds()));
			std::stringstream outString;
			outString << "doWork: GenerateRandFloatCuda " << m_timer.getCycleCount(1) << " took " << m_timer.getAverageDeltaInSeconds() << " seconds and ran at " << (size_t)((double)numRandFloats.AmountOfWork / m_timer.getAverageDeltaInSeconds()) << " floats/second" << endl;
			m_cudaTimestamps[m_cudaTimestampsIndex++] = outString.str();
			//printf("doWork: GenerateRandFloatCuda took %f seconds and ran at %f floats/sec\n", elapsed / 1000.0, (double)numRandFloats.amountOfWork / (elapsed / 1000.0));
			//printf("Done generating randoms on GPU\n");
			return result;
		}
		gen_output_type result = BufferResultMsg::createBufferResultMsg(m_seqId, 0);	// generate an empty result
		m_seqId++;
	}

	bool m_endWork;		// must be first in the order to be initialized first in the list (this is stupid and VS should warn about this - the order of variable is not the same as the order in the init list - and this order is the one that's followed)
	enum ComputationalUnitType m_computionalUnitType;
	int m_CudaDeviceID;
	tbb::concurrent_bounded_queue< work_type > m_workQueue;
	std::thread m_workThread;
	size_t m_seqId;
	bool m_copyGPUresultsToSystemMemory;
	bool m_freeCudaMemory;
	TimerCycleAccurateArray	m_timer;
};

// first element of the tuple is the worker type, the second is the amount of work to be performed
// multifunction_node template takes an input and an output type
typedef tbb::flow::multifunction_node<WorkItemType, tbb::flow::tuple<WorkItemType, WorkItemType>> select_worker_node;

struct SelectWorkerBody
{
	void operator()(const WorkItemType& work, select_worker_node::output_ports_type &op)
	{
		//printf("SelectorWorkerBody, selecting worker with workerType %d and amountOfWork %zd\n", work.workerType, work.amountOfWork);
		if (work.WorkerType == 0) {
			std::get<0>(op).try_put(work);
		}
		else if (work.WorkerType == 1) {
			std::get<1>(op).try_put(work);
		}
	}
};

// TODO: Add type of random to generate
int RngHetero(RandomsToGenerate& genSpec, ofstream& benchmarkFile, unsigned numTimes)
{
	size_t NumOfRandomsToGenerate = genSpec.randomsToGenerate;
	size_t NumOfRandomsInWorkQuanta = genSpec.CPU.workQuanta;		// TODO: Need to separate CPU and GPU workQuanta, and handle them being different
	// TODO: Fix the problem with the case of asking the CudaGPU to generate more randoms that can fit into it's memory, but no other computational units are helping to generate more
	// TODO: One possible way to do this is to pre-determine the NumOfWorkItems and shrink it in case there is not enough memory between all of the generators
	// TODO: Another way is to create a method that takes genSpec as input and outputs all of the needed setup variables with their values for the rest of the code to use
	size_t NumOfWorkItems = (genSpec.resultDestination == ResultInCudaGpuMemory && !genSpec.CPU.helpOthers) ?
		min(genSpec.CudaGPU.maxRandoms, NumOfRandomsToGenerate) / NumOfRandomsInWorkQuanta : NumOfRandomsToGenerate / NumOfRandomsInWorkQuanta;
	size_t NumOfBytesForRandomArray = NumOfRandomsInWorkQuanta * sizeof(float);	// TODO: Need to change to NumOfRandomsToGenerate, and not be a hack of a single work item's worth of memory

	float * randomFloatArray = NULL;
	if (genSpec.generated.CPU.buffer == NULL) {		// allocate only if haven't allocated yet
		printf("Before allocation of NumOfBytesForRandomArray = %zd\n", NumOfRandomsToGenerate * sizeof(float));
		randomFloatArray = new float[NumOfRandomsToGenerate];
		// Clearing the arrays also pages them in (warming them up), which improves performance by 3X for the first generator due to first use
		// TODO: reading one byte from each page may be a faster way to warm up (page in) the array. I already have code for this.
		memset((void *)randomFloatArray, 0, NumOfRandomsToGenerate * sizeof(float));
		genSpec.generated.CPU.buffer = (char *)randomFloatArray;
	}
	else {
		randomFloatArray = (float *)genSpec.generated.CPU.buffer;
	}
	// TODO: Need to set the number of randoms generated in CPU memory and GPU memory at the end of all generation once it's known
	// TODO: Only allocate system memory when we are going to put randoms into it

	printf("After allocation of NumOfBytesForRandomArray = %zd at CPU memory location = %p\n", NumOfRandomsToGenerate * sizeof(float), randomFloatArray);
	tbb::flow::graph g;

	typedef indexer_node<gen_output_type, gen_output_type> indexer_type;
	indexer_type indexer(g);
	queue_node<indexer_type::output_type> indexer_queue(g);

	AsyncGenerateNodeActivity asyncNodeActivityCPU(AsyncGenerateNodeActivity::CPU, 0, true, genSpec.CPU.prngSeed);
	size_t preallocateGPUmemorySize;
	if (genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers)
		preallocateGPUmemorySize = NumOfRandomsInWorkQuanta * sizeof(float);	// since GPU is a helper, only pre-allocate workQuanta size in GPU memory and use the same memory buffer for each work item
	else if (genSpec.resultDestination == ResultInSystemMemory && !genSpec.CudaGPU.helpOthers)
		preallocateGPUmemorySize = 0;
	else if (genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers)
		preallocateGPUmemorySize = genSpec.CudaGPU.maxRandoms * sizeof(float);
	else if (genSpec.resultDestination == ResultInCudaGpuMemory)
		preallocateGPUmemorySize = genSpec.CudaGPU.maxRandoms * sizeof(float);
	else {
		printf("Error: Unsupported configuration\n");
		return -1;
	}
	printf("Constructung AsyncGenerateNodeActivity with preallocated CudaGPU memory of %zd\n", preallocateGPUmemorySize);
	bool freeCudaMemory = ((genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers) || (genSpec.resultDestination == ResultInCudaGpuMemory))
		? false : true;
	// TODO: This is really hacky! We need a way to set the cudaBuffer memory pointer inside AsyncGenerateNodeActivity class in a much cleaner way, possibly in constructor
	if (genSpec.generated.CudaGPU.buffer != NULL)
		preallocateGPUmemorySize = 0;
	AsyncGenerateNodeActivity asyncNodeActivityGPU(AsyncGenerateNodeActivity::CudaGPU, preallocateGPUmemorySize, freeCudaMemory, genSpec.CudaGPU.prngSeed);	// pre-allocate GPU memory, since it takes forever to allocate
	// TODO: This is really hacky! We need a way to set the cudaBuffer memory pointer inside AsyncGenerateNodeActivity class in a much cleaner way, possibly in constructor
	if (genSpec.generated.CudaGPU.buffer != NULL)
		asyncNodeActivityGPU.m_CudaRngSupport->m_gpu_memory = (void *)genSpec.generated.CudaGPU.buffer;	// restore the cudaGPU pointer to an already allocated buffer
	float *randomFloatArray_GPU = (float *)asyncNodeActivityGPU.m_CudaRngSupport->m_gpu_memory;
	if (freeCudaMemory == false) {
		genSpec.generated.CudaGPU.buffer = (char *)asyncNodeActivityGPU.m_CudaRngSupport->m_gpu_memory;
	}

	//printf("CudaGPU address = %p\n", (float *)asyncNodeActivityCPU.m_CudaRngSupport->m_gpu_memory);	// TODO: Figure out why this line crashes

	//broadcast_node<int> input_who(g);
	broadcast_node<WorkItemType> input_work(g);		// broadcast nodes allow inputs to be fed by external code
	//join_node< tuple<int, int>, queueing > join(g);
	select_worker_node selectWorker(g, unlimited, SelectWorkerBody());
	queue_node<gen_output_type> cpu_result_queue(g);
	queue_node<gen_output_type> gpu_result_queue(g);

	// Serial seems like the right parallelism here, since within this node is where the true parallelism will be
	async_cpu_rng_node cpu_rng(g, tbb::flow::serial, [&asyncNodeActivityCPU](const gen_input_type& numRandoms, async_cpu_rng_node::gateway_type& gateway) {
		asyncNodeActivityCPU.submitWorkItem(numRandoms, &gateway);
	});
	async_gpu_rng_node gpu_rng(g, tbb::flow::serial, [&asyncNodeActivityGPU](const gen_input_type& numRandoms, async_gpu_rng_node::gateway_type& gateway) {
		asyncNodeActivityGPU.submitWorkItem(numRandoms, &gateway);
	});

	// Idea: To interface with code outside of the graph, add queue, put items in it as they come in and the main thread can take items out as the become available
	// TODO: Connect the inputs and workers into the graph last, since some of the workers take longer to get initialized than others. Those workers need to be able
	// TODO: to construct quickly and be able to take on work items. If they need further initialization, these workers can do lazy finishing of their initialization
	// TODO: once the first work item is provided. The idea here is to get all of the compute units of the graph going quickly so that everyone can be handed work, even
	// TODO: workers that take a while to start up, such as CUDA GPU. This lets CPU workers with little start-up overhead get working earlier and to get some work done
	// TODO: immediately. This will help us handle small work loads efficiently as well as large ones.

	// TODO: Separate graph constuction from graph execution, for the case of running multiple inputs over the same pre-constructed graph
	// TODO: Measure how long it takes to construct the graph and to construct all of the nodes

	// Construct the graph
	make_edge(input_work, selectWorker);
	make_edge(output_port<0>(selectWorker), cpu_rng);		// compute on CPU
	make_edge(output_port<1>(selectWorker), gpu_rng);		// compute on GPU
	make_edge(output_port<0>(cpu_rng), cpu_result_queue);	// collect results from CPU generator into a queue	
	make_edge(output_port<0>(gpu_rng), gpu_result_queue);	// collect results from GPU generator into a queue
	make_edge(cpu_result_queue, input_port<0>(indexer));
	make_edge(gpu_result_queue, input_port<1>(indexer));
	make_edge(indexer, indexer_queue);
	printf("Done constructing graph\n");

	for (unsigned i = 0; i < numTimes; i++)
	{
		TimerCycleAccurateArray	timer;
		timer.reset();
		timer.timeStamp();
		// Start each worker in the graph, once the graph has been constructructed
		// TODO: Need to handle less work than enough for each of the worker type (e.g. 1.5xWorkQuanta randoms, 0.2xWorkQuanta randoms, with two available workers)
		WorkItemType work;
		size_t resultArrayIndex = 0;		// TODO: Chage the name to _CPU
		size_t resultArrayIndex_GPU = 0;
		size_t inputWorkIndex = 0;
		if (NumOfWorkItems > 0) {
			// TODO: Consider all combinations of where the randoms end up and who is allowed to help generate them. Is there a way to handle them in a general way (flags)?
			if (genSpec.resultDestination == ResultInSystemMemory || genSpec.resultDestination == ResultInEachDevicesMemory ||
				(genSpec.resultDestination == ResultInCudaGpuMemory   && genSpec.CPU.helpOthers) ||
				(genSpec.resultDestination == ResultInOpenclGpuMemory && genSpec.CPU.helpOthers)) {
				//printf("First CPU work item\n");
				work.SetGeneratorWorkType(ComputeEngine::CPU, NumOfRandomsInWorkQuanta, (char *)(&(randomFloatArray[resultArrayIndex])), NULL);
				input_work.try_put(work);
				resultArrayIndex += NumOfRandomsInWorkQuanta;
				inputWorkIndex++;
			}
		}
		if (NumOfWorkItems > 1) {
			if (genSpec.resultDestination == ResultInSystemMemory && !genSpec.CudaGPU.helpOthers)
			{
			}	// don't generate a CudaGPU work item, which will subsequently never generate another one ever
			else if (genSpec.resultDestination == ResultInCudaGpuMemory ||
				(genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers) ||
				(genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers)) {
				if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.CudaGPU.maxRandoms) {
					//printf("First CudaGPU work item\n");
					work.SetGeneratorWorkType(ComputeEngine::CUDA_GPU, NumOfRandomsInWorkQuanta, NULL, (char *)(&(randomFloatArray_GPU[resultArrayIndex_GPU])));
					if (genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers) {
						work.HostResultPtr = (char *)(&(randomFloatArray[resultArrayIndex]));
						resultArrayIndex += NumOfRandomsInWorkQuanta;		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
					}
					else {
						work.HostResultPtr = NULL;
						resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
					}
					input_work.try_put(work);
					inputWorkIndex++;
				}
			}
			else {
				printf("Error #1: Unsupported combination of genSpec\n");
				return -1;
			}
		}

		// Collect one output at a time. As a worker produces output, give that worker new work to do.
		indexer_type::output_type result;
		size_t workCompletedByCPU = 0;
		size_t workCompletedByGPU = 0;
		size_t forLoopCount = 0;
		// TODO: Instead of polling for output, it would be better to obtain output asynchronously (as a callback function), which would then trigger generation of new work item!
		// TODO: To accomplish the async (callback) behavior, we may need to put this inside a node and then have the node call an external function to trigger new work generation
		// TODO: Figure out how to handle different size workQuanta between CPU and GPU and knowing when work is done
		for (size_t outputWorkIndex = 0; outputWorkIndex < NumOfWorkItems; )
		{
			// TODO: Improve this to have try.get() as the conditional of the for loop to not waste CPU time checking if output is available.
			// TODO: (Currently, doesn't seem to be a way to do this, except to put this logic inside a node, put it in a queue and expose an interface to it)
			// TODO: (Done no andwer) Ask TBB forum on the most efficient way to wait for results from a flow graph
			if (indexer_queue.try_get(result))
			{
				//printf("Received a result at forLoopCount = %zd\n", forLoopCount);
				//printf("Received result work item %zd. Current input work item %zd. NumOfWorkItems = %zd\n", outputWorkIndex, inputWorkIndex, NumOfWorkItems);
				if (result.tag() == 0) {	// CPU
					workCompletedByCPU++;
					//printf("Received array of random floats from CPU of length %zd\n", cast_to<gen_output_type>(result).resultBuffer.len);
					if (inputWorkIndex < NumOfWorkItems)	// Create new work item for CPU
					{
						//printf("More CPU work item\n");
						work.SetGeneratorWorkType(ComputeEngine::CPU, NumOfRandomsInWorkQuanta, (char *)(&(randomFloatArray[resultArrayIndex])), NULL);
						input_work.try_put(work);
						inputWorkIndex++;
						//printf("Gave new work item to CPU. resultArrayIndex = %zd. Completed %zd work items\n", resultArrayIndex, workCompletedByCPU);
						resultArrayIndex += NumOfRandomsInWorkQuanta;
					}
				}
				else {	// CudaGPU
					//printf("Received array of random floats from CUDA GPU of length %zd bytes\n", cast_to<gen_output_type>(result).resultBuffer.len);
					workCompletedByGPU++;
					// Create new work item for CudaGPU, only if it fits into CudaGPU memory
					if (inputWorkIndex < NumOfWorkItems) {
						if (genSpec.resultDestination == ResultInCudaGpuMemory ||
							(genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers) ||
							(genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers)) {
							if ((resultArrayIndex_GPU + NumOfRandomsInWorkQuanta) < genSpec.CudaGPU.maxRandoms) {
								//printf("More CudaGPU work item\n");
								work.SetGeneratorWorkType(ComputeEngine::CUDA_GPU, NumOfRandomsInWorkQuanta, NULL, (char *)(&(randomFloatArray_GPU[resultArrayIndex_GPU])));
								//printf("resultArrayIndex_GPU = %zd\n", resultArrayIndex_GPU);
								if (genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers)
									work.HostResultPtr = (char *)(&(randomFloatArray[resultArrayIndex]));
								else
									work.HostResultPtr = NULL;
								input_work.try_put(work);
								inputWorkIndex++;
								printf("Gave new work item to GPU. resultArrayIndex = %zd. Completed %zd work items\n", resultArrayIndex_GPU, workCompletedByGPU);
								if (genSpec.resultDestination == ResultInSystemMemory && genSpec.CudaGPU.helpOthers)
									resultArrayIndex += NumOfRandomsInWorkQuanta;	// don't advance GPU index to reuse the same GPU array
								else
									resultArrayIndex_GPU += NumOfRandomsInWorkQuanta;
							}
						}
						else {
							printf("Error #2: Unsupported combination of genSpec\n");
							return -1;
						}
					}
				}
				// de-allocate message buffer memory as result messages come out - i.e. get the results and destroy result buffers
				BufferResultMsg computed_result = cast_to<BufferResultMsg>(result);
				BufferResultMsg::destroyBufferMsg(computed_result);
				//printf("Destroyed result Buffer message received\n");
				outputWorkIndex++;
			}
			else
				Sleep(0);	// TODO: Not the best way to reduce polling. Need to figure out a way to not poll efficiently
			forLoopCount++;
		}

		checkCudaErrors(cudaThreadSynchronize());	// Need to synchronize all Cuda threads to make sure all are done before measuring execution time.
		checkCudaErrors(cudaDeviceSynchronize());	// Make sure all GPU tasks have finished, before we can measure execution time
		g.wait_for_all();

		if (genSpec.resultDestination == ResultInSystemMemory && !genSpec.CudaGPU.helpOthers)
		{
			timer.timeStamp();
			printf("To generate randoms by CPU only, ran at %zd floats/second\n", (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()));
			printf("RngHetero: Ran successfully. CPU generated %zd randoms, CudaGPU generated %zd randoms\n", resultArrayIndex, resultArrayIndex_GPU);
			benchmarkFile << NumOfWorkItems * NumOfRandomsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
		}
		else if ((genSpec.resultDestination == ResultInEachDevicesMemory && !genSpec.CudaGPU.helpOthers) ||
			(genSpec.resultDestination == ResultInCudaGpuMemory && !genSpec.CudaGPU.helpOthers))
		{
			timer.timeStamp();
			printf("Just generation of randoms runs at %zd floats/second, forLoopCount = %zd\n", (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()), forLoopCount);
			printf("RngHetero: Ran successfully. CPU generated %zd randoms, CudaGPU generated %zd randoms.\nAsked to generate %zd, generated %zd\n",
				resultArrayIndex, resultArrayIndex_GPU, NumOfRandomsToGenerate, resultArrayIndex + resultArrayIndex_GPU);
			benchmarkFile << NumOfWorkItems * NumOfRandomsInWorkQuanta << "\t" << (size_t)((double)NumOfWorkItems * NumOfRandomsInWorkQuanta / timer.getAverageDeltaInSeconds()) << endl;
			timer.reset();
			timer.timeStamp();
			// Copy all of the GPU generated randoms for verification of correctness by some rudamentary statistics
			copyCudaToSystemMemory(&randomFloatArray[resultArrayIndex], randomFloatArray_GPU, resultArrayIndex_GPU * sizeof(float));
			timer.timeStamp();
			//printf("Copy from CudaGPU to CPU runs at %zd bytes/second\n", (size_t)((double)(resultArrayIndex_GPU * sizeof(float)) / timer.getAverageDeltaInSeconds()));
		}

		printf("CPU     completed %zd\n", workCompletedByCPU);
		printf("CudaGPU completed %zd\n", workCompletedByGPU);

		//for (unsigned i = 0; i < 200; i++)
		//	cout << asyncNodeActivityCPU.m_cpuTimestamps[i];
		//for (unsigned i = 0; i < 200; i++)
		//	cout << asyncNodeActivityGPU.m_cudaTimestamps[i];

		double average = 0.0;
		size_t totalRandomsGenerated = 0;
		if (genSpec.resultDestination == ResultInSystemMemory)
			totalRandomsGenerated = resultArrayIndex;
		else
			totalRandomsGenerated = resultArrayIndex + resultArrayIndex_GPU;
		for (size_t i = 0; i < totalRandomsGenerated; i++)
			average += randomFloatArray[i];
		average /= totalRandomsGenerated;
		printf("Mean = %f of %zd random values. Random array size is %zd\n", average, totalRandomsGenerated, genSpec.randomsToGenerate);

		genSpec.generated.CPU.buffer = (char *)randomFloatArray;
		genSpec.generated.CPU.numberOfRandoms = resultArrayIndex;
		genSpec.generated.CudaGPU.buffer = (char *)randomFloatArray_GPU;
		genSpec.generated.CudaGPU.numberOfRandoms = resultArrayIndex_GPU;
		//printf("Done with fgHeteroAsyncNode\n");
	}
#if 0

	AsyncNodeActivity asyncNodeActivity(io);

	async_file_reader_node file_reader(g, tbb::flow::unlimited, [&asyncNodeActivity](const tbb::flow::continue_msg& msg, async_file_reader_node::gateway_type& gateway) {
		asyncNodeActivity.submitRead(gateway);
	});
	async_cpu_rng_node cpu_rng(g, tbb::flow::unlimited, [&asyncNodeActivity](const tbb::flow::continue_msg& msg, async_file_reader_node::gateway_type& gateway) {
		asyncNodeActivity.submitRead(gateway);
	});

	tbb::flow::function_node< BufferMsg, BufferMsg > compressor(g, tbb::flow::unlimited, BufferCompressor(blockSizeIn100KB));

	// sequencer_node re-orders bufferMsg's based on (size_t bufferMsg.seqId)
	tbb::flow::sequencer_node< BufferMsg > ordering(g, [](const BufferMsg& bufferMsg)->size_t {
		return bufferMsg.seqId;
	});

	// The node is serial to preserve the right order of buffers set by the preceding sequencer_node
	async_file_writer_node output_writer(g, tbb::flow::serial, [&asyncNodeActivity](const BufferMsg& bufferMsg, async_file_writer_node::gateway_type& gateway) {
		asyncNodeActivity.submitWrite(bufferMsg);
	});

	// make_edge(work_creator, dispatcher);
	// make_edge(dispatcher, cpu_rng);			// first  output of dispatcher
	// make_edge(dispatcher, gpu_rng);			// second output of dispatcher

	make_edge(file_reader, compressor);					// create the graph
	make_edge(compressor, ordering);
	make_edge(ordering, output_writer);

	file_reader.try_put(tbb::flow::continue_msg());		// start operations up, from the front of the graph

	g.wait_for_all();
#endif
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
int GenerateHetero(RandomsToGenerate& genSpec, ofstream& benchmarkFile, unsigned NumTimes)
{

	if (genSpec.CPU.memoryCapacity < (genSpec.CPU.maxRandoms * sizeof(float))) {
		printf("Error: Maximum number of randoms for CPU memory exceeds memory capacity.\n");
		return -1;
	}
	if (genSpec.CudaGPU.memoryCapacity < (genSpec.CudaGPU.maxRandoms * sizeof(float))) {
		printf("Error: Maximum number of randoms for GPU memory exceeds memory capacity.\n");
		return -2;
	}
	if (genSpec.CPU.workQuanta == 0)
		genSpec.CPU.workQuanta     = 20 * 1024 * 1024;	// TODO: Need to define a global constant for this
	if (genSpec.CudaGPU.workQuanta == 0)
		genSpec.CudaGPU.workQuanta = 20 * 1024 * 1024;	// TODO: Need to define a global constant for this
													// TODO: Develop a way to determine optimal work chunk size, possibly dynamically or during install on that machine, or over many runs get to better and better performance
	printf("NumOfRandomsToGenerate = %zd, CPU.workQuanta = %zd, GPU.workQuanta = %zd\n", genSpec.randomsToGenerate, genSpec.CPU.workQuanta, genSpec.CudaGPU.workQuanta);

	if (genSpec.resultDestination == ResultInEachDevicesMemory)
	{
		printf("RngHetero with ResultInEachDevicesMemory\n");
		// TODO: Need to return an address, number of randoms returned and the size of memory allocated.
		// TODO: Need to NOT free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
		return RngHetero(genSpec, benchmarkFile, NumTimes);
	}
	else if (genSpec.resultDestination == ResultInCudaGpuMemory)
	{
		printf("RngHetero with ResultInCudaGPUMemory\n");
		// TODO: Need to return an address, number of randoms returned and the size of memory allocated.
		// TODO: Need to NOT free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
		RngHetero(genSpec, benchmarkFile, NumTimes);
		genSpec.generated.CPU.numberOfRandoms = 0;
	}
	else if (genSpec.resultDestination == ResultInSystemMemory)
	{
		// TODO: Need to not free that memory and make it the responsibility of the user, but provide an interface to de-allocate thru for each device.
// TODO: GPU generated memory allocation needs to be different than for non-copy case, to handle each workItem and to copy the result of each workItem
// TODO: In this way we can handle way bigger array and need to fail it if workItem's worth of randoms don't fit into GPU memory
		if (genSpec.CudaGPU.helpOthers && genSpec.CudaGPU.workQuanta > genSpec.CudaGPU.memoryCapacity)
			return -3;		// TODO: Define error return codes in the .h file we provide to the users

		RngHetero(genSpec, benchmarkFile, NumTimes);
		genSpec.generated.CudaGPU.buffer = NULL;
		genSpec.generated.CudaGPU.numberOfRandoms = 0;
	}
	return 0;
}

void runHeteroRandomGenerator()
{
	const int numRandomsToGenerate = 1000000;
	size_t workChunkSize = 1000;

	// General interface to work with I/O buffers operations
	// We may want the generator to follow the same pattern as the IOOperation and how it's used by the async_file_reader_node which parallelizes,
	// but in our case should not read a file, but instead should call CUDA or MKL random number geneators
	//Generator generator(inputStream, outputStream, workChunkSize);

	bool copyGPUresultsToSystemMemory = false;
	//RngHetero(numRandomsToGenerate, workChunkSize, copyGPUresultsToSystemMemory);
}

#endif