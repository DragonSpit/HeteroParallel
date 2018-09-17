// Utilities and system includes
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <curand.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA Error handling

#include <cuda_runtime.h>
//#include <curand.h>
#include "CudaSupport.h"

const int    DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

float compareResults(size_t rand_n, float *h_RandGPU, float *h_RandCPU)
{
	size_t i;
	float rCPU, rGPU, delta;
	float max_delta = 0.;
	float sum_delta = 0.;
	float sum_ref = 0.;

	for (i = 0; i < rand_n; i++)
	{
		rCPU = h_RandCPU[i];
		rGPU = h_RandGPU[i];
		delta = fabs(rCPU - rGPU);
		sum_delta += delta;
		sum_ref += fabs(rCPU);

		if (delta >= max_delta)
		{
			max_delta = delta;
		}
	}

	float L1norm = (float)(sum_delta / sum_ref);
	printf("Max absolute error: %E sum_ref = %E\n", max_delta, sum_ref);
	printf("L1 norm: %E\n", L1norm);

	return L1norm;
}

double compareResults(int rand_n, double *h_RandGPU, double *h_RandCPU)
{
	int i;
	double rCPU, rGPU, delta;
	double max_delta = 0.;
	double sum_delta = 0.;
	double sum_ref = 0.;

	for (i = 0; i < rand_n; i++)
	{
		rCPU = h_RandCPU[i];
		rGPU = h_RandGPU[i];
		delta = fabs(rCPU - rGPU);
		sum_delta += delta;
		sum_ref += fabs(rCPU);

		if (delta >= max_delta)
		{
			max_delta = delta;
		}
	}

	double L1norm = (sum_delta / sum_ref);
	printf("Max absolute error: %E\n", max_delta);
	printf("L1 norm: %E\n\n", L1norm);

	return L1norm;
}

void *allocateCudaMemory(size_t numBytes)
{
	void *devMemPtr;
	// TODO: Figured out that CUDA memory allocation takes a herendously long time (seconds)!! We'll need to work on avoiding this operation! Otherwise, it's killing all of GPU benefit!
	// TODO: Need to pre-allocate a large memory instead (as big as the entire array, in case if GPU will be the only one generating the results. Actually, N-1 work arrays
	printf("Before CUDA memory allocation of %zd bytes\n", numBytes);
	checkCudaErrors(cudaMalloc((void **)&devMemPtr, numBytes));		// allocate in Cuda memory
	//checkCudaErrors(cudaMallocManaged((void **)&devMemPtr, numBytes));	// allocate in shared/managed memory
	//memset((void *)devMemPtr, 0, numBytes);		// page system memory in for faster first use
	//checkCudaErrors(cudaMallocHost((void **)&devMemPtr, numBytes));	// allocate in host memory (free'ing failed for some reason, after several allocations)
	//checkCudaErrors(cudaHostAlloc((void **)&devMemPtr, numBytes, cudaHostAllocDefault));	// allocate in host memory, works better than cudaMallocHost, but needs a matching cudaFreeHost
	if (!devMemPtr)
		printf("CUDA error: couldn't allocate CUDA memory\n");
	//printf("After allocation of CUDA memory\n");
	printf("CUDA allocated memory %p number of bytes %zd\n", (void *)(devMemPtr), numBytes);
	return devMemPtr;
}

int freeCudaMemory(void * devMemPtr)
{
	printf("CUDA memory free %p\n", devMemPtr);
	checkCudaErrors(cudaFree(devMemPtr));
	//checkCudaErrors(cudaFreeHost(devMemPtr));		// matches cudaHostAlloc
	return 0;
}

void createCudaPrng(curandGenerator_t& prng, unsigned long long seed, curandRngType_t rngAlgorithm)
{
	checkCudaErrors(curandCreateGenerator(&prng, rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prng, seed));
	//printf("After creation of CUDA PRNG\n");
}

void createCudaPrngHost(curandGenerator_t& prng, unsigned long long seed, curandRngType_t rngAlgorithm)
{
	checkCudaErrors(curandCreateGeneratorHost(&prng, rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prng, seed));
	//printf("After creation of CUDA PRNG\n");
}

void freeCudaPrng(curandGenerator_t& prngGPU)
{
	checkCudaErrors(curandDestroyGenerator(prngGPU));
}

void copyCudaToSystemMemory(void *systemMemPtr, void *cudaMemPtr, size_t numBytes)
{
	//printf("Reading back the GPU RNG results %p %p, number of bytes = %zd...\n", systemMemPtr, cudaMemPtr, numBytes);
	checkCudaErrors(cudaMemcpy(systemMemPtr, cudaMemPtr, numBytes, cudaMemcpyDeviceToHost));
}

int setupCudaRandomGenerator(curandGenerator_t& prngGPU, int rngAlgorithm, unsigned long long seed)
{
	//curandGenerator_t prngGPU;
	printf("Before creation of CUDA PRNG\n");
	checkCudaErrors(curandCreateGenerator(&prngGPU, (curandRngType_t)rngAlgorithm));
	printf("After creation of CUDA PRNG\n");
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));
	printf("After seeding CUDA PRNG\n");
	return 0;
}
int generateRandomFloat(curandGenerator_t& prngGPU, float *f_Rand, size_t numRandoms)
{
	//float *f_Rand;
	//checkCudaErrors(cudaMalloc((void **)&f_Rand, numRandoms * sizeof(float)));
	//printf("CUDA #2 allocated memory %p number of bytes %zd\n", f_Rand, numRandoms * sizeof(float));
#if 0
	curandGenerator_t prngCPU;
	checkCudaErrors(curandCreateGeneratorHost(&prngCPU, (curandRngType_t)rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngCPU, seed));

	//
	// Example 1: Compare random numbers generated on GPU and CPU
	float *h_RandGPU = (float *)malloc(numRandoms * sizeof(float));
#endif
	//printf("Generating random numbers on CudaGPU...\n");
	checkCudaErrors(curandGenerateUniform(prngGPU, f_Rand, numRandoms));
#if 0
	printf("\nReading back the results...\n");
	checkCudaErrors(cudaMemcpy(h_RandGPU, d_Rand, numRandoms * sizeof(float), cudaMemcpyDeviceToHost));

	float *h_RandCPU = (float *)malloc(numRandoms * sizeof(float));

	printf("Generating random numbers on CPU...\n\n");
	checkCudaErrors(curandGenerateUniform(prngCPU, (float *)h_RandCPU, numRandoms));

	printf("Comparing CPU/CudaGPU random numbers...\n\n");
	float L1norm = compareResults(numRandoms, h_RandGPU, h_RandCPU);
#endif
#if 0
	//
	// Example 2: Timing of random number generation on GPU
	//const int numIterations = 10;
	int i;
	StopWatchInterface *hTimer;

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (i = 0; i < numIterations; i++)
	{
		checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_Rand, numRandoms));
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

	printf("%d algorithm, Throughput = %.4f GFloats/s, Time = %.5f s, Size = %u Numbers\n",
		rngAlgorithm, 1.0e-9 * numRandoms / gpuTime, gpuTime, numRandoms);
#endif
	//printf("CudaGPU Shutting down...\n");

	//checkCudaErrors(curandDestroyGenerator(prngGPU));
#if 0
	checkCudaErrors(curandDestroyGenerator(prngCPU));
#endif
	//checkCudaErrors(cudaFree(f_Rand));
#if 0
	sdkDeleteTimer(&hTimer);
	free(h_RandGPU);
	free(h_RandCPU);

	return(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
#endif
	return 0;
}

int generateRandomFloat_CudaGenOnly(int rngAlgorithm, int numRandoms, int seed)
{
	float *f_Rand;
	checkCudaErrors(cudaMalloc((void **)&f_Rand, numRandoms * sizeof(float)));
	//printf("CUDA allocated memory at %p of size %zd\n", f_Rand, numRandoms * sizeof(float));

	curandGenerator_t prngGPU;
	checkCudaErrors(curandCreateGenerator(&prngGPU, (curandRngType_t)rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));
	//printf("After creation of CUDA PRNG\n");

	//printf("Generating random numbers on GPU...\n");
	checkCudaErrors(curandGenerateUniform(prngGPU, f_Rand, numRandoms));
	//printf("GPU done generating randoms\n");

	checkCudaErrors(curandDestroyGenerator(prngGPU));
	checkCudaErrors(cudaFree(f_Rand));

	return 0;
}

void CudaThreadSynchronize()
{
	// Cuda RNG calls are asynchronous. To measure performance properly we need to synchronize here or once at the very end of all Cuda work items, for proper measurement and proper use
	cudaThreadSynchronize();		// TODO: I added this, which was not part of examples. Is it really necessary? Does it make benchmarking more accurate, since we actually finish generation. It may be that we currently get fooled into correct results because copy to the host is so slow, that async RNG always gets done in time
}

void GenerateRandFloatCuda(float *devMemPtr, float *sysMemPtr, curandGenerator_t& prngGPU, size_t numRandoms, bool verify, curandGenerator_t& prngCPU)
{
	//printf("GPU generating random floats...\n");
	checkCudaErrors(curandGenerateUniform(prngGPU, devMemPtr, numRandoms));
	//checkCudaErrors(cudaDeviceSynchronize());	// Need to synchronize all Cuda threads to make sure all are done before measuring execution time.
	checkCudaErrors(cudaThreadSynchronize());	// Need to synchronize all Cuda threads to make sure all are done before measuring execution time.
												//printf("GPU done generating random floats\n");

	if (sysMemPtr)
		checkCudaErrors(cudaMemcpy((void *)sysMemPtr, devMemPtr, numRandoms * sizeof(float), cudaMemcpyDeviceToHost));

	if (verify)
	{
		// Verify by comparing random numbers generated on GPU with ones generated on CPU
		float *h_RandGPU = new float[numRandoms];
		printf("Reading back the CudaGPU RNG results %p %p, numRandoms = %zd...\n", h_RandGPU, devMemPtr, numRandoms);
		checkCudaErrors(cudaMemcpy(h_RandGPU, devMemPtr, numRandoms * sizeof(float), cudaMemcpyDeviceToHost));

		float *h_RandCPU = new float[numRandoms];
		printf("Generating CUDA random numbers on CPU for comparison with CudaGPU...\n");
		checkCudaErrors(curandGenerateUniform(prngCPU, h_RandCPU, numRandoms));

		printf("Comparing CPU/CudaGPU random numbers...\n");
		float L1norm = compareResults(numRandoms, h_RandGPU, h_RandCPU);

		if (L1norm < 1e-6 )
			printf("Comparison of CudaGPU and CPU RNG using CUDA algorithm succeeded\n");
		else {
			printf("Comparison of CudaGPU and CPU RNG using CUDA algorithm failed\n");
			exit(1);
		}
		delete[] h_RandCPU;
		delete[] h_RandGPU;
	}
}
int generateRandomDouble(int devID, int rngAlgorithm, int numRandoms, int seed, int numIterations)
{
	double *d_Rand;
	checkCudaErrors(cudaMalloc((void **)&d_Rand, numRandoms * sizeof(double)));

	curandGenerator_t prngGPU;
	checkCudaErrors(curandCreateGenerator(&prngGPU, (curandRngType_t)rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));

	curandGenerator_t prngCPU;
	checkCudaErrors(curandCreateGeneratorHost(&prngCPU, (curandRngType_t)rngAlgorithm));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngCPU, seed));

	//
	// Example 1: Compare random numbers generated on GPU and CPU
	double *h_RandGPU = (double *)malloc(numRandoms * sizeof(double));

	printf("Generating random numbers on CudaGPU...\n\n");
	checkCudaErrors(curandGenerateUniformDouble(prngGPU, (double *)d_Rand, numRandoms));

	printf("\nReading back the results...\n");
	checkCudaErrors(cudaMemcpy(h_RandGPU, d_Rand, numRandoms * sizeof(double), cudaMemcpyDeviceToHost));


	double *h_RandCPU = (double *)malloc(numRandoms * sizeof(double));

	printf("Generating random numbers on CPU...\n\n");
	checkCudaErrors(curandGenerateUniformDouble(prngCPU, (double *)h_RandCPU, numRandoms));

	printf("Comparing CPU/CudaGPU random numbers...\n\n");
	double L1norm = compareResults(numRandoms, h_RandGPU, h_RandCPU);

	//
	// Example 2: Timing of random number generation on GPU
	//const int numIterations = 10;
	int i;
	StopWatchInterface *hTimer;

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (i = 0; i < numIterations; i++)
	{
		checkCudaErrors(curandGenerateUniformDouble(prngGPU, (double *)d_Rand, numRandoms));
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

	printf("%d algorithms, Throughput = %.4f GDoubles/s, Time = %.5f s, Size = %u Numbers\n",
		rngAlgorithm, 1.0e-9 * numRandoms / gpuTime, gpuTime, numRandoms);

	printf("Shutting down...\n");

	checkCudaErrors(curandDestroyGenerator(prngGPU));
	checkCudaErrors(curandDestroyGenerator(prngCPU));
	checkCudaErrors(cudaFree(d_Rand));
	sdkDeleteTimer(&hTimer);
	free(h_RandGPU);
	free(h_RandCPU);

	return(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}

int GenerateRandomsCUDA(int argc, char **argv)
{
	// Start logs
	printf("%s Starting...\n\n", argv[0]);

	// initialize the GPU, either identified by --device
	// or by picking the device with highest flop rate.
	int devID = findCudaDevice(argc, (const char **)argv);

	// parsing the number of random numbers to generate
	//int numRandoms = DEFAULT_RAND_N;
	size_t numRandoms = 20000000;
	unsigned NumTimes = 1000;

	if (checkCmdLineFlag(argc, (const char **)argv, "count"))
	{
		numRandoms = getCmdLineArgumentInt(argc, (const char **)argv, "count");
	}

	printf("Allocating data for %zd samples...\n", numRandoms);
	//double *d_Rand;
	//checkCudaErrors(cudaMalloc((void **)&d_Rand, numRandoms * sizeof(double)));
	float *f_Rand;
	checkCudaErrors(cudaMalloc((void **)&f_Rand, numRandoms * sizeof(float)));

	// parsing the seed
	int seed = DEFAULT_SEED;

	if (checkCmdLineFlag(argc, (const char **)argv, "seed"))
	{
		seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
	}

	printf("Seeding with %i ...\n", seed);

	const int NumRngTypes = 5;
	int arrayOfRngTypes[NumRngTypes] = {	CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
											CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10 };

	printf("CudaGPU Random Number Generation Algorithms (Floats)\n");

	curandGenerator_t prngGPU[NumRngTypes];
	for (unsigned rngAlgorithm = 0; rngAlgorithm < NumRngTypes; rngAlgorithm++)
	{
		printf("CudaGPU RNG algorithm %d\n", rngAlgorithm);
		setupCudaRandomGenerator(prngGPU[rngAlgorithm], (curandRngType_t)arrayOfRngTypes[rngAlgorithm], seed);

		for (unsigned i = 0; i < NumTimes; i++)
		{
			//printf("CudaGPU generating %u time\n", i);
			generateRandomFloat(prngGPU[rngAlgorithm], f_Rand, numRandoms);
		}
		checkCudaErrors(cudaThreadSynchronize());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(curandDestroyGenerator(prngGPU[rngAlgorithm]));
	}

	//printf("CudaGPU Random Number Generation Algorithms (Doubles)\n");

	//for (unsigned k = 0; k < NumRngTypes; k++)
	//{
	//	generateRandomDouble(devID, arrayOfRngTypes[k], numRandoms, seed, 1);
	//}

	checkCudaErrors(cudaFree(f_Rand));
	//checkCudaErrors(cudaFree(d_Rand));

	return 1;
}

//enum CudaRandType {
//	CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
//	CURAND_RNG_PSEUDO_MT19937, CURAND_RNG_PSEUDO_PHILOX4_32_10
//};
enum CudaRandDataType { Float, Double };

void GenerateSpecificRandomsCUDA(enum CudaRandDataType getType, void *devMemPtr, int numRandoms)
{
	//if (getType == Float)
	//	generateRandomFloat((float *)devMemPtr, numRandoms);
	//else if (getType == Double)
	//	generateRandomDouble(0, RndType, numRandoms, seed, 1);
}

int GenerateSpecificRandomsCUDA(int numRandoms, int seed, enum CudaRandDataType getType, enum curandRngType RndType)
{
	if (getType == Float)
		generateRandomFloat_CudaGenOnly(RndType, numRandoms, seed);
	else if (getType == Double)
		generateRandomDouble(0, RndType, numRandoms, seed, 1);

	return 1;
}

int GenerateRandCuda(int numRandoms, int seed)
{
	GenerateSpecificRandomsCUDA(numRandoms, seed, Float, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	return 1;
}

int GenerateRandCuda(int numRandoms)
{
	//GenerateSpecificRandomsCUDA(numRandoms, Float, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	return 1;
}