/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

template <typename T>
static void CudaThrustHostToHostSort(T *hostSourcePrt, T *hostResultPrt, size_t numElements)
{
	// TODO: Pre-allocate this array to speed up sorting on CUDA GPU. However, how do you know how big to make it? Probably, need to make it as big as possible and use only a portion of it.
	// TODO: This will work once we can handle sorting in chunks and merging the results
	thrust::device_vector<T> d_keys(numElements);	// Device memory used for sorting in-place

	thrust::copy(hostSourcePrt, hostSourcePrt + numElements, d_keys.begin());		// copy from Host memory to Device memory

	thrust::sort(d_keys.begin(), d_keys.end());

	//thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());
	thrust::copy(d_keys.begin(), d_keys.end(), hostResultPrt);		// copy from Device memory to Host memory

	getLastCudaError("copying results to host memory");
}

void CudaThrustHostToHostSort(unsigned *hostSourcePrt, unsigned *hostResultPrt, size_t numElements)
{
	CudaThrustHostToHostSort<unsigned>(hostSourcePrt, hostResultPrt, numElements);
}

template <typename T>
bool CudaThrustSetup(size_t numElements)
{
	findCudaDevice(0, NULL);

	int deviceID = -1;

	if (cudaSuccess == cudaGetDevice(&deviceID))
	{
		cudaDeviceProp devprop;
		cudaGetDeviceProperties(&devprop, deviceID);
		unsigned int totalMem = 2 * numElements * sizeof(T);

		if (devprop.totalGlobalMem < totalMem)
		{
			printf("Error: insufficient amount of memory to sort %d elements.\n", numElements);
			printf("%d bytes needed, %d bytes available\n", (int)totalMem, (int)devprop.totalGlobalMem);
			return false;
		}
	}
	return true;
}

template <typename T, bool floatKeys>
bool testSort(T *hostSourcePrt, T *hostResultPrt, size_t numElements, int keybits, bool quiet, unsigned numIterations)
{
    printf("Sorting %ul %d-bit %s keys only\n", numElements, keybits, floatKeys ? "float" : "unsigned int");

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    float totalTime = 0;

    for (unsigned int i = 0; i < numIterations; i++)
    {
		checkCudaErrors(cudaEventRecord(start_event, 0));

		CudaThrustHostToHostSort(hostSourcePrt, hostResultPrt, numElements);

		checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

		float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;

		totalTime /= 1.0e3f;
		printf("radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
			1.0e-6f * numElements / totalTime, totalTime, numElements);
	}

    getLastCudaError("after radixsort");

	checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));

	return true;
}

bool IsSorted(unsigned *hostBufferPtr, size_t length)
{
	for (size_t i = 0; i < length - 1; i++)
	{
		//if (i < 20)
		//	printf("%u\n", hostBufferPtr[i]);
		if (hostBufferPtr[i] > hostBufferPtr[i+1])
			return false;
	}
	return true;
}

int CudaThrustSort(unsigned *hostSourcePrt, unsigned *hostResultPrt, size_t length)
{
	CudaThrustSetup<unsigned>(length);

    bool bTestResult = false;

	for (unsigned i = 0; i < 5; i++)
	{
		//bTestResult = testSort<float, true>(argc, argv);
		bTestResult = testSort<unsigned, false>(hostSourcePrt, hostResultPrt, length, 32, true, 1);

		bTestResult = IsSorted(hostResultPrt, length);

		printf(bTestResult ? "Test passed\n" : "Test failed!\n");
	}

	return 0;
}
