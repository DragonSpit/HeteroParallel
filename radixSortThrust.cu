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

template <typename T, bool floatKeys>
bool testSort(unsigned *hostSourcePrt, unsigned numElements, int keybits, bool keysOnly, bool quiet, unsigned numIterations)
{
    printf("Sorting %d %d-bit %s keys %s\n", numElements, keybits, floatKeys ? "float" : "unsigned int", keysOnly ? "only" : "and values");

    int deviceID = -1;

    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = (keysOnly ? 2 : 4) * numElements * sizeof(T);

        if (devprop.totalGlobalMem < totalMem)
        {
            printf("Error: insufficient amount of memory to sort %d elements.\n", numElements);
            printf("%d bytes needed, %d bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(EXIT_SUCCESS);
        }
    }

    thrust::host_vector<T> h_keys(numElements);
    thrust::host_vector<T> h_keysSorted(numElements);
    thrust::host_vector<unsigned int> h_values;

    if (!keysOnly)
        h_values = thrust::host_vector<unsigned int>(numElements);

    // Fill up with some random data
    thrust::default_random_engine rng(clock());

    if (floatKeys)
    {
        thrust::uniform_real_distribution<float> u01(0, 1);

        for (int i = 0; i < (int)numElements; i++)
            h_keys[i] = u01(rng);
    }
    else
    {
        thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

        for (int i = 0; i < (int)numElements; i++)
            h_keys[i] = u(rng);
    }

    if (!keysOnly)
        thrust::sequence(h_values.begin(), h_values.end());

    // Copy data onto the GPU
    thrust::device_vector<T> d_keys(numElements);
    thrust::device_vector<unsigned int> d_values;

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    float totalTime = 0;

    for (unsigned int i = 0; i < numIterations; i++)
    {
		checkCudaErrors(cudaEventRecord(start_event, 0));

		// reset data before sort
        //d_keys = h_keys;			// copy from Host memory to Device memory
		//thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());					// another way to copy from Host memory to Device memory
		thrust::copy(hostSourcePrt, hostSourcePrt + numElements, d_keys.begin());		// another way to copy from Host memory to Device memory

        if (!keysOnly)
            d_values = h_values;

        if (keysOnly)
            thrust::sort(d_keys.begin(), d_keys.end());
        else
            thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

		// Get results back to host for correctness checking
		thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

		if (!keysOnly)
			thrust::copy(d_values.begin(), d_values.end(), h_values.begin());

		getLastCudaError("copying results to host memory");

		checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));

		// Check results
		bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

		if (!bTestResult && !quiet)
		{
			printf("Error: Result array is not sorted\n");
			return false;
		}

		float time = 0;
        checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
        totalTime += time;
    }

    totalTime /= (1.0e3f * numIterations);
    printf("radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements\n",
           1.0e-6f * numElements / totalTime, totalTime, numElements);

    getLastCudaError("after radixsort");

	checkCudaErrors(cudaEventDestroy(start_event));
	checkCudaErrors(cudaEventDestroy(stop_event));

	return true;
}

int CudaThrustSort(unsigned *hostSourcePrt, size_t length)
{
    // Start logs
    //printf("%d %s Starting...\n\n", argc, argv[0]);

	findCudaDevice(0, NULL);

    bool bTestResult = false;

	for (unsigned i = 0; i < 11; i++)
	{
		//bTestResult = testSort<float, true>(argc, argv);
		bTestResult = testSort<unsigned int, false>(hostSourcePrt, length, 32, true, true, 1);

		printf(bTestResult ? "Test passed\n" : "Test failed!\n");
	}

	return 0;
}

