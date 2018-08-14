#include "mkl_vsl.h"
#include <ppl.h>
#include <random>
#include <iostream>
#include <sstream>
#include <windows.h>
#include <algorithm>
#include "TimerCycleAccurateArray.h"

using namespace concurrency;
using namespace std;

// Calls the provided work function and returns the number of milliseconds that it takes to call that function.
template <class Function>
__int64 time_call(Function&& f)
{
	__int64 begin = GetTickCount();
	f();
	return GetTickCount() - begin;
}

TimerCycleAccurateArray	timer;		// global because being local inside loops was just too slow (about 1 second per creation)

// Touch one byte per OS page to page in the entire array
// Otherwise the first use of the array pays the performance penalty
// must use the return value, such as printing it out, otherwise the compiler optimizes the pageInBuffer function away, since its result is not being used
int pageInBuffer(void * buffer, size_t numBytes)
{
	unsigned pageSize = 4 * 1024;	// page size can probably discovered for each OS, and only once
	unsigned long average = 0;		// need to return something, otherwise the compiler will optimize the reads
	byte * pByteArray = (byte *)buffer;
	for (size_t i = 0; i < numBytes; i += pageSize)	// page in the entire array into system memory
		average += pByteArray[i];
	return average;
}

// Parallel function that fills an array with random numbers using variety of supported algorithms
// For filling in a real array of random numbers, the tail needs to be handled properly, since the array will most likely not divide evenly into the number of cores
// TODO: Handle arbitrary array sizes that may not divide evenly by the number of cores
// TODO: Make sure to pass in a different seed each time, if we keep seeding the RNG all of the time, or figure out how to pre-construct Rng stream separately from using it over and over
int mklRandomFloatParallel_SkipAhead(float * RngArray, size_t NumValues, unsigned int seed, int RngType, int NumCores)
{
	size_t nSamplesPerCore = NumValues / NumCores;					// TODO: Wont's divide evenly and will leave up to NumCores-1 elements (tail) to handle sequentially
	size_t nSkip = nSamplesPerCore;
	float** arrayOfPtrs = new float*[NumCores];					// array of pointers
	for (size_t i = 0; i < NumCores; i++)
		arrayOfPtrs[i] = &RngArray[i * nSkip];

#pragma omp parallel for
	for (int k = 0; k < NumCores; k++)
	{
		int returnCode;
		//TimerCycleAccurateArray	timer;		// local and independent for each task
		//std::wstringstream msg;
		//double average = 0.0;
		VSLStreamStatePtr stream;
		vslNewStream(&stream, RngType, seed);				// fast operation
		int status = vslSkipAheadStream(stream, nSkip*k);	// fast operation
		if (!(status == VSL_ERROR_OK || status == VSL_STATUS_OK))
		{
			//msg << "RNG type " << RngType << " does not support Skip Ahead parallel method" << endl;
			//wcout << msg.str();
			break;
		}
		//msg.imbue(std::locale(""));	// prints integers with commas separating every three digits
#if 0
		auto elapsed = time_call([&stream, &nSamplesPerCore, &arrayOfPtrs, &returnCode, &k] {
			returnCode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, nSamplesPerCore, arrayOfPtrs[k], 0.0f, 1.0f);
		});
		msg << L"MKL::vsRngUniform of RNG type " << RngType << " runs at " << (long long)((double)nSamplesPerCore / (elapsed / 1000.0)) << " floats/second, with return value " << returnCode << " k = " << k << endl;
#else
		//timer.reset();
		//timer.timeStamp();
		//returnCode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, nSamplesPerCore, arrayOfPtrs[k], 0.0f, 1.0f);
		//auto elapsed = time_call([&] {
			returnCode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int)nSamplesPerCore, arrayOfPtrs[k], 0.0f, 1.0f);
		//});
		//wcout << L"vsRngUniform took " << (elapsed / 1000.0) << "seconds and ran at " << (long long)((double)nSamplesPerCore / (elapsed / 1000.0)) << " floats/second" << endl;
		//printf("vsRngUniform took %f seconds and ran at %f floats/sec\n", elapsed / 1000.0, (double)nSamplesPerCore / (elapsed / 1000.0));
		//timer.timeStamp();
		//msg << L"MKL::vsRngUniform of RNG type " << RngType << " runs at " << (long long)((double)nSamplesPerCore / timer.getAverageDeltaInSeconds()) << " floats/second, with return value " << returnCode << endl;
#endif
		//msg << L"The range of values in array is " << *std::min_element(arrayOfPtrs[k], arrayOfPtrs[k] + nSamplesPerCore) << " to " << *std::max_element(arrayOfPtrs[k], arrayOfPtrs[k] + nSamplesPerCore) << ".  ";

		//for (int j = 0; j < nSamplesPerCore; j++)
		//	average += arrayOfPtrs[k][j];
		//average /= nSamplesPerCore;
		//msg << "Mean = " << average << endl;
		//wcout << msg.str();

		vslDeleteStream(&stream);
	}
	delete[] arrayOfPtrs;
	return 0;
}

// Same function as above, but for double
// TODO: Once it's working turn these two functions into a template function that handles both floating-point types and integer types supported by MKL
int mklRandomDoubleParallel_SkipAhead(double * RngArray, int NumValues, unsigned int seed, int RngType, int NumCores)
{
	int nSamplesPerCore = NumValues / NumCores;					// TODO: Wont's divide evenly and will leave up to NumCores-1 elements (tail) to handle sequentially
	int nSkip = nSamplesPerCore;
	double** arrayOfPtrs = new double*[NumCores];				// array of pointers
	for (size_t i = 0; i < NumCores; i++)
		arrayOfPtrs[i] = &RngArray[i * nSkip];

#pragma omp parallel for
	for (int k = 0; k < NumCores; k++)
	{
		int returnCode;
		// TODO: Timer is causing some sort of severe performance trouble. Move timer to higher level surrounding the entire operation instead of individual ones
		//       maybe MKL will perform better then!!!
		//TimerCycleAccurateArray	timer;		// local and independent for each task
		//std::wstringstream msg;
		//double average = 0.0;
		VSLStreamStatePtr stream;
		vslNewStream(&stream, RngType, seed);
		int status = vslSkipAheadStream(stream, nSkip*k);
		if (!(status == VSL_ERROR_OK || status == VSL_STATUS_OK))
		{
			//msg << "RNG type " << RngType << " does not support Skip Ahead parallel method" << endl;
			//wcout << msg.str();
			break;
		}
		//msg.imbue(std::locale(""));	// prints integers with commas separating every three digits

#if 0
		auto elapsed = time_call([&stream, &nSamplesPerCore, &arrayOfPtrs, &returnCode, &k] {
			returnCode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, nSamplesPerCore, arrayOfPtrs[k], 0.0f, 1.0f);
		});
		msg << L"MKL::vsRngUniform of RNG type " << RngType << " runs at " << (long long)((double)nSamplesPerCore / (elapsed / 1000.0)) << " floats/second, with return value " << returnCode << " k = " << k << endl;
#else
		//timer.reset();
		//timer.timeStamp();
		returnCode = vdRngUniform(VSL_RNG_METHOD_UNIFORMBITS64_STD, stream, nSamplesPerCore, arrayOfPtrs[k], 0.0L, 1.0L);
		//timer.timeStamp();
		//msg << L"MKL::vdRngUniform of RNG type " << RngType << " runs at " << (long long)((double)nSamplesPerCore / timer.getAverageDeltaInSeconds()) << " doubles/second, number of Doubles " << nSamplesPerCore << " with return value " << returnCode << endl;
#endif
		//msg << L"The range of values in array is " << *std::min_element(arrayOfPtrs[k], arrayOfPtrs[k] + nSamplesPerCore) << " to " << *std::max_element(arrayOfPtrs[k], arrayOfPtrs[k] + nSamplesPerCore) << ".  ";

		//for (int j = 0; j < nSamplesPerCore; j++)
		//	average += arrayOfPtrs[k][j];
		//average /= nSamplesPerCore;
		//msg << "Mean = " << average << endl;
		//wcout << msg.str() << flush;

		vslDeleteStream(&stream);
	}
	delete[] arrayOfPtrs;
	return 0;
}

int BenchmarkMKLparallel_SkipAhead_Double(int NumRandomValues, int seed)
{
	double * randomDoubleArray_1 = new double[NumRandomValues];
	double * randomDoubleArray_2 = new double[NumRandomValues];
	int returnValue;

	returnValue = pageInBuffer((void *)randomDoubleArray_1, NumRandomValues * sizeof(double));
	std::wcout << "Array warm up returned " << returnValue << endl;		// must use the return value, such as printing it out, otherwise the compiler optimizes the pageInBuffer function away, since its result is not being used
	returnValue = pageInBuffer((void *)randomDoubleArray_2, NumRandomValues * sizeof(double));
	std::wcout << "Array warm up returned " << returnValue << endl;		// must use the return value, such as printing it out, otherwise the compiler optimizes the pageInBuffer function away, since its result is not being used

	int arrayOfRngTypes[10] = { VSL_BRNG_MCG31, VSL_BRNG_MRG32K3A, VSL_BRNG_MCG59, VSL_BRNG_WH, VSL_BRNG_MT19937, VSL_BRNG_SFMT19937, VSL_BRNG_SOBOL, VSL_BRNG_NIEDERR,
		VSL_BRNG_PHILOX4X32X10, VSL_BRNG_ARS5 };
	for (unsigned k = 0; k < 10; k++)
	{
		std::wcout << "Benchmarking Algorithm: " << arrayOfRngTypes[k] << endl;
		for (unsigned numCores = 2; numCores <= 4; numCores++)
		{
			std::wcout << "Benchmarking Number of Cores = " << numCores << endl;
			memset((void *)randomDoubleArray_1, 0, NumRandomValues * sizeof(double));		// clear both arrays to clear results from previous iteration
			memset((void *)randomDoubleArray_2, 0, NumRandomValues * sizeof(double));

			returnValue = mklRandomDoubleParallel_SkipAhead(randomDoubleArray_1, NumRandomValues, seed, arrayOfRngTypes[k], numCores);
			returnValue = mklRandomDoubleParallel_SkipAhead(randomDoubleArray_2, NumRandomValues, seed, arrayOfRngTypes[k], 1);

			for (size_t i = 0; i < NumRandomValues; i++)
			{
				if (randomDoubleArray_1[i] != randomDoubleArray_2[i])
				{
					std::wcout << "Random arrays did not compare at index " << i << " : " << randomDoubleArray_1[i] << " " << randomDoubleArray_2[i] << endl;
					break;
				}
				//else
				//	std::wcout << randomFloatArray[i] << " " << randomFloatArray[i] << endl;
			}
		}
	}
	delete[] randomDoubleArray_1;
	delete[] randomDoubleArray_2;

	return 0;
}
