//
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include "TimerCycleAccurateArray.h"

using namespace af;
static size_t numRandoms = 16 * 1024 * 1024;
array unsortedRandomArrayU32;
array sortedArrayU32;
array indexes;

void generateRandomArray()
{
	// Generate a random unsigned array
	array randomArrayU32 = randu(numRandoms, f32);
	af::sync();
}

// Develop a function that generates several smaller sub-arrays at a time in a loop and transfers the results into a single large array
// or fills in a portion of a larger array at per iteration of the loop
void generateRandomArrayInChunks(int device, size_t numChunks, size_t chunkSize, unsigned long long seed = 2)
{
	TimerCycleAccurateArray	timer;
	af::setDevice(device);
	af::info();

	array randomArrayFloat(numChunks * chunkSize, f32);	// the overall array
	setSeed(seed);

	timer.reset();
	timer.timeStamp();
	// Filling an overall array in chunks seems to run at about 1/2 the speed of filling a single array in one shot, at least at 16M floats per chunk
	for (size_t i = 0; i < numChunks; i++)
	{
		randomArrayFloat( seq(i * chunkSize, i * chunkSize + chunkSize - 1)) = randu(chunkSize, f32);	// fill the overall array in chunks
	}
	af::sync();
	timer.timeStamp();
	std::cout << "Generate array of random floats in chunks: " << (double)(numChunks * chunkSize) / timer.getAverageDeltaInSeconds() << " randoms/sec" << std::endl;

	float * randomArray = new float[numChunks * chunkSize];
	randomArrayFloat.host((void *)randomArray);

	double sum = 0;
	for (unsigned long i = 0; i < numChunks * chunkSize; i++)
		sum += randomArray[i];
	double mean = (double)sum / (numChunks * chunkSize);
	std::cout << "Mean of ArrayFire random array of size " << numChunks * chunkSize << " is " << mean << std::endl;
	delete[] randomArray;
}

void sortArray()
{
	//af_print(randomArrayU32);
	//array sortedArrayU32, indexes;
	sort(sortedArrayU32, indexes, unsortedRandomArrayU32);
	//af_print(sortedArrayU32);
	//af_print(indexes);
}

int ArrayFireTest(int device)
{
	try {
		TimerCycleAccurateArray	timer;
		// Select a device and display arrayfire info
		//int device = argc > 1 ? atoi(argv[1]) : 0;
		af::setDevice(device);
		af::info();

		timer.reset();
		timer.timeStamp();

		//std::cout << "Generate array of random U32: " << (double)numRandoms / timeit(generateRandomArray) << " randoms/sec" << std::endl;
		generateRandomArray();

		timer.timeStamp();
		std::cout << "Generate array of random U32: " << (double)numRandoms / timer.getAverageDeltaInSeconds() << " randoms/sec" << std::endl;

		timer.reset();
		timer.timeStamp();

		// Generate a random unsigned array
		generateRandomArray();
		//array randomArrayU32 = randu(numRandoms, f32);	// was u32, which runs about the same speed
		//af::sync();

		timer.timeStamp();
		std::cout << "Generate array of random U32: " << (double)numRandoms / timer.getAverageDeltaInSeconds() << " randoms/sec" << std::endl;

		float * randomArray = new float [numRandoms];
		//randomArrayU32.host((void *)randomArray);

		double sum = 0;
		for (unsigned long i = 0; i < numRandoms; i++)
			sum += randomArray[i];
		double mean = (double)sum / numRandoms;
		std::cout << "Mean of ArrayFire random array = " << mean << std::endl;
		delete[] randomArray;

		//array sortedArrayU32, indexes;
		//sort(sortedArrayU32, indexes, unsortedRandomArrayU32);

		//std::cout << "Sort array of random U32: "
		//	<< (double)numRandoms / timeit(sortArray) << " randoms/sec" << std::endl;

		return 0;

		printf("Create a 5-by-3 matrix of random floats on the GPU\n");
		array A = randu(5, 3, f32);
		af_print(A);

		printf("Element-wise arithmetic\n");
		array B = sin(A) + 1.5;
		af_print(B);

		printf("Negate the first three elements of second column\n");
		B(seq(0, 2), 1) = B(seq(0, 2), 1) * -1;
		af_print(B);

		printf("Fourier transform the result\n");
		array C = fft(B);
		af_print(C);

		printf("Grab last row\n");
		array c = C.row(end);
		af_print(c);

		printf("Scan Test\n");
		dim4 dims(16, 4, 1, 1);
		array r = constant(2, dims);
		af_print(r);

		printf("Scan\n");
		array S = af::scan(r, 0, AF_BINARY_MUL);
		af_print(S);

		printf("Create 2-by-3 matrix from host data\n");
		float d[] = { 1, 2, 3, 4, 5, 6 };
		array D(2, 3, d, afHost);
		af_print(D);

		printf("Copy last column onto first\n");
		D.col(0) = D.col(end);
		af_print(D);

		// Sort A
		printf("Sort A and print sorted array and corresponding indices\n");
		array vals, inds;
		sort(vals, inds, A);
		af_print(vals);
		af_print(inds);

	}
	catch (af::exception& e) {

		fprintf(stderr, "%s\n", e.what());
		throw;
	}

	return 0;
}

int ArrayFireIntegerExample(int device)
{
	try {
		//int device = argc > 1 ? atoi(argv[1]) : 0;
		af::setDevice(device);
		af::info();
		printf("\n=== ArrayFire signed(s32) / unsigned(u32) Integer Example ===\n");
		int h_A[] = { 1, 2, 4, -1, 2, 0, 4, 2, 3 };
		int h_B[] = { 2, 3, -5, 6, 0, 10, -12, 0, 1 };
		array A = array(3, 3, h_A);
		array B = array(3, 3, h_B);
		printf("--\nSub-refencing and Sub-assignment\n");
		af_print(A);
		af_print(A.col(0));
		af_print(A.row(0));
		A(0) = 11;
		A(1) = 100;
		af_print(A);
		af_print(B);
		A(1, span) = B(2, span);
		af_print(A);
		printf("--Bit-wise operations\n");
		// Returns an array of type s32
		af_print(A & B);
		af_print(A | B);
		af_print(A ^ B);
		printf("\n--Logical operations\n");
		// Returns an array of type b8
		af_print(A && B);
		af_print(A || B);
		printf("\n--Transpose\n");
		af_print(A);
		af_print(A.T());
		printf("\n--Flip Vertically / Horizontally\n");
		af_print(A);
		af_print(flip(A, 0));
		af_print(flip(A, 1));
		printf("\n--Sum along columns\n");
		af_print(A);
		af_print(sum(A));
		printf("\n--Product along columns\n");
		af_print(A);
		af_print(product(A));
		printf("\n--Minimum along columns\n");
		af_print(A);
		af_print(min(A));
		printf("\n--Maximum along columns\n");
		af_print(A);
		af_print(max(A));
		printf("\n--Minimum along columns with index\n");
		af_print(A);
		array out, idx;
		min(out, idx, A);
		af_print(out);
		af_print(idx);
	}
	catch (af::exception& e) {
		fprintf(stderr, "%s\n", e.what());
		throw;
	}
	return 0;
}