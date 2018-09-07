#pragma once

#if !defined(CUDA_SUPPORT_H_)
#define CUDA_SUPPORT_H_

#include <curand.h>

extern void *allocateCudaMemory(size_t numBytes);
extern int freeCudaMemory(void *devPtr);
extern void createCudaPrng(curandGenerator_t& prngGPU, unsigned long long seed, curandRngType_t rngAlgorithm);
extern void createCudaPrngHost(curandGenerator_t& prngGPU, unsigned long long seed, curandRngType_t rngAlgorithm);
extern void freeCudaPrng(curandGenerator_t& prngGPU);

class CudaRngEncapsulation {
public:
	CudaRngEncapsulation(unsigned long long PrngSeed) :
		rngAlgorithm(CURAND_RNG_PSEUDO_PHILOX4_32_10)
	{
		printf("CudaRngEncapsulation constructor. Creating RPNGs.\n");
		createCudaPrng(    prngGPU, PrngSeed, rngAlgorithm);	// create PRNG on the CUDA device
		createCudaPrngHost(prngCPU, PrngSeed, rngAlgorithm);	// for correctness verification
	}

	~CudaRngEncapsulation() {
		//printf("CudaRngEncapsulation destructor\n");
		freeCudaPrng(prngGPU);
		freeCudaPrng(prngCPU);
	}

	curandRngType_t rngAlgorithm;
	curandGenerator_t prngGPU;
	curandGenerator_t prngCPU;
};

class CudaMemoryEncapsulation {
public:
	CudaMemoryEncapsulation(size_t numBytesToPreallocate) :
		m_gpu_memory(NULL)
	{
		printf("CudaMemoryEncapsulation constructor. Allocating %zd bytes of Cuda GPU memory.\n", numBytesToPreallocate);
		if (numBytesToPreallocate != 0)
			m_gpu_memory = allocateCudaMemory(numBytesToPreallocate);
		m_num_bytes = numBytesToPreallocate;
		printf("CudaRngEncapsulation: CUDA allocated memory %p number of bytes %zd\n", m_gpu_memory, m_num_bytes);
	}

	~CudaMemoryEncapsulation() {
		//printf("CudaMemoryEncapsulation destructor\n");
		//printf("Cuda memory pointer = %p\n", m_gpu_memory);
		freeCudaMemory(m_gpu_memory);
	}

	void *m_gpu_memory;
	size_t  m_num_bytes;
};

#endif /* !defined(CUDA_SUPPORT_H_) */
