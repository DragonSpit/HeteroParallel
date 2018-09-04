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
	CudaRngEncapsulation(unsigned long long PrngSeed, size_t numBytesToPreallocate, bool freeCudaMemory) :
		m_gpu_memory(NULL), rngAlgorithm(CURAND_RNG_PSEUDO_PHILOX4_32_10), m_freeCudaMemory(freeCudaMemory)
	{
		printf("CudaRngEncapsulation constructor. Allocating %zd bytes of Cuda GPU memory. Creating RPNGs.\n", numBytesToPreallocate);
		createCudaPrng(    prngGPU, PrngSeed, rngAlgorithm);	// create PRNG on the CUDA device
		createCudaPrngHost(prngCPU, PrngSeed, rngAlgorithm);	// for correctness verification

		if (numBytesToPreallocate != 0)
			m_gpu_memory = allocateCudaMemory(numBytesToPreallocate);
		m_num_bytes = numBytesToPreallocate;
		printf("CudaRngEncapsulation: CUDA allocated memory %p number of bytes %zd\n", m_gpu_memory, m_num_bytes);
	}

	~CudaRngEncapsulation() {
		//printf("CudaRngEncapsulation destructor\n");
		freeCudaPrng(prngGPU);
		freeCudaPrng(prngCPU);
		//printf("Cuda memory pointer = %p\n", m_gpu_memory);
		if (m_freeCudaMemory)
			freeCudaMemory(m_gpu_memory);
	}

	void *m_gpu_memory;
	size_t  m_num_bytes;
	curandRngType_t rngAlgorithm;
	curandGenerator_t prngGPU;
	curandGenerator_t prngCPU;
	bool m_freeCudaMemory;
};

#endif /* !defined(CUDA_SUPPORT_H_) */
