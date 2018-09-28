#pragma once

#if !defined(ARRAY_FIRE_SUPPORT_H_)
#define ARRAY_FIRE_SUPPORT_H_

#include <arrayfire.h>

//extern void *allocateCudaMemory(size_t numBytes);
//extern int freeCudaMemory(void *devPtr);
//extern void createCudaPrng(curandGenerator_t& prngGPU, unsigned long long seed, curandRngType_t rngAlgorithm);
//extern void createCudaPrngHost(curandGenerator_t& prngGPU, unsigned long long seed, curandRngType_t rngAlgorithm);
//extern void freeCudaPrng(curandGenerator_t& prngGPU);

class OpenClGpuRngEncapsulation {
public:
	OpenClGpuRngEncapsulation(unsigned long long PrngSeed)
	{
		printf("OpenClGpuRngEncapsulation constructor. Creating RPNGs.\n");
		af::setSeed(PrngSeed);
	}

	~OpenClGpuRngEncapsulation() {
		//printf("OpenClGpuRngEncapsulation destructor\n");
	}
};

class OpenClGpuMemoryEncapsulation {
public:
	OpenClGpuMemoryEncapsulation(size_t numBytesToPreallocate) :
		m_gpu_memory(numBytesToPreallocate, f32), m_num_bytes(numBytesToPreallocate)
	{
	}

	~OpenClGpuMemoryEncapsulation() {
	}

	af::array m_gpu_memory;
	size_t  m_num_bytes;
};

#endif /* !defined(ARRAY_FIRE_SUPPORT_H_) */
