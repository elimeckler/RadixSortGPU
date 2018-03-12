#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <vector>

typedef unsigned char uchar;
typedef unsigned int uint;


/*
* Check for any errors launching the kernel.
*/
#ifndef KERNEL_LAUNCH_ERROR
#define KERNEL_LAUNCH_ERROR(kernelCall, statusVar) do { \
	statusVar = cudaGetLastError(); \
	if (status != cudaSuccess) { \
		fprintf(stderr, #kernelCall" launch failed: %s\n", cudaGetErrorString(status)); \
	goto Error; \
	} \
} while (0);
#endif // KERNEL_LAUNCH_ERROR

/*
* cudaDeviceSynchronize waits for the kernel to finish, and returns
* any errors encountered during the launch.
*/
#ifndef SYNC_KERNEL
#define SYNC_KERNEL(kernelCall, statusVar) do { \
	statusVar = cudaDeviceSynchronize(); \
	if (statusVar != cudaSuccess) { \
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching "#kernelCall"\n", statusVar); \
		goto Error; \
	} \
} while (0); 
#endif // SYNC_KERNEL

/*
* Wraps cudaMemcpy to hide all the egregious error checking.
*/
#ifndef CUDA_MEMCPY
#define CUDA_MEMCPY(dst, src, size, type, statusVar) do { \
	statusVar = cudaMemcpy(dst, src, size, type); \
	if (statusVar != cudaSuccess) { \
		fprintf(stderr, "cudaMemcpy ("#src" -> "#dst") failed."); \
		goto Error; \
	} \
} while(0);
#endif //CUDA_MEMCPY

/*
* Wraps cudaMalloc to hide all the egregious error checking.
*/
#ifndef CUDA_MALLOC
#define CUDA_MALLOC(ptr, size, statusVar) do { \
	statusVar = cudaMalloc((void**)&ptr, size); \
	if (status != cudaSuccess) \
	{ \
		fprintf(stderr, "cudaMalloc ("#ptr") failed."); \
		goto Error; \
	} \
} while (0);
#endif // CUDA_MALLOC

/*
* Wraps thread id lookup. May become useful later.
*/
#ifndef THREAD_ID
#define THREAD_ID (blockDim.x * blockIdx.x + threadIdx.x)
#endif // THREAD_ID


// taken from http://www.cplusplus.com/forum/articles/7312/
void PressEnterToContinue()
{
	int c;
	printf("Press ENTER to exit.");
	fflush(stdout);
	do c = getchar(); while ((c != '\n') && (c != EOF));
}

/**
*  Prints c array to console.
*  @param v vector to print
*  @param size length of \param v
*  @param lineSize maximum number of elements per newline
*/
void printV(const int* v, const long int size, const int lineSize)
{
	int i = 0;
	int lineI = 0;
	while (i < size)
	{
		std::cout << " " << v[i];
		++i;
		lineI = i % lineSize;

		if (lineI == 0)
		{
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

/**
*  Prints int array in GPU memory to console.
*  @param dev_v vector to print
*  @param size length of \param dev_v
*  @param lineSize maximum number of elements per newline
*/
void printOnCPU(const int* dev_v, const int size, const int lineSize)
{
	int* v = (int*)malloc(size * sizeof(int));
	cudaMemcpy(v, dev_v, size * sizeof(int), cudaMemcpyDeviceToHost);

	printV(v, size, lineSize);

	free(v);
}

/**
*  Prints uint array in GPU memory to console.
*  @param dev_v vector to print
*  @param size length of \param dev_v
*  @param lineSize maximum number of elements per newline
*/
void printOnCPU(const uint* dev_v, const int size, const int lineSize)
{
	int* v = (int*)malloc(size * sizeof(uint));
	cudaMemcpy(v, dev_v, size * sizeof(uint), cudaMemcpyDeviceToHost);

	printV(v, size, lineSize);

	free(v);
}

/**
*  Positive-exponent integer exponentiation.
*  @param base number to be raised to power \param exp
*  @param exp non-negative power
*  @return base^exp, or -1 if exp is negative
*/
int intPow(const int base, const int exp)
{
	if (exp < 0)    return -1;
	if (exp == 0)   return 1;
	if (exp == 1)   return base;
	if (!(exp & 1)) return intPow(base * base, exp / 2);
	return base * intPow(base, exp - 1);
}

/**
*  Maximum value in integer array
*  @param v integer array to be searched
*  @param size number of elements in v
*/
int findMax(int* v, uint size)
{
	int max = v[0];

	for (uint i = 0; i < size; ++i)
	{
		if (v[i] > max)
			max = v[i];
	}

	return max;
}

/**
*  Verifies int array is sorted.
*  @param v array to be checked
*  @param size number of elements in v
*  @return every element is greater or equal to the element that comes before it
*/
bool isSorted(const int* v, const int size)
{
	for (int i = 1; i < size; ++i)
	{
		if (v[i] < v[i - 1])
			return false;
	}

	return true;
}

/**
*  Verifies two int arrays contain equivalent values.
*  @param v1 array to be checked
*  @param v2 array to be checked
*  @param size number of elements in v1 and v2
*  @return value stored at any given location in v1 = value stored in same location of v2
*/
bool isEqual(const int* v1, const int* v2, uint size)
{
	for (int i = 0; i < size; ++i)
	{
		if (v1[i] != v2[i]) return false;
	}
	return true;
}

/**
*  Comparison function to be passed to qsort.
*/
int compare(const void* e1, const void* e2)
{
	int f = *((int*)e1);
	int s = *((int*)e2);
	if (f > s) return 1;
	if (f < s) return -1;
	return 0;
}


/**
*  Maximum value in uint std::vector
*  @param v vector to be searched
*  @param size number of elements in v
*/
int findMax(const std::vector<uint32_t>& v)
{
	int max = v[0];

	for (int i : v)
	{
		if (i > max)
			max = i;
	}

	return max;
}

/**
*  Verifies uint std::vector is sorted.
*  @param v vector to be checked
*  @return every element is greater or equal to the element that comes before it
*/
bool isSorted(const std::vector<uint32_t>& v)
{
	int i = 1;
	while (i < v.size())
	{
		if (v[i - 1] > v[i])
			return false;
		++i;
	}

	return true;
}

/**
*  Prints std::vector to console.
*  @param v vector to print
*/
void printV(const std::vector<uint32_t>& v, const int lineSize)
{
	int i = 0;
	int lineI = 0;
	while (i < v.size())
	{
		std::cout << " " << v[i];
		++i;
		lineI = i % lineSize;

		if (lineI == 0)
		{
			std::cout << std::endl;
		}
	}
}

void VectorToBuckets(const std::vector<uint32_t>& v, std::vector<std::vector<uint32_t>>& b, const int base, const int iteration)
{
	for (int item : v)
	{
		int bucketNum = (item / intPow(base, iteration)) % base;
		b[bucketNum].push_back(item);
	}
}

void BucketsToVector(std::vector<uint32_t>& v, std::vector<std::vector<uint32_t>>& b)
{
	int VLoc = 0;
	for (std::vector<uint32_t>& bucket : b)
	{
		for (int item : bucket)
		{
			v[VLoc] = item;
			VLoc += 1;
		}

		bucket.clear();
	}
}

void radixSort(std::vector<uint32_t>& v, const int base)
{
	//allocate buckets based on size of v and base
	std::vector<std::vector<uint32_t>> buckets(base);
	assert(buckets.size() == base);

	int maxVal = findMax(v);

	int i = 0;

	while (intPow(base, i) <= maxVal)
	{
		//std::cout << "iteration " << i << std::endl;
		VectorToBuckets(v, buckets, base, i);
		BucketsToVector(v, buckets);
		++i;
	}
}
