
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <ctime>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


// taken from http://www.cplusplus.com/forum/articles/7312/
void PressEnterToContinue()
{
	int c;
	printf("Press ENTER to exit.");
	fflush(stdout);
	do c = getchar(); while ((c != '\n') && (c != EOF));
}


/**
 *	Prints vector to console.
 *  @param v vector to print
*/
void printV(const std::vector<int>& v, const int lineSize)
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

// No fear of recursion--Deepest we'll go is ~10 calls due to use case
int intPow(const int base, const int exp)
{
	if (exp == 0)
	{
		return 1;
	}
	else if (exp == 1)
	{
		return base;
	}
	else if (exp % 2 == 0)
	{
		return intPow(base * base, exp / 2);
	}
	else
	{
		return base * intPow(base, exp - 1);
	}
}


int findMax(const std::vector<int>& v)
{
	int max = v[0];

	for (int i : v)
	{
		if (i > max)
			max = i;
	}

	return max;
}


bool isSorted(const std::vector<int>& v)
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

void VectorToBuckets(const std::vector<int>& v, std::vector<std::vector<int>>& b, const int base, const int iteration)
{
	for (int item : v)
	{
		int bucketNum = (item / intPow(base, iteration)) % base;
		b[bucketNum].push_back(item);
	}
}

void BucketsToVector(std::vector<int>& v, std::vector<std::vector<int>>& b)
{
	int VLoc = 0;
	for (std::vector<int>& bucket : b)
	{
		for (int item : bucket)
		{
			v[VLoc] = item;
			VLoc += 1;
		}

		bucket.clear();
	}
}


void radixSort(std::vector<int>& v, const int base)
{
	//allocate buckets based on size of v and base
	std::vector<std::vector<int>> buckets(base);
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


int main()
{
	/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	*/
	
	const unsigned long vSize = 256000000;
	const unsigned long range = vSize * 10;

	using namespace std;

	cout << "Beginning radix sort.\nNum elements: " << vSize << endl;

	vector<int> v(vSize);

	string fileName = "output\\RadixTimes_processorCPU_size" + to_string(vSize) + "_range" + to_string(range) + ".txt";

	ofstream f;
	f.open(fileName);

	//cout << "Initial vector:" << endl;
	//printV(v, 5);
	for (int r = 2; r < 17; ++r)
	{

		for (int& i : v)
		{
			i = rand() % range;
		}

		clock_t t0 = clock();

		radixSort(v, r);

		clock_t t1 = clock();

		double dt = (double)(t1 - t0) / CLOCKS_PER_SEC;
		double TpE = dt / vSize;


		assert(isSorted(v));
		f << TpE << endl;

		if (r == 16)
		{
			f << "Time for r16 sort: " << dt << endl;
		}
	}
	
	f.close();
	cout << "Sort complete!" << endl;

	//printV(v, 5);

	//PressEnterToContinue();

    return 0;
	
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
