
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
cudaError_t radixCuda(int* v, const unsigned int size);

typedef unsigned char uchar; 
typedef unsigned int uint;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__global__ void isEvenKernel(const int* v, uint* e, const int size, const int powOfTwo)
{
	int i = threadIdx.x;
	e[i] = !((1 << powOfTwo) & v[i]);

}


/*
1: for d = 1 to log2 n do
2:     for all k in parallel do
3:          if k >= 2^d  then
4:              x[k] = x[k – 2^(d-1)] + x[k]
*/

__global__ void scanKernel(uint* f, const int size, const int round)
{
	int readOffset = (round % 2 == 0) ? size : 0;
	int writeOffset = (readOffset) ? 0 : size;

	int i = threadIdx.x;
	int prevValOffset = 1 << (round - 1);


	f[writeOffset + i] = (i >= prevValOffset) ? f[readOffset + i] + f[readOffset + i - prevValOffset] : f[readOffset + i];
}

__global__ void populateTrueMaskKernel(uint* t, const uint* f, const int totalF)
{
	const int i = threadIdx.x;
	t[i] = i - f[i] + totalF;
}

__global__ void calculateDstKernel(uint* dst, const uint* e, const uint* f, const uint* t)
{
	const int i = threadIdx.x;
	dst[i] = (!e[i]) ? t[i] : f[i];
}

__global__ void mapOutputKernel(int* o, const int* v, const uint* dst)
{
	const int i = threadIdx.x;
	o[dst[i] - 1] = v[i];
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

void printOnCPU(const int* dev_v, const int size, const int lineSize)
{
	int* v = (int*)malloc(size * sizeof(int));
	cudaMemcpy(v, dev_v, size * sizeof(int), cudaMemcpyDeviceToHost);

	printV(v, size, lineSize);

	free(v);
}

void printOnCPU(const uint* dev_v, const int size, const int lineSize)
{
	int* v = (int*)malloc(size * sizeof(uint));
	cudaMemcpy(v, dev_v, size * sizeof(uint), cudaMemcpyDeviceToHost);

	printV(v, size, lineSize);

	free(v);
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
	else if (!(exp & 1))
	{
		return intPow(base * base, exp / 2);
	}
	else
	{
		return base * intPow(base, exp - 1);
	}
}


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

bool isSorted(const int* v, const int size)
{
	for (int i = 1; i < size; ++i)
	{
		if (v[i] < v[i - 1])
			return false;
	}

	return true;
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
	
	const unsigned long vSize = 1024;
	const unsigned long range = vSize * 100;

	using namespace std;

	cout << "Beginning radix sort.\nNum elements = " << vSize << endl;

	/*
	vector<uint32_t> v(vSize);

	string fileName = "output\\RadixTimes.txt";

	ofstream f;
	f.open(fileName);

	//cout << "Initial vector:" << endl;
	//printV(v, 5);
	for (int r = 9; r < 11; ++r)
	{

		for (uint32_t& i : v)
		{
			i = rand() % range;
		}

		cout << "r = " << r << "...";

		clock_t t0 = clock();

		radixSort(v, r);

		clock_t t1 = clock();

		double dt = (double)(t1 - t0) / CLOCKS_PER_SEC;
		double TpE = dt / vSize;


		assert(isSorted(v));
		f << TpE << endl;
		cout << "done." << endl;

		if (r == 16)
		{
			f << "Time for r16 sort: " << dt << endl;
		}
	}
	
	f.close();
	cout << "Sort complete!" << endl;
	*/

	//printV(v, 5);

	//PressEnterToContinue();

	srand(clock());

	int* v = (int*)malloc(vSize * sizeof(int));
	for (int i = 0; i < vSize; ++i)
	{
		v[i] = rand() % range;
	}

	printV(v, vSize, 16);
	cout << "================================================\n================================================" << endl;

	cudaError_t status = radixCuda(v, vSize);
	if (status != cudaSuccess) {
		fprintf(stderr, "radixCuda failed!");
		PressEnterToContinue();
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	
	cout << "================================================\n================================================" << endl;
	printV(v, vSize, 16);


	assert(isSorted(v, vSize));
	cout << "\nSort complete." << endl;
	free(v);

	PressEnterToContinue();

    return 0;
	
}


cudaError_t radixCuda(int* v, const unsigned int size)
{
	int*   dev_v   = 0;
	uint*  dev_e   = 0;
	uint*  dev_f   = 0;
	uint*  dev_t   = 0;
	uint*  dev_dst = 0;
	int*   dev_o   = 0;

	cudaError_t status;

	// get log of v length
	int logOfSize = 0;
	int logSizeTester = size;
	while (logSizeTester >>= 1)
		++logOfSize;

	// set GPU
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed. Check for CUDA-capable GPU.");
		goto Error;
	}

	// allocate GPU buffers
	status = cudaMalloc((void**)&dev_v, size * sizeof(int));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_e, size * sizeof(uint));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	// needs to be twice as long as input for hillis steele scan
	status = cudaMalloc((void**)&dev_f, size * sizeof(uint) * 2);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_t, size * sizeof(uint));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_dst, size * sizeof(uint));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_o, size * sizeof(int));
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed.");
		goto Error;
	}

	status = cudaMemcpy(dev_v, v, size * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (v->dev_v) failed.");
		goto Error;
	}

	//std::cout << "dev_v:   ";
	//printOnCPU(dev_v, size, size);

	// Find log of max number in v
	int max = findMax(v, size);
	uint logOfMax = 0;
	while (max >>= 1)
		++logOfMax;

	// sort every binary digit

	for (uint i = 0; i <= logOfMax; ++i)
	{

		//std::cout << "-----------------------------------------------------------------\n";
		//std::cout << "Round " << i << "\n";

		isEvenKernel<<<1, size >>>(dev_v, dev_e, size, i);
		// Check for any errors launching the kernel
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			fprintf(stderr, "isEvenKernel launch failed: %s\n", cudaGetErrorString(status));
			goto Error;
		}


		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching isEvenKernel!\n", status);
			goto Error;
		}

	/*	std::cout << "dev_e:   ";
		printOnCPU(dev_e, size, size);*/


		// Copy isEven buffer to scan buffer.
		status = cudaMemcpy(dev_f, dev_e, size * sizeof(uint), cudaMemcpyDeviceToDevice);
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (dev_e -> dev_f) failed!");
			goto Error;
		}

		/*std::cout << "dev_f_0: ";
		printOnCPU(dev_f, size * 2, size * 2);*/

		//std::cout << "Scan rounds: " << logOfSize << std::endl;

		// create prefix sum
		for (int j = 1; j <= logOfSize; ++j)
		{
			scanKernel<<< 1, size >>>(dev_f, size, j);

			// Check for any errors launching the kernel
			status = cudaGetLastError();
			if (status != cudaSuccess) {
				fprintf(stderr, "scanKernel launch (round %d.%d) failed: %s\n", i,j, cudaGetErrorString(status));
				goto Error;
			}

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			status = cudaDeviceSynchronize();
			if (status != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scanKernel (round %d.%d).\n", status, i, j);
				goto Error;
			}

			//std::cout << "scan round " << j << std::endl;
			//std::cout << "dev_f_" << j << ": ";
			//printOnCPU(dev_f, size * 2, size * 2);
		}

		// if an odd number of scan rounds, copy the second half to the first half
		if (logOfSize & 1) {
			status = cudaMemcpy(dev_f, dev_f + size, size * sizeof(uint), cudaMemcpyDeviceToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy (dev_f -> dev_f) failed.");
				goto Error;
			}
		}

		//std::cout << "dev_f':  ";
		//printOnCPU(dev_f, size * 2, size * 2);

		// totalF = e[n-1] + f[n-1]
		uint eLast = 0;
		uint fLast = 0;

		status = cudaMemcpy(&eLast, dev_e + size - 1, sizeof(uint), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (dev_e -> eLast) failed.");
			goto Error;
		}

		status = cudaMemcpy(&fLast, dev_f + size - 1, sizeof(uint), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy (dev_f -> fLast) failed.");
			goto Error;
		}

		//std::cout << "eLast: " << eLast << "\nfLast: " << fLast << std::endl;

		const uint totalF = eLast + fLast;

		// populate dev_t
		populateTrueMaskKernel<<<1, size>>>(dev_t, dev_f, totalF);
		// Check for any errors launching the kernel
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			fprintf(stderr, "populateTrueMaskKernel launch (round %d) failed: %s\n", i, cudaGetErrorString(status));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching populateTrueMaskKernel (round %d).\n", status, i);
			goto Error;
		}

		//std::cout << "dev_t:   ";
		//printOnCPU(dev_t, size, size);


		calculateDstKernel<<<1, size>>> (dev_dst, dev_e, dev_f, dev_t);
		// Check for any errors launching the kernel
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			fprintf(stderr, "calculateDstKernel launch (round %d) failed: %s\n", i, cudaGetErrorString(status));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateDstKernel (round %d).\n", status, i);
			goto Error;
		}

		//std::cout << "dev_dst: ";
		//printOnCPU(dev_dst, size, size);

		mapOutputKernel << <1, size >> > (dev_o, dev_v, dev_dst);
		// Check for any errors launching the kernel
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			fprintf(stderr, "mapOutputKernel launch (round %d) failed: %s\n", i, cudaGetErrorString(status));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mapOutputKernel (round %d).\n", status, i);
			goto Error;
		}

		//std::cout << "dev_o:   ";
		//printOnCPU(dev_o, size, size);

		if (i < logOfMax)
		{
			status = cudaMemcpy(dev_v, dev_o, size * sizeof(int), cudaMemcpyDeviceToDevice);
			if (status != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy (dev_o -> dev_v) failed.");
				goto Error;
			}

			//std::cout << "dev_v:   ";
			//printOnCPU(dev_v, size, size);
		}
	}

	// copy back to CPU
	status = cudaMemcpy(v, dev_o, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (dev_o -> v) failed.");
		goto Error;
	}

Error:
	if (dev_v)
		cudaFree(dev_v);
	if (dev_e)
		cudaFree(dev_e);
	if (dev_f)
		cudaFree(dev_f);
	if (dev_t)
		cudaFree(dev_t);
	if (dev_dst)
		cudaFree(dev_dst);
	if (dev_o)
		cudaFree(dev_o);

	return status;
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
