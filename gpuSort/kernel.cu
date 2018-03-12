#include <ctime>
#include "radixSort.h"

/*
 * CudaRadixSort: Runs radix sort on the GPU. Currently supports up to 1024 non-negative integers.
 */


cudaError_t radixCuda(int* v, const unsigned int size);

__global__ void isEvenKernel(const int* v, uint* e, const int size, const int powOfTwo)
{
	int i = THREAD_ID;
	if (i < size)
		e[i] = !((1 << powOfTwo) & v[i]);

}

__global__ void exclusiveScanShiftCopyKernel(uint* dst, const uint* src, const int size)
{
	int i = THREAD_ID;
	if (i < size)
		dst[i] = (i > 0) ? src[i - 1] : 0;
}

__global__ void scanKernel(uint* f, const int size, const int round)
{
	/*
	for d = 1 to log2 n do
	for all k in parallel do
	if k >= 2^d  then
	x[k] = x[k - 2^(d-1)] + x[k]
	*/
	int readOffset = (round % 2 == 0) ? size : 0;
	int writeOffset = (readOffset) ? 0 : size;

	int i = THREAD_ID;
	int prevValOffset = 1 << (round - 1);

	if (i < size)
		f[writeOffset + i] = (i >= prevValOffset) 
		? f[readOffset + i] + f[readOffset + i - prevValOffset] 
		: f[readOffset + i];
}

__global__ void populateTrueMaskKernel(uint* t, const uint* f, const int totalF, const int size)
{
	const int i = THREAD_ID;
	if (i < size)
		t[i] = i - f[i] + totalF;
}

__global__ void calculateDstKernel(uint* dst, const uint* e, const uint* f, const uint* t, const int size)
{
	const int i = THREAD_ID;
	if (i < size)
		dst[i] = (!e[i]) ? t[i] : f[i];
}

__global__ void mapOutputKernel(int* o, const int* v, const uint* dst, const int size)
{
	const int i = THREAD_ID;
	if (i < size)
		o[dst[i]] = v[i];
}


int main()
{
	using namespace std;
	const unsigned long minSize = 32768000;
	const unsigned long maxSize = 32768000;

	cout << "Beginning tests size " << minSize << " to " << maxSize << endl;
	for (unsigned long vSize = minSize; vSize <= maxSize; vSize *= 2)
	{
		const unsigned long range = vSize * 5;

		//cout << vSize << endl;

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


		srand(clock());

		int* v = (int*)malloc(vSize * sizeof(int));
		for (int i = 0; i < vSize; ++i)
		{
			v[i] = rand() % range;
		}

		int* vTest = (int*)malloc(vSize * sizeof(int));

		for (int i = 0; i < vSize; ++i)
		{
			vTest[i] = v[i];
		}

		/*cout << "vTest_0:\n";
		printV(vTest, vSize, 16);*/
		qsort(vTest, vSize, sizeof(int), compare);
		assert(isSorted(vTest, vSize));


		//printV(v, vSize, 16);
		//cout << "================================================\n================================================" << endl;

		clock_t t0 = clock();
		cudaError_t status = radixCuda(v, vSize);
		clock_t t1 = clock();

		if (status != cudaSuccess) {
			fprintf(stderr, "radixCuda failed!\n");
			PressEnterToContinue();
			return 1;
		}

		double dt = (double)(t1 - t0) / CLOCKS_PER_SEC;
		double TpE = dt / vSize;

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		status = cudaDeviceReset();
		if (status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!\n");
			return 1;
		}

		//cout << "================================================\n================================================" << endl;
		//printV(v, vSize, 16);
		//cout << "vTest:\n";
		//printV(vTest, vSize, 16);

		_ASSERT_EXPR(isSorted(v, vSize), "Not sorted at size %d", vSize);
		_ASSERT_EXPR(isEqual(v, vTest, vSize), "Not equivalent to vTest at size %d", vSize);
		//cout << "\nSort complete." << endl;
		free(v);
		free(vTest);
		cout << TpE << endl;

	}
	cout << "Tests from list size " << minSize << " to " << maxSize << " completed.\n";

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
	{
		++logOfSize;
		if (logSizeTester == 1) ++logOfSize;
	}

	// TODO: generalize this
	const uint threadsPerBlock = 1024;
	// for multiple blocks
	const uint numBlocks = size / threadsPerBlock + !!(size % threadsPerBlock);

	// set GPU
	status = cudaSetDevice(0);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed. Check for CUDA-capable GPU.");
		goto Error;
	}

	// allocate GPU buffers
	CUDA_MALLOC(dev_v,   size * sizeof(int),      status);
	CUDA_MALLOC(dev_e,   size * sizeof(uint),     status);
	CUDA_MALLOC(dev_f,   size * sizeof(uint) * 2, status); // needs to be twice as long as input for hillis steele scan
	CUDA_MALLOC(dev_t,   size * sizeof(uint),     status);
	CUDA_MALLOC(dev_dst, size * sizeof(uint),     status);
	CUDA_MALLOC(dev_o,   size * sizeof(int),      status);

	CUDA_MEMCPY(dev_v, v, size * sizeof(int), cudaMemcpyHostToDevice, status);

	/*std::cout << "dev_v:   ";
	printOnCPU(dev_v, size, size);*/

	// Find log of max number in v
	int max = findMax(v, size);
	uint logOfMax = 0;
	while (max >>= 1)
		++logOfMax;

	// sort every binary digit
	for (uint i = 0; i <= logOfMax; ++i)
	{

		/*std::cout << "-----------------------------------------------------------------\n";
		std::cout << "Round " << i << "\n";*/
		isEvenKernel <<< numBlocks, threadsPerBlock >>> (dev_v, dev_e, size, i);
		KERNEL_LAUNCH_ERROR(isEvenKernel, status);
		SYNC_KERNEL(isEvenKernel, status);

		/*std::cout << "dev_e:   ";
		printOnCPU(dev_e, size, size);
*/
		// Copy isEven buffer to scan buffer (shifting for prefix scan).
		exclusiveScanShiftCopyKernel<<< numBlocks, threadsPerBlock >>>(dev_f, dev_e, size);
		KERNEL_LAUNCH_ERROR(exclusiveScanShiftCopyKernel, status);
		SYNC_KERNEL(exclusiveScanShiftCopyKernel, status);

	/*	std::cout << "dev_f_0: ";
		printOnCPU(dev_f, size * 2, size * 2);*/

		/*std::cout << "Scan rounds: " << logOfSize << std::endl;*/

		// create prefix sum
		for (int j = 1; j <= logOfSize; ++j)
		{
			scanKernel<<< numBlocks, threadsPerBlock >>>(dev_f, size, j);
			KERNEL_LAUNCH_ERROR(scanKernel, status);
			SYNC_KERNEL(scanKernel, status);

			/*std::cout << "scan round " << j << std::endl;
			std::cout << "dev_f_" << j << ": ";
			printOnCPU(dev_f, size * 2, size * 2);*/
		}

		// if an odd number of scan rounds, copy the second half to the first half
		if (logOfSize & 1) {
			CUDA_MEMCPY(dev_f, dev_f + size, size * sizeof(uint), cudaMemcpyDeviceToDevice, status);
		}

		/*std::cout << "dev_f':  ";
		printOnCPU(dev_f, size * 2, size * 2);*/

		// totalF = e[n-1] + f[n-1]
		uint eLast = 0;
		uint fLast = 0;
		CUDA_MEMCPY(&eLast, dev_e + size - 1, sizeof(uint), cudaMemcpyDeviceToHost, status);
		CUDA_MEMCPY(&fLast, dev_f + size - 1, sizeof(uint), cudaMemcpyDeviceToHost, status);

		/*std::cout << "eLast: " << eLast << "\nfLast: " << fLast << std::endl;*/

		const uint totalF = eLast + fLast;

		// populate dev_t
		populateTrueMaskKernel<<< numBlocks, threadsPerBlock >>>(dev_t, dev_f, totalF, size);
		KERNEL_LAUNCH_ERROR(populateTrueMaskKernel, status);
		SYNC_KERNEL(populateTrueMaskKernel, status);

		/*std::cout << "dev_t:   ";
		printOnCPU(dev_t, size, size);*/

		calculateDstKernel<<< numBlocks, threadsPerBlock >>> (dev_dst, dev_e, dev_f, dev_t, size);
		KERNEL_LAUNCH_ERROR(calculateDstKernel, status);
		SYNC_KERNEL(calculateDstKernel, status);
		

	/*	std::cout << "dev_dst: ";
		printOnCPU(dev_dst, size, size);*/

		mapOutputKernel<<< numBlocks, threadsPerBlock >>>(dev_o, dev_v, dev_dst, size);
		KERNEL_LAUNCH_ERROR(mapOutputKernel, status);
		SYNC_KERNEL(mapOutputKernel, status);

		/*std::cout << "dev_o:   ";
		printOnCPU(dev_o, size, size);*/

		if (i < logOfMax)
		{
			CUDA_MEMCPY(dev_v, dev_o, size * sizeof(int), cudaMemcpyDeviceToDevice, status);
			/*std::cout << "dev_v:   ";
			printOnCPU(dev_v, size, size);*/
		}
	}

	// copy back to CPU
	CUDA_MEMCPY(v, dev_o, size * sizeof(int), cudaMemcpyDeviceToHost, status);

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
