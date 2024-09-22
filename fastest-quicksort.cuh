#include<chrono>
#include<vector>
#include<iostream>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void resetTasks(int* tasks, int* tasks2, int* tasks3, int* tasks4, const int n);

template<typename Type>
__global__ void copyTasksBack(Type* __restrict__ data, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
	int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
	Type* __restrict__ leftLeft, Type* __restrict__ rightRight);



/* 
	"fast" in context of how "free" a CPU core is
	this sorting is asynchronous to CPU
*/
template<typename Type>
struct FastestQuicksort
{

	Type* data;
	Type* leftLeft;
	Type* left;
	Type* right;
	Type* rightRight;
	int* tasks;
	int* tasks2;
	int* tasks3;
	int* tasks4;
	int *numTasks;
	int maxN;
	std::vector<Type>* toSort;
	std::chrono::nanoseconds t1, t2;
	FastestQuicksort(int maxElements)
	{
		maxN = maxElements;

		gpuErrchk(cudaSetDevice(0));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMalloc(&data, maxN * sizeof(Type)));
		gpuErrchk(cudaMalloc(&left, maxN * sizeof(Type)));
		gpuErrchk(cudaMalloc(&leftLeft, maxN * sizeof(Type)));
		gpuErrchk(cudaMalloc(&right, maxN * sizeof(Type)));
		gpuErrchk(cudaMalloc(&rightRight, maxN * sizeof(Type)));
		gpuErrchk(cudaMalloc(&numTasks, 4 * sizeof(int)));
		gpuErrchk(cudaMalloc(&tasks, maxN * sizeof(int)));
		gpuErrchk(cudaMalloc(&tasks2, maxN * sizeof(int)));
		gpuErrchk(cudaMalloc(&tasks3, maxN * sizeof(int)));
		gpuErrchk(cudaMalloc(&tasks4, maxN * sizeof(int)));
		resetTasks << <1 + maxN / 1024, 1024 >> > (tasks, tasks2, tasks3, tasks4, maxN);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	// starts sorting in GPU, returns immediately
	// arrayToSort needs to be in scope until Sync() is called
	void StartSorting(std::vector<Type>* arrayToSort)
	{
		t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		toSort = arrayToSort;
		int numTasksHost[4] = { 1,0,1,0 };
		int hostTasks[2] = { 0,toSort->size() - 1 };

		gpuErrchk(cudaMemcpy((void*)data, toSort->data(), toSort->size() * sizeof(Type), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy((void*)numTasks, numTasksHost, 4 * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy((void*)tasks2, hostTasks, 2 * sizeof(int), cudaMemcpyHostToDevice));
		copyTasksBack << <1, 1024 >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);

	}

	// waits for sorting to complete
	// returns elapsed time in seconds
	double Sync()
	{
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(toSort->data(), (void*)data, toSort->size() * sizeof(Type), cudaMemcpyDeviceToHost));
		t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		
		return (t2.count() - t1.count()) / 1000000000.0;
	}

	~FastestQuicksort()
	{
		gpuErrchk(cudaFree(data));
		gpuErrchk(cudaFree(left));
		gpuErrchk(cudaFree(right));
		gpuErrchk(cudaFree(leftLeft));
		gpuErrchk(cudaFree(rightRight));
		gpuErrchk(cudaFree(tasks));
		gpuErrchk(cudaFree(tasks2));
		gpuErrchk(cudaFree(tasks3));
		gpuErrchk(cudaFree(tasks4));
		gpuErrchk(cudaFree(numTasks));
	}
};



class Bench
{
public:
	Bench(size_t* targetPtr)
	{
		target = targetPtr;
		t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		t2 = t1;
	}

	~Bench()
	{
		t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
		if (target)
		{
			*target = t2.count() - t1.count();
		}
		else
		{
			std::cout << (t2.count() - t1.count()) / 1000000000.0 << " seconds" << std::endl;
		}
	}
private:
	size_t* target;
	std::chrono::nanoseconds t1, t2;
};