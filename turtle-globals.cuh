#pragma once
#ifndef TURTLE_GLOBALS
#define TURTLE_GLOBALS 1

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include <iostream>
#include <chrono>
namespace TurtleGlobals
{
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}
	inline
	void gpuErrchk(cudaError_t err) 
	{ 
		gpuAssert(err, __FILE__, __LINE__); 
	}


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
}
#endif