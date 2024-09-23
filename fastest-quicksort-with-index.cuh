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
#include "helper.cuh"
namespace Quick
{
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
	__global__ void copyTasksBack(const bool trackIdValues, Type* __restrict__ data,int* __restrict__ numTasks,
		int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
		int* __restrict__ idData,Type*__restrict__ arrTmp,int*__restrict__ idArrTmp);



	/*
		"fast" in context of how "free" a CPU core is
		this sorting is asynchronous to CPU
	*/
	template<typename Type, bool TrackIndex=true>
	struct FastestQuicksort
	{
	private:
		int deviceId;
		int compressionSupported;

		Type* data;
		bool dataCompressed;
		bool dataCompressFail;

		int* tasks;
		int* tasks2;
		int* tasks3;
		int* tasks4;
		int* numTasks;
		int maxN;

		int* idData;

		Type* dataTmp;
		int* idDataTmp;


		std::vector<Type>* toSort;
		std::vector<int>* idTracker;
		std::chrono::nanoseconds t1, t2;
		cudaStream_t stream0;

		
	public:
		FastestQuicksort(int maxElements, bool optInCompression=false)
		{
			deviceId = 0;
			maxN = maxElements;
			toSort = nullptr;
			idTracker = nullptr;
			cuInit(0);
			gpuErrchk(cudaSetDevice(0));
			gpuErrchk(cudaDeviceSynchronize());
			CUdevice currentDevice;
			auto cuErr = cuCtxGetDevice(&currentDevice);
			const char* pStr;
			if (cuGetErrorString(cuErr, &pStr) != CUDA_SUCCESS)
			{
				std::cout << "CUDA ERROR: " << pStr << std::endl;
			}
			
			
			cuErr = cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, currentDevice);


			if (cuGetErrorString(cuErr, &pStr) != CUDA_SUCCESS)
			{
				std::cout << "CUDA ERROR: " << pStr << std::endl;
			}

			
			if (MemoryCompressionSupported())
			{
				if (optInCompression)
				{

					if (CUDA_SUCCESS != QuickHelper::allocateCompressible((void**)&data, maxN * sizeof(Type), true))
					{
						dataCompressed = false;
						std::cout << "Compressible memory failed. Trying normal allocation" << std::endl;
						gpuErrchk(cudaMalloc(&data, maxN * sizeof(Type)));
					}
					else
						dataCompressed = true;
					
				}
				else
					dataCompressed = false;
				
			}
			else
				dataCompressed = false;
			
			if (!dataCompressed)
			{
				gpuErrchk(cudaMalloc(&data, maxN * sizeof(Type)));
			}

			
			if (cuGetErrorString(cuErr, &pStr) != CUDA_SUCCESS)
			{
				std::cout<<"CUDA ERROR: " << pStr << std::endl;
			}



			gpuErrchk(cudaStreamCreateWithFlags(&stream0,cudaStreamNonBlocking));
	
			gpuErrchk(cudaMalloc(&dataTmp, maxN * sizeof(Type)));
			gpuErrchk(cudaMalloc(&numTasks, 4 * sizeof(int)));
			gpuErrchk(cudaMalloc(&tasks, maxN * sizeof(int)));
			gpuErrchk(cudaMalloc(&tasks2, maxN * sizeof(int)));
			gpuErrchk(cudaMalloc(&tasks3, maxN * sizeof(int)));
			gpuErrchk(cudaMalloc(&tasks4, maxN * sizeof(int)));

			if (TrackIndex)
			{
				gpuErrchk(cudaMalloc(&idData, maxN * sizeof(int)));
				gpuErrchk(cudaMalloc(&idDataTmp, maxN * sizeof(int)));
			}



			resetTasks << <1 + maxN / 1024, 1024,0,stream0 >> > (tasks, tasks2, tasks3, tasks4, maxN);
			gpuErrchk(cudaGetLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		bool MemoryCompressionSupported()
		{
			return compressionSupported && dataCompressed;
		}

		// starts sorting in GPU, returns immediately
		// arrayToSort: this array is sorted by comparing its element values
		// indicesToTrack: this array's elements follow same path with arrayToSort to be used for sorting objects
		// sizes of two arrays have to be same
		void StartSorting(std::vector<Type>* arrayToSort, std::vector<int>* indicesToTrack=nullptr)
		{
			t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			toSort = arrayToSort;
			idTracker = indicesToTrack;
			int numTasksHost[4] = { 1,0,1,0 };
			int hostTasks[2] = { 0,toSort->size() - 1 };

			if (dataCompressed)
			{
				gpuErrchk(cudaMemcpy((void*)data, toSort->data(), toSort->size() * sizeof(Type), cudaMemcpyHostToDevice));
			}
			else
			{
				gpuErrchk(cudaMemcpy((void*)data, toSort->data(), toSort->size() * sizeof(Type), cudaMemcpyHostToDevice));
			}
			gpuErrchk(cudaMemcpy((void*)numTasks, numTasksHost, 4 * sizeof(int), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy((void*)tasks2, hostTasks, 2 * sizeof(int), cudaMemcpyHostToDevice));

			if(TrackIndex && idTracker !=nullptr)
				gpuErrchk(cudaMemcpy((void*)idData, indicesToTrack->data(), indicesToTrack->size() * sizeof(int), cudaMemcpyHostToDevice));

			if (TrackIndex && idTracker != nullptr)
				copyTasksBack << <1, 1024,0,stream0 >> > (1,data, numTasks, tasks, tasks2, tasks3, tasks4,idData, dataTmp, idDataTmp);
			else
				copyTasksBack << <1, 1024, 0, stream0 >> > (0, data, numTasks, tasks, tasks2, tasks3, tasks4, idData, dataTmp, idDataTmp);

		}

		// waits for sorting to complete
		// returns elapsed time in seconds
		double Sync()
		{
			gpuErrchk(cudaStreamSynchronize(stream0));
			//gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(toSort->data(), (void*)data, toSort->size() * sizeof(Type), cudaMemcpyDeviceToHost));
			if (TrackIndex && idTracker != nullptr)
				gpuErrchk(cudaMemcpy(idTracker->data(), (void*)idData, idTracker->size() * sizeof(int), cudaMemcpyDeviceToHost));
			t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			toSort = nullptr;
			idTracker = nullptr;
			return (t2.count() - t1.count()) / 1000000000.0;
		}

		~FastestQuicksort()
		{
			if(dataCompressed)
			if (CUDA_SUCCESS != QuickHelper::freeCompressible((void *)data, maxN * sizeof(Type), true))
			{
				std::cout << "Compressible memory-free failed. Trying normal deallocation" << std::endl;
				gpuErrchk(cudaFree(data));
			}
			
			gpuErrchk(cudaFree(dataTmp));
			gpuErrchk(cudaFree(tasks));
			gpuErrchk(cudaFree(tasks2));
			gpuErrchk(cudaFree(tasks3));
			gpuErrchk(cudaFree(tasks4));
			gpuErrchk(cudaFree(numTasks));

			gpuErrchk(cudaFree(idData));
			gpuErrchk(cudaFree(idDataTmp));

			gpuErrchk(cudaStreamDestroy(stream0));
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
}