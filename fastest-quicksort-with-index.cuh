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
#include "quick-buffer.cuh"
#include <memory>
#include <algorithm>
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


	template<typename Type>
		__global__ void mergeSortedChunks(const bool trackIdValues, int* __restrict__ tasks, Type* __restrict__ arr, 
			Type* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

		template<typename Type>
		__global__ void copyMergedChunkBack(const bool trackIdValues, int* __restrict__ tasks, Type* __restrict__ arr,
			Type* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);
		

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

		std::shared_ptr<QuickBuffer::Buffer<Type>> data;
		std::shared_ptr < QuickBuffer::Buffer<Type>> dataTmp;

		std::shared_ptr < QuickBuffer::Buffer<int>> tasks;
		std::shared_ptr < QuickBuffer::Buffer<int>> tasks2;
		std::shared_ptr < QuickBuffer::Buffer<int>> tasks3;
		std::shared_ptr < QuickBuffer::Buffer<int>> tasks4;

		std::shared_ptr < QuickBuffer::Buffer<int>> numTasks;
		std::shared_ptr < QuickBuffer::Buffer<int>> idData;
		std::shared_ptr < QuickBuffer::Buffer<int>> idDataTmp;
		
		int maxN;


		std::vector<Type>* toSort;
		std::vector<int>* idTracker;
		std::chrono::nanoseconds t1, t2;
		cudaStream_t stream0;

		std::vector<int> hostTasks;
		bool merge;
		int nChunks;
		int chunkSize; // not same for the last chunk
	public:
		FastestQuicksort(int maxElements, bool optInCompression=false)
		{
			chunkSize = 1024 * 64;
			nChunks = 0;
			merge = false;
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
			
			data = std::make_shared<QuickBuffer::Buffer<Type>>("data",maxN, currentDevice, compressionSupported && optInCompression);
			dataTmp = std::make_shared<QuickBuffer::Buffer<Type>>("dataTmp", maxN, currentDevice, compressionSupported && optInCompression);
			tasks = std::make_shared<QuickBuffer::Buffer<int>>("tasks", maxN, currentDevice,  compressionSupported && optInCompression);
			tasks2 = std::make_shared<QuickBuffer::Buffer<int>>("tasks2", maxN, currentDevice,  compressionSupported && optInCompression);
			tasks3 = std::make_shared<QuickBuffer::Buffer<int>>("tasks3", maxN, currentDevice, compressionSupported && optInCompression);
			tasks4 = std::make_shared<QuickBuffer::Buffer<int>>("tasks4", maxN, currentDevice, compressionSupported && optInCompression);
			numTasks = std::make_shared<QuickBuffer::Buffer<int>>("numTasks", 4, currentDevice, false);
		
			if (TrackIndex)
			{
				idData = std::make_shared<QuickBuffer::Buffer<int>>("idData", maxN, currentDevice,  compressionSupported && optInCompression);
				idDataTmp = std::make_shared<QuickBuffer::Buffer<int>>("idDataTmp", maxN, currentDevice, compressionSupported && optInCompression);
			}

			gpuErrchk(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));

			

			resetTasks << <1 + maxN / 1024, 1024,0,stream0 >> > (tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), maxN);
			
			
			gpuErrchk(cudaGetLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}

		bool MemoryCompressionSuccessful()
		{
			return data->CompressionEnabled();
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

			nChunks = 8;
			chunkSize = toSort->size() / 8;
			int curOfs = 0;
			int numTasksHost[4] = { 1,0,1,0 };

			hostTasks.clear();
			const int sz = toSort->size();
			if (chunkSize>1)
			{
				merge = true;
				while (curOfs < sz)
				{
					if (curOfs < sz-1)
						hostTasks.push_back(curOfs);
					else
						break;


					if (curOfs + chunkSize - 1 < sz)
						hostTasks.push_back(curOfs + chunkSize - 1);
					else
						hostTasks.push_back(sz - 1);
						
					const int curChunkSize = 1 + hostTasks[hostTasks.size() - 1] - hostTasks[hostTasks.size() - 2];

					//nChunks++;
					curOfs += curChunkSize;
				}
				numTasksHost[0] = nChunks;
				numTasksHost[1] = 0;
				numTasksHost[2] = nChunks;
				numTasksHost[3] = 0;
			}
			else
			{
				numTasksHost[0] = 1;
				numTasksHost[1] = 0;
				numTasksHost[2] = 1;
				numTasksHost[3] = 0;
				hostTasks.push_back(0);
				hostTasks.push_back(sz - 1);
				merge = false;
			}
			
	
			if (hostTasks.size() % 2 != 0)
			{
				std::cout << "ERROR: task scheduling error!" << std::endl;

				return;
			}
			gpuErrchk(cudaMemcpy((void*)data->Data(), toSort->data(), toSort->size() * sizeof(Type), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMemcpy((void*)numTasks->Data(), numTasksHost, 4 * sizeof(int), cudaMemcpyHostToDevice));
			
			gpuErrchk(cudaMemcpy((void*)tasks2->Data(), hostTasks.data(), hostTasks.size()* 2 * sizeof(int), cudaMemcpyHostToDevice));
			
			if(TrackIndex && idTracker !=nullptr)
				gpuErrchk(cudaMemcpy((void*)idData->Data(), indicesToTrack->data(), indicesToTrack->size() * sizeof(int), cudaMemcpyHostToDevice));
			
			if (TrackIndex && idTracker != nullptr)
				copyTasksBack << <1, 1024,0,stream0 >> > (1,data->Data(), numTasks->Data(), tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), idData->Data(), dataTmp->Data(), idDataTmp->Data());
			else
				copyTasksBack << <1, 1024, 0, stream0 >> > (0, data->Data(), numTasks->Data(), tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), idData->Data(), dataTmp->Data(), idDataTmp->Data());
			gpuErrchk(cudaGetLastError());
		}

		// waits for sorting to complete
		// returns elapsed time in seconds
		double Sync()
		{
			gpuErrchk(cudaStreamSynchronize(stream0));
			/*
			   o o o o o o o o
			   0 1 2 3

			   i= 0
			   0 1 2 3 ==> 0 3
			   i= 1
			   4 5 6 7 ==> 4 7


			   i=0
			   0 3 4 7 ==> 0 7

			   i=0
			  
			*/
			std::vector<std::vector<int>> mrg = { 
				// 8 chunks --> 4 chunks
				{ 0,1,2,3,chunkSize },{4,5,6,7,chunkSize},{8,9,10,11,chunkSize},{12,13,14,15,chunkSize},

				// 4 chunks --> 2 chunks
				{0,3,4,7,chunkSize*2},{8,11,12,15,chunkSize * 2},

				// 2 chunks --> 1 chunk
				{0,7,8,15,chunkSize * 4}			
			};

			/* merge start */
			if (merge)
			{
				auto mergeTasks = hostTasks;
				
					for(auto mr:mrg)
					{

						mergeTasks[0] = hostTasks[mr[0]];
						mergeTasks[1] = hostTasks[mr[1]];
						mergeTasks[2] = hostTasks[mr[2]];
						mergeTasks[3] = hostTasks[mr[3]];

						gpuErrchk(cudaMemcpy((void*)tasks2->Data(), (mergeTasks.data()), 4 * sizeof(int), cudaMemcpyHostToDevice));
						mergeSortedChunks << <1 + (mr[4] / 1024), 1024 >> > (TrackIndex, tasks2->Data(), data->Data(), dataTmp->Data(), idData->Data(), idDataTmp->Data());
						copyMergedChunkBack << <1 + ((mr[4] * 2) / 1024), 1024 >> > (TrackIndex, tasks2->Data(), data->Data(), dataTmp->Data(), idData->Data(), idDataTmp->Data());
						gpuErrchk(cudaStreamSynchronize(stream0));
					}
				


			}
			/* merge stop */


			gpuErrchk(cudaMemcpy(toSort->data(), (void*)data->Data(), toSort->size() * sizeof(Type), cudaMemcpyDeviceToHost));

			if (TrackIndex && idTracker != nullptr)
				gpuErrchk(cudaMemcpy(idTracker->data(), (void*)idData->Data(), idTracker->size() * sizeof(int), cudaMemcpyDeviceToHost));

		

			t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			toSort = nullptr;
			idTracker = nullptr;
			return (t2.count() - t1.count()) / 1000000000.0;
		}

		~FastestQuicksort()
		{
			
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