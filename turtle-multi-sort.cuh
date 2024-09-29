#include<chrono>
#include<vector>
#include<iostream>
#include"turtle-globals.cuh"


namespace HelperForMultiSorter
{

	// when shared-memory performance is to be tested for multi-sorting
	#define USE_SHARED_MEM 1

#define USE_INTERLEAVED_MEM

	template<int ArrSize, int BlockSize>
	__device__ int ind(const int index, const int threadId)
	{
#ifdef USE_INTERLEAVED_MEM
		return (index * BlockSize) + threadId;
#else
		return index + (ArrSize * threadId);
#endif
	}

	template<typename Type, int ArrSize, int BlockSize>
	__device__ void swap(Type* __restrict__ buf, const int index1, const int index2, const int threadId)
	{
		//printf(" (%i %i %i) ", threadId, buf[convertToInterleavedIndex(index1, threadId)], buf[convertToInterleavedIndex(index2, threadId)]);
		const Type tmp = buf[ind<ArrSize, BlockSize>(index1, threadId)];
		buf[ind<ArrSize, BlockSize>(index1, threadId)] = buf[ind<ArrSize, BlockSize>(index2, threadId)];
		buf[ind<ArrSize, BlockSize>(index2, threadId)] = tmp;
	}
	template<typename Type, int ArrSize, int BlockSize>
	__device__ void insert(Type* __restrict__ buf, const int index, const int threadId)
	{
		const  int parentIndex = ((index + 1) / 2) - 1;
		if (parentIndex >= 0 && parentIndex != index)
		{

			if (buf[ind<ArrSize, BlockSize>(parentIndex, threadId)] < buf[ind<ArrSize, BlockSize>(index, threadId)])
			{
				swap<Type, ArrSize, BlockSize>(buf, parentIndex, index, threadId);
				insert<Type, ArrSize, BlockSize>(buf, parentIndex, threadId);
			}

		}
	}

	template<typename Type, int ArrSize, int BlockSize>
	__device__ void remove(Type* __restrict__ buf, const int index, const int currentSize, const int threadIndex)
	{
		const int leftChildIndex = ((index + 1) * 2) - 1;
		const int rightChildIndex = leftChildIndex + 1;

		if (leftChildIndex < currentSize && rightChildIndex < currentSize)
		{
			if (buf[ind<ArrSize, BlockSize>(leftChildIndex, threadIndex)] > buf[ind<ArrSize, BlockSize>(rightChildIndex, threadIndex)])
			{
				if (buf[ind<ArrSize, BlockSize>(index, threadIndex)] < buf[ind<ArrSize, BlockSize>(leftChildIndex, threadIndex)])
				{
					swap<Type, ArrSize, BlockSize>(buf, index, leftChildIndex, threadIndex);
					remove<Type, ArrSize, BlockSize>(buf, leftChildIndex, currentSize, threadIndex);
				}
			}
			else
			{
				if (buf[ind<ArrSize, BlockSize>(index, threadIndex)] < buf[ind<ArrSize, BlockSize>(rightChildIndex, threadIndex)])
				{
					swap<Type, ArrSize, BlockSize>(buf, index, rightChildIndex, threadIndex);
					remove<Type, ArrSize, BlockSize>(buf, rightChildIndex, currentSize, threadIndex);
				}
			}
		}
		else if (leftChildIndex < currentSize)
		{
			if (buf[ind<ArrSize, BlockSize>(index, threadIndex)] < buf[ind<ArrSize, BlockSize>(leftChildIndex, threadIndex)])
			{
				swap<Type, ArrSize, BlockSize>(buf, index, leftChildIndex, threadIndex);
				remove<Type, ArrSize, BlockSize>(buf, leftChildIndex, currentSize, threadIndex);
			}
		}


	}

	template<typename Type, int ArrSize, int BlockSize>
	__global__ void multiHeapSortWithShared(Type* __restrict__  data, Type* dataInterleaved)
	{

		int tid = threadIdx.x;
		int gid = blockIdx.x;


		__shared__ Type mem[ArrSize * BlockSize];
		__shared__ Type memTmp[ArrSize * BlockSize];

		for (int i = 0; i < ArrSize; i++)
			memTmp[tid + BlockSize * i] = data[gid * BlockSize * ArrSize + tid + BlockSize * i];
		__syncthreads();

		for (int i = 0; i < ArrSize; i++)
			mem[ind<ArrSize, BlockSize>(i, tid)] = memTmp[tid * ArrSize + i];


		__syncthreads();




		// single-thread sort
		// insert elements 1 by 1
		for (int i = 1; i < ArrSize; i++)
		{
			insert<Type, ArrSize, BlockSize>(mem, i, tid);
		}
		// remove elements 1 by 1   
		for (int i = 0; i < ArrSize; i++)
		{
			const int currentSize = ArrSize - i;

			// remove root
			int tmp = mem[ind<ArrSize, BlockSize>(0, tid)];
			mem[ind<ArrSize, BlockSize>(0, tid)] = mem[ind<ArrSize, BlockSize>(currentSize - 1, tid)];
			remove<Type, ArrSize, BlockSize>(mem, 0, currentSize - 1, tid);
			mem[ind<ArrSize, BlockSize>(currentSize - 1, tid)] = tmp;
		}


		__syncthreads();

		for (int i = 0; i < ArrSize; i++)
			memTmp[tid * ArrSize + i] = mem[ind<ArrSize, BlockSize>(i, tid)];

		__syncthreads();
		for (int i = 0; i < ArrSize; i++)
			data[gid * BlockSize * ArrSize + tid + BlockSize * i] = memTmp[tid + BlockSize * i];
		__syncthreads();

	}



	template<typename Type, int ArrSize, int BlockSize>
	__global__ void multiHeapSort(Type* __restrict__  data, Type* dataInterleaved)
	{

		int tid = threadIdx.x;
		int gid = blockIdx.x;


	
		Type* mem = dataInterleaved + (gid * ArrSize * BlockSize);
		

		__syncthreads();




		// single-thread sort
		// insert elements 1 by 1
		for (int i = 1; i < ArrSize; i++)
		{
			insert<Type, ArrSize, BlockSize>(mem, i, tid);
		}
		// remove elements 1 by 1   
		for (int i = 0; i < ArrSize; i++)
		{
			const int currentSize = ArrSize - i;

			// remove root
			int tmp = mem[ind<ArrSize, BlockSize>(0, tid)];
			mem[ind<ArrSize, BlockSize>(0, tid)] = mem[ind<ArrSize, BlockSize>(currentSize - 1, tid)];
			remove<Type, ArrSize, BlockSize>(mem, 0, currentSize - 1, tid);
			mem[ind<ArrSize, BlockSize>(currentSize - 1, tid)] = tmp;
		}


		__syncthreads();

		for (int i = 0; i < ArrSize; i++)
			data[gid * BlockSize * ArrSize + tid * ArrSize + i] = mem[ind<ArrSize, BlockSize>(i, tid)];




	}
}

namespace Turtle
{
	namespace Multi
	{
		template<typename Type, int ArrSize, int BlockSize,bool UseSharedMemory>
		struct MultiSorter
		{
		private:

		public:
			void MultiSort(const int numArrays, Type* hostData, Type* deviceData, Type* deviceDataInterleaved)
			{
				const int nElementsTotal = numArrays * ArrSize;
				TurtleGlobals::gpuErrchk(cudaMemcpy((void*)deviceData, hostData, nElementsTotal * sizeof(Type), cudaMemcpyHostToDevice));
				if(UseSharedMemory)
					HelperForMultiSorter::multiHeapSortWithShared<Type, ArrSize, BlockSize> << < numArrays / BlockSize, BlockSize >> > (deviceData, deviceDataInterleaved);
				else
					HelperForMultiSorter::multiHeapSort<Type, ArrSize, BlockSize> << < numArrays / BlockSize, BlockSize >> > (deviceData, deviceDataInterleaved);
				TurtleGlobals::gpuErrchk(cudaDeviceSynchronize());
				TurtleGlobals::gpuErrchk(cudaMemcpy(hostData, (void*)deviceData, nElementsTotal * sizeof(Type), cudaMemcpyDeviceToHost));
			}
		};
	}
}
#undef USE_INTERLEAVED_MEM