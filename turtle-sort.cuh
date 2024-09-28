#include"turtle-buffer.cuh"
#include"turtle-multi-sort.cuh"
#include<chrono>
#include<vector>
#include<iostream>
#include <memory>
#include <algorithm>
namespace Turtle
{


	__global__ void resetTasks(int* tasks, int* tasks2, int* tasks3, int* tasks4, const int n);
	 
	template<typename Type>
	__global__ void copyTasksBack(const bool trackIdValues, Type* __restrict__ data,int* __restrict__ numTasks,
		int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
		int* __restrict__ idData,Type*__restrict__ arrTmp,int*__restrict__ idArrTmp);


	template<typename Type>
		__global__ void mergeSortedChunks(const bool trackIdValues, int* __restrict__ tasks, Type* __restrict__ arr, 
			Type* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

		template<typename Type>
		__global__ void copyMergedChunkBack(const bool trackIdValues, const int n, Type* __restrict__ arr,
			Type* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);
		
		
		void resetTasksHost(cudaStream_t& stream0, int* tasks, int* tasks2, int* tasks3, int* tasks4, const int n);
	
	template<typename Type, bool TrackIndex=true>
	struct TurtleSort
	{
	private:
		int deviceId;
		int compressionSupported;
		
		std::shared_ptr<TurtleBuffer::Buffer<Type>> data;
		std::shared_ptr < TurtleBuffer::Buffer<Type>> dataTmp;

		std::shared_ptr < TurtleBuffer::Buffer<int>> tasks;
		std::shared_ptr < TurtleBuffer::Buffer<int>> tasks2;
		std::shared_ptr < TurtleBuffer::Buffer<int>> tasks3;
		std::shared_ptr < TurtleBuffer::Buffer<int>> tasks4;

		std::shared_ptr < TurtleBuffer::Buffer<int>> numTasks;
		std::shared_ptr < TurtleBuffer::Buffer<int>> idData;
		std::shared_ptr < TurtleBuffer::Buffer<int>> idDataTmp;
		
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
		TurtleSort(int maxElements, bool optInCompression=false)
		{
			chunkSize = 1024 * 64;
			nChunks = 0;
			merge = false;
			deviceId = 0;
			maxN = maxElements;
			toSort = nullptr;
			idTracker = nullptr;
			cuInit(0);
			TurtleGlobals::gpuErrchk(cudaSetDevice(0));
			TurtleGlobals::gpuErrchk(cudaDeviceSynchronize());
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
			
			data = std::make_shared<TurtleBuffer::Buffer<Type>>("data",maxN, currentDevice, compressionSupported && optInCompression);
			dataTmp = std::make_shared<TurtleBuffer::Buffer<Type>>("dataTmp", maxN, currentDevice, compressionSupported && optInCompression);
			tasks = std::make_shared<TurtleBuffer::Buffer<int>>("tasks", maxN, currentDevice,  compressionSupported && optInCompression);
			tasks2 = std::make_shared<TurtleBuffer::Buffer<int>>("tasks2", maxN, currentDevice,  compressionSupported && optInCompression);
			tasks3 = std::make_shared<TurtleBuffer::Buffer<int>>("tasks3", maxN, currentDevice, compressionSupported && optInCompression);
			tasks4 = std::make_shared<TurtleBuffer::Buffer<int>>("tasks4", maxN, currentDevice, compressionSupported && optInCompression);
			numTasks = std::make_shared<TurtleBuffer::Buffer<int>>("numTasks", 4, currentDevice, false);
		
			if (TrackIndex)
			{
				idData = std::make_shared<TurtleBuffer::Buffer<int>>("idData", maxN, currentDevice,  compressionSupported && optInCompression);
				idDataTmp = std::make_shared<TurtleBuffer::Buffer<int>>("idDataTmp", maxN, currentDevice, compressionSupported && optInCompression);
			}

			TurtleGlobals::gpuErrchk(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));

			
			resetTasksHost(stream0, tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), maxN);
			//resetTasks <<<1 + maxN / 1024, 1024,0,stream0 >>> (tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), maxN);
			
			
			TurtleGlobals::gpuErrchk(cudaGetLastError());
			TurtleGlobals::gpuErrchk(cudaDeviceSynchronize());
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



			int curOfs = 0;
			int numTasksHost[4] = { 1,0,1,0 };

			hostTasks.clear();
			const int sz = toSort->size();
			chunkSize = sz / nChunks; // only for making  number of threads exceed number of items
			if (chunkSize>1024)
			{
				merge = true;
			
				hostTasks.push_back(0);
				for (int i = 0; i < nChunks-1; i++)
				{
					hostTasks.push_back(i* chunkSize + chunkSize -1);
					hostTasks.push_back(i * chunkSize + chunkSize);
				}
				hostTasks.push_back(sz-1); // task borders are inclusive

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
			
	
			
			TurtleGlobals::gpuErrchk(cudaMemcpy((void*)data->Data(), toSort->data(), toSort->size() * sizeof(Type), cudaMemcpyHostToDevice));
			
			TurtleGlobals::gpuErrchk(cudaMemcpy((void*)numTasks->Data(), numTasksHost, 4 * sizeof(int), cudaMemcpyHostToDevice));
			
			TurtleGlobals::gpuErrchk(cudaMemcpy((void*)tasks2->Data(), hostTasks.data(), hostTasks.size()* 2 * sizeof(int), cudaMemcpyHostToDevice));
			
			if(TrackIndex && idTracker !=nullptr)
				TurtleGlobals::gpuErrchk(cudaMemcpy((void*)idData->Data(), indicesToTrack->data(), indicesToTrack->size() * sizeof(int), cudaMemcpyHostToDevice));
			
			if (TrackIndex && idTracker != nullptr)
				copyTasksBack << <1, 1024,0,stream0 >> > (1,data->Data(), numTasks->Data(), tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), idData->Data(), dataTmp->Data(), idDataTmp->Data());
			else
				copyTasksBack << <1, 1024, 0, stream0 >> > (0, data->Data(), numTasks->Data(), tasks->Data(), tasks2->Data(), tasks3->Data(), tasks4->Data(), idData->Data(), dataTmp->Data(), idDataTmp->Data());
			TurtleGlobals::gpuErrchk(cudaGetLastError());
		}

		// waits for sorting to complete
		// returns elapsed time in seconds
		double Sync()
		{
			TurtleGlobals::gpuErrchk(cudaStreamSynchronize(stream0));

			std::vector<std::vector<int>> mrg = { 				
				
				{0,1,2,3,chunkSize,1},{4,5,6,7,chunkSize ,1},{8,9,10,11,chunkSize ,1},{12,13,14,15,chunkSize ,1},

				// 4 chunks --> 2 chunks
				{0,3,4,7,chunkSize*2,0},{8,11,12,15,chunkSize*2,0},

				// 2 chunks --> 1 chunk
				{0,7,8,15,chunkSize*4,1},
			};

			/* merge start */
			if (merge)
			{
				auto mergeTasks = hostTasks;
				int ctr = 0;
				bool lastDir = true;
					for(auto mr:mrg)
					{
						mergeTasks[0] = hostTasks[mr[0]];
						mergeTasks[1] = hostTasks[mr[1]];
						mergeTasks[2] = hostTasks[mr[2]];
						mergeTasks[3] = hostTasks[mr[3]];
						bool dir = mr[5];
						if ((lastDir != dir))
						{
							if(!dir)
								copyMergedChunkBack <<<1 + (toSort->size() / 1024), 1024 >>> (TrackIndex, toSort->size(), data->Data(), dataTmp->Data(), idData->Data(), idDataTmp->Data());
							else
								copyMergedChunkBack << <1 + (toSort->size() / 1024), 1024 >> > (TrackIndex, toSort->size(), dataTmp->Data(), data->Data(), idDataTmp->Data(), idData->Data());
						}
						TurtleGlobals::gpuErrchk(cudaMemcpy((void*)tasks2->Data(), (mergeTasks.data()), 4 * sizeof(int), cudaMemcpyHostToDevice));
						if (dir)
						{
							mergeSortedChunks << <1 + ((mr[4]*1.1) / 1024), 1024 >> > (TrackIndex, tasks2->Data(), data->Data(), dataTmp->Data(), idData->Data(), idDataTmp->Data());							
						}
						else
						{
							mergeSortedChunks << <1 + ((mr[4] * 1.1) / 1024), 1024 >> > (TrackIndex, tasks2->Data(), dataTmp->Data(), data->Data(), idDataTmp->Data(), idData->Data());
						}

						
						TurtleGlobals::gpuErrchk(cudaStreamSynchronize(stream0));
						lastDir = dir;
					}
				
					if (lastDir)
					{
						copyMergedChunkBack << <1 + (toSort->size() / 1024), 1024 >> > (TrackIndex, toSort->size(), data->Data(), dataTmp->Data(), idData->Data(), idDataTmp->Data());
						TurtleGlobals::gpuErrchk(cudaStreamSynchronize(stream0));
					}
			}
			/* merge stop */


			TurtleGlobals::gpuErrchk(cudaMemcpy(toSort->data(), (void*)data->Data(), toSort->size() * sizeof(Type), cudaMemcpyDeviceToHost));

			if (TrackIndex && idTracker != nullptr)
				TurtleGlobals::gpuErrchk(cudaMemcpy(idTracker->data(), (void*)idData->Data(), idTracker->size() * sizeof(int), cudaMemcpyDeviceToHost));

		

			t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
			toSort = nullptr;
			idTracker = nullptr;
			return (t2.count() - t1.count()) / 1000000000.0;
		}

		/* type: element type of arrays
		 arr size: size of each array to sort
		 blocksize: number of cuda threads per block (blocksize=32 means they sort 32 arrays together, fast, minimal divergence)
		 returns elapsed time in seconds
		*/ 
		template<typename Type, int ArrSize, int BlockSize=32>
		double MultiSort(const int numArraysToSort, Type* hostDataToSort)
		{
			size_t t0;
			{					
				TurtleGlobals::Bench bench(&t0);
				
				Multi::MultiSorter<Type, ArrSize, BlockSize> sorter;
				sorter.MultiSort(numArraysToSort, hostDataToSort, data->Data(), dataTmp->Data());
			}
			return t0 / 1000000000.0;
		}

		~TurtleSort()
		{
			
			TurtleGlobals::gpuErrchk(cudaStreamDestroy(stream0));
		}
	};



}