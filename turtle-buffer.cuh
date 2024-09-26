#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <device_functions.h>
#include"quick-helper.cuh"
#include<string>
namespace TurtleBuffer
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
	template<typename Type>
	struct Buffer
	{
	private:

		bool compressed;
		bool allocated;
		Type* data;
		int n;
		std::string name;
	public:
		Buffer()
		{

			compressed = false;
			data = nullptr;
			n = 0;
			name = "unnamed";
			allocated = false;
		}
		Buffer(std::string nameString, int numElements, CUdevice currentDevice, bool optInCompression)
		{
			
			//cudaSetDevice(currentDevice);
			name = nameString;
			data = nullptr;
			n = numElements;
			int compSup;
			allocated = false;




			if (optInCompression)
			{

				if (CUDA_SUCCESS != QuickHelper::allocateCompressible((void**)&data, n * sizeof(Type), true))
				{
					compressed = false;
					std::cout <<name<< " buffer has CUDA ERROR: compressible memory failed. Trying normal allocation" << std::endl;
				}
				else
				{
					compressed = true;
					allocated = true;
				
				}
			}
			else
			{
				compressed = false;
			}



			if (!compressed)
			{
				
				auto errCu = cudaMalloc(&data, n * sizeof(Type));
				gpuErrchk(errCu);
				allocated = true;
			}

		}

		Type* Data() 
		{
			return data;
		}

		bool CompressionEnabled()
		{
			return compressed;
		}

		~Buffer()
		{
			
			if (allocated && data != nullptr && n > 0 && name!="unnamed")
			{
		
				if (compressed)
				{
					if (CUDA_SUCCESS != QuickHelper::freeCompressible((void*)data, n * sizeof(Type), true))
					{
						std::cout<< name << " buffer has CUDA ERROR: compressible memory-free failed. Trying normal deallocation" << std::endl;
						gpuErrchk(cudaFree(data));
					}
				}
				else
					gpuErrchk(cudaFree(data));
			}
		}
	};
}