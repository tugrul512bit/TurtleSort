#include"turtle-globals.cuh"
#include"quick-helper.cuh"
#include<string>
namespace TurtleBuffer
{

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
				TurtleGlobals::gpuErrchk(errCu);
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
						TurtleGlobals::gpuErrchk(cudaFree(data));
					}
				}
				else
					TurtleGlobals::gpuErrchk(cudaFree(data));
			}
		}
	};
}