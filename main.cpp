#include"turtle-sort.cuh"

// test program
int main()
{
    using Type = double;

    constexpr int arrSize = 49;
    std::cout << "test" << std::endl;

    // number of cuda threads per block
    constexpr int blockSize = 32;
    int numArraysToSort = 100000 * blockSize; // has to be multiple of blockSize

    int n = arrSize * numArraysToSort;
    Type* data;
    Type* dataInterleaved;
    bool compress = true;
    Turtle::TurtleSort<double> sorter(n, compress);
    std::vector<Type> hostData(n);

    for (int k = 0; k < 10; k++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }



        sorter.MultiSort<double,arrSize>(numArraysToSort, hostData.data());

        for (int i = 0; i < numArraysToSort; i++)
        {
            for (int j = 0; j < arrSize - 1; j++)
            {
                if (hostData[i * arrSize + j] > hostData[i * arrSize + j + 1])
                {
                    std::cout << "sort failed" << std::endl;
                    std::cout << "array-id:" << i << std::endl;
                    std::cout << "element-id:" << j << std::endl;
                    for (int k = 0; k < arrSize; k++)
                        std::cout << hostData[i * arrSize + k] << " ";
                    std::cout << std::endl;
                    return 0;
                }
            }
        }



        std::cout << "sort success" << std::endl;
    }
    if (CUDA_SUCCESS != cudaFree(data))
    {
        std::cout << "failed memory free" << std::endl;
        return 0;
    }
    if (CUDA_SUCCESS != cudaFree(dataInterleaved))
    {
        std::cout << "failed memory free" << std::endl;
        return 0;
    }
    return 0;

}