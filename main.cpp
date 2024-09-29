#include"turtle-sort.cuh"

// test program
int main()
{
    using Type = int;

    constexpr int arrSize = 10;
    std::cout << "test" << std::endl;

    // number of cuda threads per block
    constexpr int blockSize = 64;
    int numArraysToSort = 100000 * blockSize; // has to be multiple of blockSize

    int n = arrSize * numArraysToSort;

    bool compress = false;
    Turtle::TurtleSort<Type> sorter(n, compress);
    std::vector<Type> hostData(n);

    for (int k = 0; k < 10; k++)
    {
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
        }


        constexpr bool useSharedMemory = true;
        double seconds = sorter.MultiSort<Type, arrSize, blockSize, useSharedMemory>(numArraysToSort, hostData.data());
        std::cout << "Sorting " << numArraysToSort << " arrays of " << arrSize << " elements took " << seconds << " seconds" << std::endl;
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

    return 0;

}