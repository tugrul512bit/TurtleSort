# TurtleSort

Quicksort algorithm boosted with optional paths for different sized data chunks with different sorting algorithms while keeping the "quick" part as main coordinator. Uses CUDA for accelerating parallel algorithms such as reductions, counting and others.

![How it works](https://github.com/tugrul512bit/FastestQuicksort/blob/master/quicksort.drawio%20(1).png)


# Features:

- Uses 3 pivots instead of 1. Each pivot has counting-sort type of optimization and filters all duplicates of pivots for a good speedup. Duplicated elements in array make the sorting faster. Pivots are chosen from start, middle and stop indices.
- CUDA acceleration with dynamic parallelism enables asynchronous computation to CPU. CPU can do more work at the same time with full bandwidth of RAM.
- CUDA compression improves performance for redundancy in data.(work in progress)
- Supports different data types such as int, unsigned char, long, short, etc
- Already-sorted arrays are 2x slow compared to random-initialized arrays. Being only 2x slower for 64M suggests that either the constant factor in worst-case O(n^2 x c) is very good or it is not O(n^2) anymore due to said optimizations.

# Performance
- Up to 25x faster than single-thread std::qsort
- Up to 15x faster than single-thread std::sort
- Up to 3x faster than 24-thread std::sort(par_unseq,a,b)
- Runs faster when array elements have duplicates (but std::sort runs even faster)
- Same speed with sorted array (but std::sort runs faster so theres only 2x speedup against std::sort)


Test system: RTX4070, Ryzen7900, DDR5-6000 dual-channel RAM.


# Requirements

- GPU: Nvidia with dynamic-parallelism + CUDA 12 support.
- Video-memory: 2GB per 32M elements for index-tracked version
- RAM: only value vector and index vector are used as i/o.
- C++17 supporting CUDA compiler

Compiler options: 

- generate relocatable device code (-rdc=true)
- C++ compiler's linkage for cuda.lib (this uses CUDA driver api to enable features such as compressible memory)
- compute_89, sm_89 for latest RTX cards
- -lineinfo for debugging
- host: /Ox highest optimization level
- release mode, x64 selected

  
# Sample Code

```C++
#include"turtle-sort.cuh" 

// test program
int main()
{
    using Type = int;

    // maximum array length to sort
    const int n = 1000;
    Turtle::TurtleSort<Type> sortVal(n);
    std::vector<Type> sample = { 5,4,3,9,8,1 };
    sortVal.StartSorting(&sample);


    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;

    return 0;
}
```

output:
```
Asynchronous computing...
1 3 3 5 7
```

# Sample Benchmark Code

```C++
#include"turtle-sort.cuh" 
#include <execution>
#include<algorithm>

// test program
int main()
{


    using Type = int;
    const int n = 12*1024*1024;


    // n: number of elements supported for sorting
    // compress: (if possible) enables nvidia's compressible memory to possibly increase effective bandwidth/cache capacity
    bool compress = true;

    Turtle::TurtleSort<Type> sortVal(n, compress);
    std::vector<Type> sample = { 5,4,3,9,8,1 };
    sortVal.StartSorting(&sample);
    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;
    std::cout << "Memory compression successful=" << sortVal.MemoryCompressionSuccessful() << std::endl;


    // compression disabled by default
    Turtle::TurtleSort<Type> sort(n, compress);
    std::cout << "Check GPU boost frequency if performance drops." << std::endl;

    // sample
    std::vector<Type> hostData(n);
    std::vector<int> hostIndex(n);

    // sample for std::sort, std::qsort
    struct StdData
    {
        Type data;
        int index;
    };
    std::vector<StdData> backup(n), backup2(n), backup3(n);
    for (int j = 0; j < 10; j++)
    {
        std::cout << "-------------------------" << std::endl;
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand();
            hostIndex[i] = hostData[i];
            backup[i].data = hostData[i];
            backup2[i].data = hostData[i];
            backup[i].index = hostData[i];
            backup2[i].index = hostData[i];
            backup3[i].data = hostData[i];
            backup3[i].index = hostData[i];
        }

        size_t t1, t2, t3, t4;
        {
            Turtle::Bench bench(&t1);
            sort.StartSorting(&hostData, &hostIndex);
            double t = sort.Sync();

        }


        {
            Turtle::Bench bench(&t2);
            std::qsort
            (
                backup.data(),
                backup.size(),
                sizeof(decltype(backup)::value_type),
                [](const void* x, const void* y)
                {
                    const int arg1 = static_cast<const StdData*>(x)->data;
                    const int arg2 = static_cast<const StdData*>(y)->data;

                    if (arg1 < arg2)
                        return -1;
                    if (arg1 > arg2)
                        return 1;
                    return 0;
                }
            );
        }

        {
            Turtle::Bench bench(&t3);
            std::sort(backup2.begin(), backup2.end(), [](auto& e1, auto& e2) { return e1.data < e2.data; });
        }
        {
            Turtle::Bench bench(&t4);
            std::sort(std::execution::par_unseq, backup3.begin(), backup3.end(), [](auto& e1, auto& e2) { return e1.data < e2.data; });
        }
        std::cout << "gpu: " << t1 / 1000000000.0 <<
            "   std::qsort:" << t2 / 1000000000.0 <<
            "   std::sort:" << t3 / 1000000000.0 <<
            "   std::sort(par_unseq):" << t4 / 1000000000.0 <<
            std::endl;
        bool err = false, err2 = false;


        for (int i = 0; i < n - 1; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i] << " " << hostData[i + 1] << " " << hostData[i + 2] << std::endl;
                err = true;
                j = 1000000;
                return 1;
            }


        for (int i = 0; i < n; i++)
        {
            if (hostData[i] != hostIndex[i])
            {
                err2 = true;
                j = 1000000;
                std::cout << "error: index calculation wrong" << std::endl;
                return 1;
            }
        }
        if (!err && !err2)
        {
            std::cout << "quicksort (" << n << " elements) completed successfully " << std::endl;
        }


    }
    return 0;
}
```

output on ryzen7900 and rtx4090:

```
gpu: 0.036844   std::qsort:0.775426   std::sort:0.430385   std::sort(par_unseq):0.0875657
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0334487   std::qsort:0.766197   std::sort:0.429026   std::sort(par_unseq):0.0793185
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0331019   std::qsort:0.777763   std::sort:0.430848   std::sort(par_unseq):0.0780624
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0329223   std::qsort:0.77231   std::sort:0.424476   std::sort(par_unseq):0.0859999
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0323176   std::qsort:0.765149   std::sort:0.431423   std::sort(par_unseq):0.0798207
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.148611 <---- GPU boost is disabled by driver due to staying idle (waiting for other sorters)
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.162047   <---- GPU boost is disabled by driver due to staying idle (waiting for other sorters)
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0333234   std::qsort:0.770051   std::sort:0.431438   std::sort(par_unseq):0.0776
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.151563    <---- GPU boost is disabled by driver due to staying idle (waiting for other sorters)
quicksort (12582912 elements) completed successfully
-------------------------
gpu: 0.0327161   std::qsort:0.771654   std::sort:0.43577   std::sort(par_unseq):0.0918065
quicksort (12582912 elements) completed successfully
```

15x std::sort performance, 2.3x against 24-thread std::sort(std::execution::par_unseq)!

# CUDA Compressible Memory Test Result

![because processing a sorted array is faster](https://github.com/tugrul512bit/FastestQuicksort/blob/master/qHu9lk%5B1%5D.jpg)

---

# Multiple Sorting Support

```C++
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



        double seconds = sorter.MultiSort<Type,arrSize, blockSize>(numArraysToSort, hostData.data());
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
```

output (copying arrays takes 90% of the total time, its actually 10x faster on gpu-side):

```
Sorting 6400000 arrays of 10 elements took 0.0263472 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0255799 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0255251 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.025334 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0256296 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0256111 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0257346 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0256958 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.0256691 seconds
sort success
Sorting 6400000 arrays of 10 elements took 0.025927 seconds
sort success
```
