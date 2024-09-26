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
- Up to 25x faster than std::qsort
- Up to 15x faster than std::sort
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


15x std::sort performance, 2.3x against 24-thread std::sort(std::execution::par_unseq)!

# CUDA Compressible Memory Test Result

![because processing a sorted array is faster](https://github.com/tugrul512bit/FastestQuicksort/blob/master/qHu9lk%5B1%5D.jpg)
