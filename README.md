# Fastest Quicksort

This is only fastest compared to CPU-based quicksort versions and runs asynchronously to CPU so the host-side can do work while sorting is made.

# Features:

- Uses 3 pivots instead of 1. Each pivot has counting-sort type of optimization and filters all duplicates of pivots for a good speedup. Duplicated elements in array make the sorting faster. Pivots are chosen from start, middle and stop indices.
- CUDA acceleration with dynamic parallelism enables asynchronous computation to CPU. CPU can do more work at the same time with full bandwidth of RAM.
- Supports different data types such as int, unsigned char, long, short, etc
- Already-sorted arrays are 2x slow compared to random-initialized arrays. Being only 2x slower for 64M suggests that either the constant factor in worst-case O(n^2 x c) is very good or it is not O(n^2) anymore due to said optimizations.

# Performance
- 10x faster than std::qsort for 64M random elements (loses some performance with index-tracking)
- 5x faster than std::sort for 64M random elements
- 9x faster than std::sort for 1M random * random + random elements
- 2x faster than std::sort for 4M elements of 0,1,2,...99 repeated (data[i]=i%100)

Test system: RTX4070, Ryzen7900, DDR5-6000 dual-channel RAM.


# Requirements

- GPU: Nvidia with dynamic-parallelism + CUDA 12 support. 
- Video-memory: currently too much but is being optimized (2GB per 32M elements for index-tracked version, higher for simple version for now)
- RAM: only input vector is used as i/o.

Compiler options: 

- generate relocatable device code (-rdc=true)
- compute_89, sm_89 for latest RTX cards
- -lineinfo for debugging
- host: /Ox highest optimization level
- release mode, x64 selected

# Headers

fastest-quicksort.cuh: In ```Quick``` namespace, ```FastestQuicksort``` struct can be initialized with a maximum size to sort. Then any sized array can be sorted by ```StartSorting``` method and synchronized with ```Sync``` method.

fastest-quicksort-with-index.cuh: In ```QuickIndex``` namespace, same method takes a second vector for index values. Indices follow same data path as the data. But it works as SOA rather than AOS (unlike std::sort or std::qsort).
  
# Sample Code

```C++
#include"fastest-quicksort.cuh"
#include<vector>

// test program
int main()
{
    using Type = unsigned long;
    constexpr int n = 1024 * 1024;

    // this can sort any length of arrays up to n
    Quick::FastestQuicksort<Type> sort(n);
    std::vector<Type> test = { 5,3,7,3,1 };
    sort.StartSorting(&test);
    std::cout << "Asynchronous computing..." << std::endl;
    sort.Sync();
    for (auto e : test)
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
#include"fastest-quicksort.cuh" 
#include"fastest-quicksort-with-index.cuh" 
#include<algorithm>

// test program
int main()
{
    using Type = int;
    constexpr int n = 1024*1024;


    // only sorts values (faster)
    Quick::FastestQuicksort<Type> sortVal(n);
    std::vector<Type> sample = { 5,4,3,9,8,7 };
    sortVal.StartSorting(&sample);
    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;


    // sorts & tracks id values (slower)
    QuickIndex::FastestQuicksort<Type> sort(n);
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
    std::vector<StdData> backup(n), backup2(n);
    for (int j = 0; j < 25; j++)
    {
        std::cout << "-------------------------" << std::endl;
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand()*rand()+rand();
            hostIndex[i] = hostData[i];
            backup[i].data = hostData[i];
            backup2[i].data = hostData[i];
            backup[i].index = hostData[i];
            backup2[i].index = hostData[i];
        }

        size_t t1, t2, t3;
        {
            QuickIndex::Bench bench(&t1);           
            sort.StartSorting(&hostData,&hostIndex);
            double t = sort.Sync();

        }

        {
            QuickIndex::Bench bench(&t2);
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
            QuickIndex::Bench bench(&t3);
            std::sort(backup2.begin(), backup2.end(), [](auto& e1, auto& e2) { return e1.data < e2.data; });
        }
        std::cout<<"gpu: "<<t1/1000000000.0 << "   std::qsort:" << t2 / 1000000000.0 << "   std::sort:" << t3 / 1000000000.0 << std::endl;
        bool err = false, err2=false;
        for (int i = 0; i < n - 2; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i] << " " << hostData[i + 1] << " " << hostData[i + 2] << std::endl;
                err = true;
                j = 1000000;
                break;
            }

        for (int i = 0; i < n ; i++)
        {
            if (hostData[i] != hostIndex[i])
            {
                err2 = true;
                j = 1000000;
                std::cout << "error: index calculation wrong" << std::endl;
                break;
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


Benchmark output:
```
3 4 5 7 8 9
Check GPU boost frequency if performance drops.
-------------------------
gpu: 0.005535   std::qsort:0.08706   std::sort:0.0489966
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0051853   std::qsort:0.092936   std::sort:0.048418
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0050253   std::qsort:0.0941636   std::sort:0.0498431
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0053617   std::qsort:0.0856727   std::sort:0.0486482
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0049196   std::qsort:0.0856767   std::sort:0.0484373
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0047965   std::qsort:0.0858625   std::sort:0.0485127
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0051842   std::qsort:0.0854649   std::sort:0.0481385
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0049616   std::qsort:0.085061   std::sort:0.0484908
quicksort (1048576 elements) completed successfully
-------------------------
gpu: 0.0054292   std::qsort:0.08536   std::sort:0.0481445
quicksort (1048576 elements) completed successfully
```
