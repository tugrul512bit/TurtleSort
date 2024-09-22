# Fastest Quicksort

This is only fastest compared to CPU-based quicksort versions and runs asynchronously to CPU so the host-side can do work while sorting is made.

# Features:

- Uses 3 pivots instead of 1. Each pivot has counting-sort type of optimization and filters all duplicates of pivots for a good speedup. Duplicated elements in array make the sorting faster. Pivots are chosen from start, middle and stop indices.
- CUDA acceleration with dynamic parallelism enables asynchronous computation to CPU. CPU can do more work at the same time with full bandwidth of RAM.
- Supports different data types such as int, unsigned char, long, short, etc
- Already-sorted arrays are 2x slow compared to random-initialized arrays. Being only 2x slower for 64M suggests that either the constant factor in worst-case O(n^2 x c) is very good or it is not O(n^2) anymore due to said optimizations.

# Performance
- 10x faster than std::qsort for 64M random elements
- 5x faster than std::sort for 64M random elements

Test system: RTX4070, Ryzen7900, DDR5-6000 dual-channel RAM.


# Requirements

Nvidia GPU with dynamic-parallelism + CUDA 12 support. 

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


    // only sorts values
    Quick::FastestQuicksort<Type> sortVal(n);
    std::vector<Type> sample = { 5,4,3,9,8,7 };
    sortVal.StartSorting(&sample);
    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;


    // sorts & tracks id values
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
            hostData[i] = rand();
            hostIndex[i] = hostData[i];
            backup[i].data = hostData[i];
            backup2[i].data = hostData[i];
            backup[i].index = hostData[i];
            backup2[i].index = hostData[i];
        }

        size_t t1, t2, t3;
        {
            sort.StartSorting(&hostData,&hostIndex);
            double t = sort.Sync();            
            std::cout << "gpu-sort elapsed time = " << t << std::endl;
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
        std::cout << "std::qsort:" << t2 / 1000000000.0 << "   std::sort:" << t3 / 1000000000.0 << std::endl;
        bool err = false, err2=false;
        for (int i = 0; i < n - 2; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i] << " " << hostData[i + 1] << " " << hostData[i + 2] << std::endl;
                err = true;
                j = 1000000;
                break;
            }

        for (int i = 0; i < n; i++)
        {
            if (hostData[i] != hostIndex[i])
            {
                std::cout << "error: index tracking failed!!! at: "<<i<<": "<<hostIndex[i]<< std::endl;
                err2 = true;
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
-------------------------
gpu-sort elapsed time = 0.0124775
std::qsort:0.0674101   std::sort:0.0379516
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0129493
std::qsort:0.0689581   std::sort:0.0427465
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0136129
std::qsort:0.0684936   std::sort:0.0380023
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0126742
std::qsort:0.0689375   std::sort:0.0377938
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0121185
std::qsort:0.0683966   std::sort:0.0376106
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0127307
std::qsort:0.0670079   std::sort:0.040169
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0126643
std::qsort:0.0679908   std::sort:0.0375375
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0124303
std::qsort:0.0676993   std::sort:0.0380867
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0123888
std::qsort:0.0681775   std::sort:0.0378545
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0124053
std::qsort:0.0682802   std::sort:0.0377692
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0126854
std::qsort:0.067635   std::sort:0.0375062
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0121689
std::qsort:0.0667947   std::sort:0.0379612
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0124026
std::qsort:0.0672219   std::sort:0.0374425
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0125424
std::qsort:0.0682399   std::sort:0.0371546
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0122665
std::qsort:0.0682838   std::sort:0.0376951
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0128245
std::qsort:0.0677027   std::sort:0.037129
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.012449
std::qsort:0.068749   std::sort:0.0379262
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0123495
std::qsort:0.0675071   std::sort:0.0376137
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0127134
std::qsort:0.0674832   std::sort:0.0380389
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0121387
std::qsort:0.068116   std::sort:0.0373989
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0125361
std::qsort:0.0671925   std::sort:0.0376637
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0126504
std::qsort:0.0675255   std::sort:0.0374682
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0124315
std::qsort:0.0685524   std::sort:0.0374617
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0123337
std::qsort:0.0684826   std::sort:0.0375642
quicksort (1048576 elements) completed successfully
-------------------------
gpu-sort elapsed time = 0.0140213
std::qsort:0.0684202   std::sort:0.0376981
quicksort (1048576 elements) completed successfully
```
