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
    FastestQuicksort<Type> sort(n);
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
#include<algorithm>

// test program
int main()
{
    using Type = unsigned int;
    constexpr int n = 1024 * 1024*4;

    // this can sort any length of arrays up to n
    FastestQuicksort<Type> sort(n);


    std::vector<Type> hostData(n),backup(n),backup2(n);
    for (int j = 0; j < 25; j++)
    {
 
        for (int i = 0; i < n; i++)
        {
            hostData[i] = rand()*rand()+rand();
            backup[i] = hostData[i];
            backup2[i] = hostData[i];
        }


        size_t t1, t2, t3;
        {
            Bench bench(&t1);
            sort.StartSorting(&hostData);
            sort.Sync();
        }
        {
            Bench bench(&t2);
            std::qsort
            (
                backup.data(),
                backup.size(),
                sizeof(decltype(backup)::value_type),
                [](const void* x, const void* y)
                {
                    const int arg1 = *static_cast<const Type*>(x);
                    const int arg2 = *static_cast<const Type*>(y);

                    if (arg1 < arg2)
                        return -1;
                    if (arg1 > arg2)
                        return 1;
                    return 0;
                }
            );
        }
        {
            Bench bench(&t3);
            std::sort(backup2.begin(), backup2.end());
        }
        std::cout << "gpu: " << t1 / 1000000000.0 << "  std::qsort:" << t2 / 1000000000.0 << "   std::sort:" << t3 / 1000000000.0 << std::endl;
        bool err = false;
        for (int i = 0; i < n - 2; i++)
            if (hostData[i] > hostData[i + 1])
            {
                std::cout << "error at: " << i << ": " << hostData[i] << " " << hostData[i + 1] << " " << hostData[i + 2] << std::endl;
                err = true;
                j = 1000000;
                break;
            }

        if (!err)
        {
            std::cout << "quicksort (" << n << " elements) completed successfully " << std::endl;
        }
    }

    return 0;
}
```


Benchmark output:
```
gpu: 0.0281749  std::qsort:0.351473   std::sort:0.210874
quicksort (4194304 elements) completed successfully
gpu: 0.0258249  std::qsort:0.349816   std::sort:0.208962
quicksort (4194304 elements) completed successfully
gpu: 0.0278634  std::qsort:0.351745   std::sort:0.206858
quicksort (4194304 elements) completed successfully
gpu: 0.0259602  std::qsort:0.349713   std::sort:0.210283
quicksort (4194304 elements) completed successfully
gpu: 0.0264832  std::qsort:0.350344   std::sort:0.208919
quicksort (4194304 elements) completed successfully
gpu: 0.0296906  std::qsort:0.349768   std::sort:0.210426
quicksort (4194304 elements) completed successfully
gpu: 0.025604  std::qsort:0.352172   std::sort:0.209454
quicksort (4194304 elements) completed successfully
gpu: 0.0286119  std::qsort:0.353272   std::sort:0.210681
quicksort (4194304 elements) completed successfully
gpu: 0.0264158  std::qsort:0.350123   std::sort:0.21063
quicksort (4194304 elements) completed successfully
gpu: 0.0271356  std::qsort:0.349506   std::sort:0.210362
quicksort (4194304 elements) completed successfully
....
after a while, GPU drivers lower the frequency of GPU because sorting does not involve enough computations
....
gpu: 0.0783411  std::qsort:0.354786   std::sort:0.209785
quicksort (4194304 elements) completed successfully
gpu: 0.0955575  std::qsort:0.34841   std::sort:0.210203
quicksort (4194304 elements) completed successfully
gpu: 0.121601  std::qsort:0.348363   std::sort:0.210336
quicksort (4194304 elements) completed successfully
gpu: 0.107721  std::qsort:0.348657   std::sort:0.211444
quicksort (4194304 elements) completed successfully
gpu: 0.105653  std::qsort:0.35383   std::sort:0.210593
^^ this is when GPU is 400 MHz instead of 2600
```
