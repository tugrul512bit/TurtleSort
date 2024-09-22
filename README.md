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

Benchmark output:
```
gpu: 0.0082205  std::qsort:0.0857989   std::sort:0.0485793
quicksort (1048576 elements) completed successfully
gpu: 0.0081355  std::qsort:0.0863304   std::sort:0.0484651
quicksort (1048576 elements) completed successfully
gpu: 0.0079001  std::qsort:0.0872137   std::sort:0.0475039
quicksort (1048576 elements) completed successfully
gpu: 0.0078362  std::qsort:0.084857   std::sort:0.047796
quicksort (1048576 elements) completed successfully
gpu: 0.0077608  std::qsort:0.0846576   std::sort:0.0471589
quicksort (1048576 elements) completed successfully
gpu: 0.0076506  std::qsort:0.0851764   std::sort:0.0474846
quicksort (1048576 elements) completed successfully
gpu: 0.0083327  std::qsort:0.0846722   std::sort:0.0471619
quicksort (1048576 elements) completed successfully
```

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
    FastestQuicksort<Type> sort(1024 * 1024);
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
