# Fastest Quicksort

Quicksort algorithm boosted with optional paths for different sized data chunks with different sorting algorithms while keeping the "quick" part as main coordinator. Uses CUDA for accelerating parallel algorithms such as reductions, counting and others.


# Features:

- Uses 3 pivots instead of 1. Each pivot has counting-sort type of optimization and filters all duplicates of pivots for a good speedup. Duplicated elements in array make the sorting faster. Pivots are chosen from start, middle and stop indices.
- CUDA acceleration with dynamic parallelism enables asynchronous computation to CPU. CPU can do more work at the same time with full bandwidth of RAM.
- CUDA compression improves performance for redundancy in data.(work in progress)
- Supports different data types such as int, unsigned char, long, short, etc
- Already-sorted arrays are 2x slow compared to random-initialized arrays. Being only 2x slower for 64M suggests that either the constant factor in worst-case O(n^2 x c) is very good or it is not O(n^2) anymore due to said optimizations.

# Performance
- 16x faster than std::qsort, 8x faster than std::sort for 64M random elements (better when not tracking index)


Test system: RTX4070, Ryzen7900, DDR5-6000 dual-channel RAM.


# Requirements

- GPU: Nvidia with dynamic-parallelism + CUDA 12 support.
- Video-memory: 2GB per 32M elements for index-tracked version
- RAM: only value vector and index vector are used as i/o.

Compiler options: 

- generate relocatable device code (-rdc=true)
- C++ compiler's linkage for cuda.lib (this uses CUDA driver api to enable features such as compressible memory)
- compute_89, sm_89 for latest RTX cards
- -lineinfo for debugging
- host: /Ox highest optimization level
- release mode, x64 selected

# GPGPU Related Algorithms and Technologies Used In This Algorithm

- CUDA Run-time API and a bit of driver API
- ```__constant__``` memory
- ```__shared__``` memory
- Dynamic parallelism
- Tail-launched kernel
- Fire-and-forget kernel
- Multi-way Reduction
- Atomic updates
- Warp shuffle
- CUDA Compressible memory
- Templated CUDA kernels
- Divide and conquer
- Sorting network
- Indirect (tail) recursion
- Two-pass algorithm
- Block synchronization
- Asynchronous computing
- Merging (in progress)
- Binary search
- Parallel Rank computation
  
# Sample Code

```C++
#include"fastest-quicksort-with-index.cuh"
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
#include"fastest-quicksort-with-index.cuh" 
#include<algorithm>

// test program
int main()
{
    if (false)
    {

        std::vector<int> A = { 7,8,9 };
        std::vector<int> B = { 2,2,3,3,4,4,100,200 };
        const int lenA = A.size();
        const int lenB = B.size();
        const int lenC = lenA + lenB;
        std::vector<int> C(lenC);
        for (int i = 0; i < lenC; i++)C[i] = -1;
        std::vector<int> scatterC(lenC);

        int numOp = 0;
        for (int i = 0; i < lenB; i++)
        {
            int val = B[i];
            int l = 0;
            int r = lenA - 1;
            int m = (r - l) / 2 + l;
            bool dir = false;

            while (r >= l)
            {

                // if bigger, go right
                if (!(val < A[m]))
                {
                    l = m + 1;
                    dir = true;
                }
                else
                {
                    r = m - 1;
                    dir = false;
                }
                m = (r - l) / 2 + l;

                if (i == 0)
                    std::cout << m << std::endl;

                numOp++;
            }


            C[m + i] = val;

        }


        for (int i = 0; i < lenA; i++)
        {
            int val = A[i];
            int l = 0;
            int r = lenB - 1;
            int m = (r - l) / 2 + l;
            bool dir = false;

            while (r >= l)
            {

                // if bigger, go right
                if (val > B[m])
                {
                    l = m + 1;
                    dir = true;
                }
                else
                {
                    r = m - 1;
                    dir = false;
                }
                m = (r - l) / 2 + l;

                if (i == 0)
                    std::cout << m << std::endl;

                numOp++;
            }


            C[m + i] = val;

        }




        for (int i = 0; i < lenC; i++)
            std::cout << C[i] << " ";
        std::cout << std::endl;
        std::wcout << "op=" << numOp << std::endl;
        return 0;
    }


    using Type = long;
    constexpr int n = 1024*1024*4;


    // n: number of elements supported for sorting
    // compress: (if possible) enables nvidia's compressible memory to possibly increase effective bandwidth/cache capacity
    bool compress = false;

    Quick::FastestQuicksort<Type> sortVal(n, compress);
    std::vector<Type> sample = { 5,4,3,9,8,7 };
    sortVal.StartSorting(&sample);
    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;
    std::cout << "Memory compression successful=" << sortVal.MemoryCompressionSuccessful() << std::endl;
    

    // compression disabled by default
    Quick::FastestQuicksort<Type> sort(n, compress);
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
        }

        size_t t1, t2, t3;
        {
            Quick::Bench bench(&t1);
            sort.StartSorting(&hostData,&hostIndex);
            double t = sort.Sync();

        }

        
        {
            Quick::Bench bench(&t2);
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
            Quick::Bench bench(&t3);
            std::sort(backup2.begin(), backup2.end(), [](auto& e1, auto& e2) { return e1.data < e2.data; });
        }
        std::cout << "gpu: " << t1 / 1000000000.0 << "   std::qsort:" << t2 / 1000000000.0 << "   std::sort:" << t3 / 1000000000.0 << std::endl;
        bool err = false, err2 = false;

  
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


Benchmark output with index-tracking:
```
3 4 5 7 8 9
Memory compression successful=0
Check GPU boost frequency if performance drops.
-------------------------
gpu: 0.014469   std::qsort:0.265007   std::sort:0.150765
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.014144   std::qsort:0.265076   std::sort:0.150385
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0140748   std::qsort:0.26494   std::sort:0.149877
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0146339   std::qsort:0.263999   std::sort:0.1503
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0141821   std::qsort:0.262419   std::sort:0.151541
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0145474   std::qsort:0.264242   std::sort:0.150362
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0143699   std::qsort:0.263078   std::sort:0.150025
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0145359   std::qsort:0.26721   std::sort:0.150186
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0145903   std::qsort:0.262706   std::sort:0.149648
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0144979   std::qsort:0.263792   std::sort:0.149752
quicksort (4194304 elements) completed successfully
```

Compression enabled and index tracking disabled:

```
3 4 5 7 8 9
Memory compression successful=1
Check GPU boost frequency if performance drops.
-------------------------
gpu: 0.0108745   std::qsort:0.267064   std::sort:0.149957
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0112093   std::qsort:0.264321   std::sort:0.150763
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0106026   std::qsort:0.262814   std::sort:0.150587
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0108687   std::qsort:0.261983   std::sort:0.149851
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0106626   std::qsort:0.260848   std::sort:0.151226
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0107278   std::qsort:0.267597   std::sort:0.152919
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0107692   std::qsort:0.261292   std::sort:0.149451
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0107154   std::qsort:0.264865   std::sort:0.149486
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0108813   std::qsort:0.262408   std::sort:0.149665
quicksort (4194304 elements) completed successfully
-------------------------
gpu: 0.0110078   std::qsort:0.263066   std::sort:0.149765
quicksort (4194304 elements) completed successfully
```

15x std::sort performance!!

# CUDA Compressible Memory Test Result

![because processing a sorted array is faster](https://github.com/tugrul512bit/FastestQuicksort/blob/master/qHu9lk%5B1%5D.jpg)
