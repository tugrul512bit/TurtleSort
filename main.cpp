#include"fastest-quicksort.cuh"
#include<algorithm>

// test program
int main()
{
    using Type = char;
    constexpr int n = 1024*1024*32;

    // this can sort any length of arrays up to n
    FastestQuicksort<Type> sort(n);
    std::cout << "Check GPU boost frequency if performance drops." << std::endl;

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