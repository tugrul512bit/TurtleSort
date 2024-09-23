#include"fastest-quicksort-with-index.cuh" 
#include<algorithm>

// test program
int main()
{
    using Type = int;
    constexpr int n = 1024 * 1024;


    // only sorts values (faster)
    Quick::FastestQuicksort<Type> sortVal(n);
    std::vector<Type> sample = { 5,4,3,9,8,7 };
    sortVal.StartSorting(&sample);
    sortVal.Sync();
    for (auto& e : sample)
        std::cout << e << " ";
    std::cout << std::endl;


    // sorts & tracks id values (slower)
    Quick::FastestQuicksort<Type> sort(n);
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