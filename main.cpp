#include"fastest-quicksort-with-index.cuh" 
#include <execution>
#include<algorithm>

// test program
int main()
{


    using Type = int;
    const int n = 12000000;


    // n: number of elements supported for sorting
    // compress: (if possible) enables nvidia's compressible memory to possibly increase effective bandwidth/cache capacity
    bool compress = false;

    Quick::FastestQuicksort<Type> sortVal(n, compress);
    std::vector<Type> sample = { 5,4,3,9,8,1 };
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
            Quick::Bench bench(&t1);
            // works as struct of arrays rather than array of structs and expects id as a different array
            sort.StartSorting(&hostData /*, &hostIndex when an object's id needs to be tracked */);

            /* cpu does async work here */

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
        {
            Quick::Bench bench(&t4);
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


        if (!err && !err2)
        {
            std::cout << "quicksort (" << n << " elements) completed successfully " << std::endl;
        }


    }
    return 0;
}