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


    using Type = int;
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

        if(false)
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