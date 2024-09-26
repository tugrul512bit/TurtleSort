
#include"turtle-sort.cuh"

namespace Turtle
{
    // maximum number of elements going into brute-force chunks
    constexpr int BRUTE_FORCE_LIMIT = 128;

    // maximum number of elements directly computed without any task mechanics
    // minimum allowed: 32 due to some warp-level computations
    // maximum allowed: BRUTE_FORCE_LIMIT - 1
    constexpr int DIRECTLY_COMPUTE_LIMIT = 32;


    // task pattern: 
    //              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
    //              start stop  start stop  start stop  start stop  ---> tasks buffer
    //              block 0     block 1     block 2     block 3     ---> cuda blocks
    template<typename Type>
    __global__ void quickSortWithoutStreamCompaction(
        const bool trackIdValues,
        Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template<typename Type>
    __global__ void bruteSort(
        const bool trackIdValues,
        Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,        
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template<typename Type>
    __global__ void resetNumTasks(
        const bool trackIdValues,
        Type* __restrict__ data, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp)
    {
        int numTask1 = 0;
        int numTask2 = 0;
        if (threadIdx.x == 0)
        {
            numTask1 = numTasks[0];
            numTasks[2] = numTask1;
            numTask2 = numTasks[1];
            numTasks[3] = numTask2;
            numTasks[0] = 0;
            numTasks[1] = 0;

            __syncthreads();

            if (numTasks[3] > 0)
                bruteSort << <numTasks[3], BRUTE_FORCE_LIMIT, 0, cudaStreamFireAndForget >> > (trackIdValues, data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);


            if (numTasks[2] > 0)
                quickSortWithoutStreamCompaction << <numTasks[2], 1024, 0, cudaStreamTailLaunch >> > (trackIdValues, data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);

        }
    }


    template<typename Type>
    __global__ void copyTasksBack(
        const bool trackIdValues, Type* __restrict__ data, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp)
    {
        const int id = threadIdx.x;
        const int n = numTasks[0];
        const int n2 = numTasks[1];
        const int steps = 1 + n / 1024;
        const int steps2 = 1 + n2 / 1024;


        // make quick-sort tasks usable
        for (int i = 0; i < steps; i++)
        {
            const int curId = id + i * 1024;
            if (curId < n)
            {
                tasks[curId * 2] = tasks2[curId * 2];
                tasks[curId * 2 + 1] = tasks2[curId * 2 + 1];
            }
        }


        // make brute-force tasks usable
        for (int i = 0; i < steps2; i++)
        {
            const int curId = id + i * 1024;
            if (curId < n2)
            {
                tasks3[curId * 2] = tasks4[curId * 2];
                tasks3[curId * 2 + 1] = tasks4[curId * 2 + 1];
            }
        }

        if (id == 0)
        {

            resetNumTasks << <1, 1, 0, cudaStreamTailLaunch >> > (
                trackIdValues, data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
        }

    }



#define compareSwap(a,x,y) if(a[y]<a[x]){auto t = a[x];a[x]=a[y];a[y]=t;}
#define compSw(a,x,y,b) if(a[y]<a[x]){ auto t = a[x];a[x]=a[y];a[y]=t; auto u = b[x];b[x]=b[y];b[y]=u;}

    template<typename Type>
    __global__ void bruteSort(
        const bool trackIdValues,
        Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idArr,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp)
    {
       
        const int id = threadIdx.x;
        const int gid = blockIdx.x;

        __shared__ int taskIdCacheStart;
        __shared__ int taskIdCacheStop;
        if (id == 0)
        {
            taskIdCacheStart = tasks3[gid * 2];
            taskIdCacheStop = tasks3[gid * 2 + 1];
            tasks3[gid * 2] = 0;
            tasks3[gid * 2 + 1] = 0;
        }
        __syncthreads();
        const int startIncluded = taskIdCacheStart;
        const int stopIncluded = taskIdCacheStop;
        const int num = stopIncluded - startIncluded + 1;
        if (startIncluded == 0 && stopIncluded == 0)
        {
            if (id == 0)
                printf("\n brute-force task id error: %i \n", gid);
            return;
        }

        __shared__ Type cache[BRUTE_FORCE_LIMIT];
        __shared__ int idTracker[BRUTE_FORCE_LIMIT];
        if (startIncluded + id <= stopIncluded)
        {
            cache[id] = arr[startIncluded + id];
            if(trackIdValues)
                idTracker[id] = idArr[startIncluded + id];
        }
        __syncthreads();


        for (int i = 0; i < num; i++)
        {
            if (id + 1 < num)
            {
                if ((id % 2 == 0))
                {
                    compSw(cache, id, id + 1, idTracker);
                }
            }
            __syncthreads();
            if (id + 1 < num)
            {
                if ((id % 2 == 1))
                {
                    compSw(cache, id, id + 1, idTracker);
                }
            }
            __syncthreads();
        }


        if (startIncluded + id <= stopIncluded)
        {
            arr[startIncluded + id] = cache[id];
            if (trackIdValues)
                idArr[startIncluded + id] = idTracker[id];
        }
    }

    // call this log2n times while doubling size on each call
    // merge 2 chunks into single array
    // single block or even single 
    template<typename Type>
    __global__ void mergeSortedChunks(
        const bool trackIdValues,
        int* __restrict__ tasks,
        Type* __restrict__ arr, Type* __restrict__ arrTmp,
        int* __restrict__ idArr, int* __restrict__ idArrTmp
    )
    {
        const int id = threadIdx.x + blockIdx.x * blockDim.x;

      
        // inclusive values
        const int startChunk1 = tasks[0];
        const int stopChunk1 = tasks[1];
        const int startChunk2 = tasks[2];
        const int stopChunk2 = tasks[3];
      
        const int sizeChunk1 = stopChunk1 - startChunk1 + 1;
        const int sizeChunk2 = stopChunk2 - startChunk2 + 1;



        if(id< sizeChunk2)
        {
            Type val = arr[startChunk2+id];
            int idTracked = 0;
            if (trackIdValues)
                idTracked = arr[startChunk2 + id];
            int l = startChunk1;
            int r = stopChunk1;
            int m = (r - l) / 2 + l;
            bool dir = false;

            while (r >= l)
            {

                // if bigger, go right
                if (!(val < arr[m]))
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
            }


            arrTmp[m + id] = val;
            if(trackIdValues)
                idArrTmp[m + id] = idTracked;
        }


        if(id<sizeChunk1)
        {
            Type val = arr[startChunk1 + id];
            int idTracked = 0;
            if (trackIdValues)
                idTracked = arr[startChunk1 + id];
            int l = startChunk2;
            int r = stopChunk2;
            int m = (r - l) / 2 + l;
            bool dir = false;

            while (r >= l)
            {

                // if bigger, go right
                if (val > arr[m])
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

            }


            arrTmp[m + id - startChunk2 + startChunk1] = val;
            if (trackIdValues)
                idArrTmp[m + id - startChunk2 + startChunk1] = idTracked;
        }
    }


    template<typename Type>
    __global__ void copyMergedChunkBack(
        const bool trackIdValues,
        const int n,
        Type* __restrict__ arr, Type* __restrict__ arrTmp,
        int* __restrict__ idArr, int* __restrict__ idArrTmp
    )
    {
        const int id = threadIdx.x + blockIdx.x * blockDim.x;

        if (id < n)
        {
            arr[id] = arrTmp[id];
            if(trackIdValues)
                idArr[id] = idArrTmp[id];
        }
    }


    __device__ void reductionWithBlock7x(
         const int count,  int* countCache,
         const int count2, int* countCache2,
         const int count3, int* countCache3,
         const int count4, int* countCache4,
         const int count5, int* countCache5,
         const int count6, int* countCache6,
         const int count7, int* countCache7,

        const int id, const int bd,

        int* output,  int* output2,
        int* output3, int* output4,
        int* output5, int* output6,
        int* output7)
    {        
        int wSum = count;
        int wSum2 = count2;
        int wSum3 = count3;
        int wSum4 = count4;
        int wSum5 = count5;
        int wSum6 = count6;
        int wSum7 = count7;
        for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
        {
            wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
            wSum2 += __shfl_down_sync(0xffffffff, wSum2, wOfs);
            wSum3 += __shfl_down_sync(0xffffffff, wSum3, wOfs);
            wSum4 += __shfl_down_sync(0xffffffff, wSum4, wOfs);
            wSum5 += __shfl_down_sync(0xffffffff, wSum5, wOfs);
            wSum6 += __shfl_down_sync(0xffffffff, wSum6, wOfs);
            wSum7 += __shfl_down_sync(0xffffffff, wSum7, wOfs);

        }
        __syncthreads();
        if (id % 32 == 0)
        {
            countCache[id / 32] = wSum;
            countCache2[id / 32] = wSum2;
            countCache3[id / 32] = wSum3;
            countCache4[id / 32] = wSum4;
            countCache5[id / 32] = wSum5;
            countCache6[id / 32] = wSum6;
            countCache7[id / 32] = wSum7;
        }
        __syncthreads();
        if (id < 32)
        {
            wSum = countCache[id];
            wSum2 = countCache2[id];
            wSum3 = countCache3[id];
            wSum4 = countCache4[id];
            wSum5 = countCache5[id];
            wSum6 = countCache6[id];
            wSum7 = countCache7[id];
            for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
            {
                wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
                wSum2 += __shfl_down_sync(0xffffffff, wSum2, wOfs);
                wSum3 += __shfl_down_sync(0xffffffff, wSum3, wOfs);
                wSum4 += __shfl_down_sync(0xffffffff, wSum4, wOfs);
                wSum5 += __shfl_down_sync(0xffffffff, wSum5, wOfs);
                wSum6 += __shfl_down_sync(0xffffffff, wSum6, wOfs);
                wSum7 += __shfl_down_sync(0xffffffff, wSum7, wOfs);
            }
        }
        __syncthreads();
        if (id == 0)
        {
            countCache[0] = wSum;
            countCache2[0] = wSum2;
            countCache3[0] = wSum3;
            countCache4[0] = wSum4;
            countCache5[0] = wSum5;
            countCache6[0] = wSum6;
            countCache7[0] = wSum7;
        }
        __syncthreads();
        *output = countCache[0];
        *output2 = countCache2[0];        
        *output3 = countCache3[0];
        *output4 = countCache4[0];
        *output5 = countCache5[0];
        *output6 = countCache6[0];
        *output7 = countCache7[0];
        return;
    }


    // all sorting networks taken from: https://bertdobbelaere.github.io/sorting_networks.html#N32L185D14
    // run by single thread
    __constant__ int sortingNetwork4[5][2] = {
        {0,2},{1,3 },
        {0,1},{2,3 },
        {1,2}      
    };

    __constant__ int snCols4[3] = { 2,2,1 };

    // run by warp
    __constant__ int sortingNetwork8[19][2] = {
        {0,2},{1,3},{4,6},{5,7},
        {0,4},{1,5},{2,6},{3,7},
        {0,1},{2,3},{4,5},{6,7},
        {2,4},{3,5},
        {1,4},{3,6},
        {1,2},{3,4},{5,6 }
    };

    __constant__ int snCols8[6] = { 4,4,4,2,2,3 };


    // run by warp
    __constant__ int sortingNetwork16[60][2] = {
        {0,13},{1,12},{2,15},{3,14},{4,8},{5,6},{7,11},{9,10 },
        {0,5},{1,7},{2,9},{3,4},{6,13},{8,14},{10,15},{11,12 },
        {0,1},{2,3},{4,5},{6,8},{7,9},{10,11},{12,13},{14,15 },
        {0,2},{1,3},{4,10},{5,11},{6,7},{8,9},{12,14},{13,15 },
        {1,2},{3,12},{4,6},{5,7},{8,10},{9,11},{13,14 },
        {1,4},{2,6},{5,8},{7,10},{9,13},{11,14 },
        {2,4},{3,6},{9,12},{11,13 },
        {3,5},{6,8},{7,9},{10,12 },
        {3,4},{5,6},{7,8},{9,10},{11,12 },
        {6,7},{8,9}
    };

    __constant__ int snCols16[10] = {8,8,8,8,7,6,4,4,5,2};



    // should fit inside constant cache
    // all sorting networks taken from: https://bertdobbelaere.github.io/sorting_networks.html#N32L185D14
    __constant__ int sortingNetwork32[185][2] = {
        {0,1},{2,3},{4,5},{6,7},{8,9},{10,11},{12,13},{14,15},{16,17},{18,19},{20,21},{22,23},{24,25},{26,27},{28,29},{30,31},
        {0,2},{1,3},{4,6},{5,7},{8,10},{9,11},{12,14},{13,15},{16,18},{17,19},{20,22},{21,23},{24,26},{25,27},{28,30},{29,31},
        {0,4},{1,5},{2,6},{3,7},{8,12},{9,13},{10,14},{11,15},{16,20},{17,21},{18,22},{19,23},{24,28},{25,29},{26,30},{27,31},
        {0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15},{16,24},{17,25},{18,26},{19,27},{20,28},{21,29},{22,30},{23,31},
        {0,16},{1,8},{2,4},{3,12},{5,10},{6,9},{7,14},{11,13},{15,31},{17,24},{18,20},{19,28},{21,26},{22,25},{23,30},{27,29},
        {1,2},{3,5},{4,8},{6,22},{7,11},{9,25},{10,12},{13,14},{17,18},{19,21},{20,24},{23,27},{26,28},{29,30},
        {1,17},{2,18},{3,19},{4,20},{5,10},{7,23},{8,24},{11,27},{12,28},{13,29},{14,30},{21,26},
        {3,17},{4,16},{5,21},{6,18},{7,9},{8,20},{10,26},{11,23},{13,25},{14,28},{15,27},{22,24},
        {1,4},{3,8},{5,16},{7,17},{9,21},{10,22},{11,19},{12,20},{14,24},{15,26},{23,28},{27,30},
        {2,5},{7,8},{9,18},{11,17},{12,16},{13,22},{14,20},{15,19},{23,24},{26,29},
        {2,4},{6,12},{9,16},{10,11},{13,17},{14,18},{15,22},{19,25},{20,21},{27,29},
        {5,6},{8,12},{9,10},{11,13},{14,16},{15,17},{18,20},{19,23},{21,22},{25,26},
        {3,5},{6,7},{8,9},{10,12},{11,14},{13,16},{15,18},{17,20},{19,21},{22,23},{24,25},{26,28},
        {3,4},{5,6},{7,8},{9,10},{11,12},{13,14},{15,16},{17,18},{19,20},{21,22},{23,24},{25,26},{27,28}
    };

    __constant__ int snCols32[14] = {16, 16, 16, 16, 16, 14, 12, 12, 12, 10, 10, 10, 12, 13};


    
    template<int ROWS, int COLS,int NUM, typename Type>
    inline
    __device__ void sortByNetwork(
        int (&sortingNetwork) [ROWS][COLS] ,  int (&snCols)[NUM], 
        Type * arr, int * idArr, Type * cacheData, int * cacheId,
        const int id, const int startIncluded, const bool trackIdValues, const int num)
    {

        if (!trackIdValues)
        {
            if (id < num)
            {
                cacheData[id] = arr[startIncluded + id];
            }

            __syncwarp();

            int ofs = 0;
            for (int i = 0; i < NUM; i++)
            {
                const int cLim = snCols[i];

                if (id < cLim)
                {
                    const int comp0 = sortingNetwork[id + ofs][0]; // this should be optimized. id values are different per thread.
                    const int comp1 = sortingNetwork[id + ofs][1]; // but different blocks need similar id's

                    if (comp0 < num && comp1 < num)
                    {
                        auto arr1 = cacheData[comp0];
                        auto arr2 = cacheData[comp1];

                        if (arr2 < arr1)
                        {
                            cacheData[comp0] = arr2;
                            cacheData[comp1] = arr1;
                        }
                    }
                }
                ofs += cLim;
                __syncwarp();
            }

            if (id < num)
            {
                arr[startIncluded + id] = cacheData[id];

            }

            __syncwarp();
        }
        else
        {

            if (id < num)
            {
                cacheData[id] = arr[startIncluded + id];
                cacheId[id] = idArr[startIncluded + id];
            }

            __syncwarp();

            int ofs = 0;
            for (int i = 0; i < NUM; i++)
            {
                const int cLim = snCols[i];

                if (id < cLim)
                {
                    const int comp0 = sortingNetwork[id + ofs][0];
                    const int comp1 = sortingNetwork[id + ofs][1];

                    if (comp0 < num && comp1 < num)
                    {
                        auto arr1 = cacheData[comp0];
                        auto arr2 = cacheData[comp1];


                        auto id1 = cacheId[comp0];
                        auto id2 = cacheId[comp1];

                        if (arr2 < arr1)
                        {
                            cacheData[comp0] = arr2;
                            cacheData[comp1] = arr1;

                            cacheId[comp0] = id2;
                            cacheId[comp1] = id1;
                        }
                    }
                }
                ofs += cLim;
                __syncwarp();
            }

            if (id < num)
            {
                arr[startIncluded + id] = cacheData[id];
                idArr[startIncluded + id] = cacheId[id];
            }

            __syncwarp();
        }
    }

   
    template<typename Type>
    inline
    __device__ const bool equals(const Type e1, const Type e2)
    {
        return  !(e1 < e2) && !(e1 > e2);
    }

    // task pattern: 
    //              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
    //              start stop  start stop  start stop  start stop  ---> tasks buffer
    //              block 0     block 1     block 2     block 3     ---> cuda blocks
    template<typename Type>
    __global__ void quickSortWithoutStreamCompaction(
        const bool trackIdValues,
        Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idArr,
        Type* __restrict__ arrTmp, int*__restrict__ idArrTmp)
    {
        // 1 block = 1 chunk of data
        const int gid = blockIdx.x;
        const int id = threadIdx.x;
        const int bd = blockDim.x;

        if (id == 0 && gid == 0)
            copyTasksBack << <1, 1024, 0, cudaStreamTailLaunch >> > (trackIdValues, arr, numTasks, tasks, tasks2, tasks3, tasks4, idArr, arrTmp, idArrTmp);


        __shared__ int taskIdCacheStart;
        __shared__ int taskIdCacheStop;
        if (id == 0)
        {
            taskIdCacheStart = tasks[gid * 2];
            taskIdCacheStop = tasks[gid * 2 + 1];
            tasks[gid * 2] = 0;
            tasks[gid * 2 + 1] = 0;
        }
        __syncthreads();
        const int startIncluded = taskIdCacheStart;
        const int stopIncluded = taskIdCacheStop;
        const int num = stopIncluded - startIncluded + 1;


        if (startIncluded == 0 && stopIncluded == 0)
        {
            if (id == 0)
                printf("\n quicksort task id error: %i \n", gid);
            return;
        }

        if (num < 2)
            return;

        __shared__ Type cacheData[DIRECTLY_COMPUTE_LIMIT];
        __shared__ int cacheId[DIRECTLY_COMPUTE_LIMIT];
        // if data is suitable for 32-input sorting-network, sort directly

        if (num <= DIRECTLY_COMPUTE_LIMIT && bd >= DIRECTLY_COMPUTE_LIMIT)
        {
        
            if(num<=4)
                sortByNetwork<5, 2, 3, Type>(sortingNetwork4, snCols4, arr, idArr, cacheData, cacheId, id, startIncluded, trackIdValues, num);
            else if(num<=8)
                sortByNetwork<19, 2, 6, Type>(sortingNetwork8, snCols8, arr, idArr, cacheData, cacheId, id, startIncluded, trackIdValues, num);
            else if(num<=16)
                sortByNetwork<60, 2, 10,Type>(sortingNetwork16, snCols16, arr, idArr, cacheData, cacheId, id, startIncluded, trackIdValues, num);
            else if(num<=32)
                sortByNetwork<185,2,14,Type>(sortingNetwork32,snCols32,arr,idArr,cacheData,cacheId,id,startIncluded,trackIdValues,num);
            return;
        }

        __shared__ int indexLeftLeft;
        __shared__ int indexPivotLeft;
        __shared__ int indexLeft;
        __shared__ int indexPivot;
        __shared__ int indexRight;
        __shared__ int indexPivotRight;
        __shared__ int indexRightRight;

        __shared__ Type pivotLoad[3];
        if (id == 0)
        {
            pivotLoad[0] = arr[startIncluded + (stopIncluded - startIncluded + 1) / 2];
            pivotLoad[1] = arr[startIncluded];
            pivotLoad[2] = arr[stopIncluded];
            compareSwap(pivotLoad, 0, 1);
            compareSwap(pivotLoad, 0, 2);
            compareSwap(pivotLoad, 1, 2);
        }
        __syncthreads();

        const Type pivotLeft = pivotLoad[0];
        const Type pivot = pivotLoad[1];
        const Type pivotRight = pivotLoad[2];



        if (id == 0)
        {
            indexLeftLeft = 0;
            indexPivotLeft = 0;
            indexLeft = 0;
            indexPivot = 0;
            indexRight = 0;
            indexPivotRight = 0;
            indexRightRight = 0;
        }
        __syncthreads();

        // two-pass
        // 1: counting to find borders of chunks
        // 2: moving to a temporary array & copying back to original                
        int countPivotLeft=0;
        int countPivot=0;
        int countPivotRight=0;
        int countLeftLeft=0;
        int countLeft=0;
        int countRight=0;
        int countRightRight=0;
        const int stepsArray = (num / bd) + 1;
        for (int i = 0; i < stepsArray; i++)
        {
            const int curId = i * bd + id;
            if (curId < num)
            {
                const Type data = arr[curId + startIncluded];                
                if (equals(data,pivotLeft))
                {
                    countPivotLeft++;
                }
                else if (equals(data,pivot))
                {
                    countPivot++;
                }
                else if (equals(data,pivotRight))
                {
                    countPivotRight++;
                }
                else
                {
                    if (data < pivotLeft)
                    {
                        countLeftLeft++;
                    }
                    else if (data < pivot)
                    {
                        countLeft++;
                    }
                    else if (data < pivotRight)
                    {
                        countRight++;
                    }
                    else if (data > pivotRight)
                    {
                        countRightRight++;
                    }
                }

            }
        }
        __syncthreads();
        __shared__ int countCache[32];
        __shared__ int countCache2[32];
        __shared__ int countCache3[32];
        __shared__ int countCache4[32];
        __shared__ int countCache5[32];
        __shared__ int countCache6[32];
        __shared__ int countCache7[32];

        // sum of all counts (reducing number of atomicAdd calls)
        int nLeftLeft = 0;
        int nPivotLeft = 0;
        int nLeft = 0;
        int nPivot = 0;
        int nRight = 0;
        int nPivotRight = 0;
        int nRightRight = 0;
        reductionWithBlock7x(
            countLeftLeft, countCache, 
            countPivotLeft, countCache2, 
            countLeft, countCache3,
            countPivot,countCache4,
            countRight, countCache5,
            countPivotRight, countCache6,
            countRightRight, countCache7,

            id, bd, 
            
            &nLeftLeft, &nPivotLeft,
            &nLeft, &nPivot,
            &nRight, &nPivotRight,
            &nRightRight);

        const int offsetLeftLeft = startIncluded;
        const int offsetPivotLeft = offsetLeftLeft+nLeftLeft;
        const int offsetLeft = offsetPivotLeft + nPivotLeft;
        const int offsetPivot = offsetLeft + nLeft;
        const int offsetRight = offsetPivot + nPivot;
        const int offsetPivotRight = offsetRight + nRight;
        const int offsetRightRight = offsetPivotRight + nPivotRight;

        __syncthreads();
        for (int i = 0; i < stepsArray; i++)
        {
            const int curId = i * bd + id;
            if (curId < num)
            {
                const Type data = arr[curId + startIncluded];
                int dataId = 0;
                if(trackIdValues)
                    dataId = idArr[curId + startIncluded];
                // todo: pivot values are known so they are "counting", no need to copy values anywhere
                // but id values will be required
                if (equals(data,pivotLeft))
                {
                    auto idx = atomicAdd(&indexPivotLeft, 1);
                    if (trackIdValues)
                        idArrTmp[offsetPivotLeft+idx] = dataId;

                }
                else if (equals(data,pivot))
                {
                    auto idx = atomicAdd(&indexPivot, 1);
                    if (trackIdValues)
                        idArrTmp[offsetPivot + idx] = dataId;
                }
                else if (equals(data,pivotRight))
                {
                    auto idx = atomicAdd(&indexPivotRight, 1);
                    if (trackIdValues)
                        idArrTmp[offsetPivotRight + idx] = dataId;                    
                }
                else
                {
                    if (data < pivotLeft)
                    {
                        auto idx = atomicAdd(&indexLeftLeft, 1);
                        if (trackIdValues)
                            idArrTmp[offsetLeftLeft + idx] = dataId;
                        arrTmp[offsetLeftLeft + idx] = data;
                    }
                    else if (data < pivot)
                    {
                        auto idx = atomicAdd(&indexLeft, 1);
                        if (trackIdValues)
                            idArrTmp[offsetLeft + idx] = dataId;
                        arrTmp[offsetLeft + idx] = data;
                    }
                    else if (data < pivotRight)
                    {
                        auto idx = atomicAdd(&indexRight, 1);
                        if (trackIdValues)
                            idArrTmp[offsetRight + idx] = dataId;
                        arrTmp[offsetRight + idx] = data;
                    }
                    else if (data > pivotRight)
                    {
                        auto idx = atomicAdd(&indexRightRight, 1);
                        if (trackIdValues)
                            idArrTmp[offsetRightRight + idx] = dataId;
                        arrTmp[offsetRightRight + idx] = data;
                    }
                }

            }
        }


        __syncthreads();
        // copying from temporary to real data
        for (int i = 0; i < stepsArray; i++)
        {
            const int curId = i * bd + id;
            if (curId < num)
            {

                if (curId >= nLeftLeft + nPivotLeft + nLeft && curId  < nLeftLeft + nPivotLeft + nLeft + nPivot)
                    arr[startIncluded + curId] = pivot;
                else if (curId >= nLeftLeft  && curId < nLeftLeft + nPivotLeft)
                    arr[startIncluded + curId] = pivotLeft;
                else if (curId >= nLeftLeft + nPivotLeft + nLeft + nPivot + nRight  && curId < nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight)
                    arr[startIncluded + curId] = pivotRight;
                else
                    arr[startIncluded + curId] = arrTmp[startIncluded + curId];


                if (trackIdValues)
                    idArr[startIncluded + curId] = idArrTmp[startIncluded + curId];
            }
        }
        __syncthreads();




        if (id == 0)
        {
            if (nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight + nRightRight != num)
                printf(" @@ ERROR: wrong partition values!! @@");

            if (nLeftLeft > 1)
            {
               
                if (nLeftLeft <= BRUTE_FORCE_LIMIT && nLeftLeft > DIRECTLY_COMPUTE_LIMIT) // push new "brute-force" task
                {
                    
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded;
                    tasks4[index * 2 + 1] = startIncluded + nLeftLeft - 1;
                }
                else// push new "quick" task
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded;
                    tasks2[index * 2 + 1] = startIncluded + nLeftLeft - 1;
                }
            }

            if (nLeft > 1)
            {

                if (nLeft <= BRUTE_FORCE_LIMIT && nLeft > DIRECTLY_COMPUTE_LIMIT) // push new "brute-force" task
                {
                    
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded + nLeftLeft + nPivotLeft;
                    tasks4[index * 2 + 1] = startIncluded + nLeftLeft + nPivotLeft + nLeft - 1;
                }
                else// push new "quick" task
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded + nLeftLeft + nPivotLeft;
                    tasks2[index * 2 + 1] = startIncluded + nLeftLeft + nPivotLeft + nLeft - 1;
                }
            }

            if (nRight > 1)
            {
                if (nRight <= BRUTE_FORCE_LIMIT && nRight > DIRECTLY_COMPUTE_LIMIT) // push new "brute-force" task
                {
                    
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot;
                    tasks4[index * 2 + 1] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight - 1;
                }
                else // push new "quick" task
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot;
                    tasks2[index * 2 + 1] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight - 1;
                }

            }

            if (nRightRight > 1)
            {
                if (nRightRight <= BRUTE_FORCE_LIMIT && nRightRight > DIRECTLY_COMPUTE_LIMIT) // push new "brute-force" task
                {
                  
                    const int index = atomicAdd(&numTasks[1], 1);
                    tasks4[index * 2] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight;
                    tasks4[index * 2 + 1] = stopIncluded;
                }
                else // push new "quick" task
                {
                    const int index = atomicAdd(&numTasks[0], 1);
                    tasks2[index * 2] = startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight;
                    tasks2[index * 2 + 1] = stopIncluded;
                }

            }
        }
    }


    __global__ void resetTasks(int* tasks, int* tasks2, int* tasks3, int* tasks4, const int n)
    {
        const int id = threadIdx.x + blockIdx.x * blockDim.x;
        if (id < n)
        {
            tasks[id] = 0;
            tasks2[id] = 0;
            tasks3[id] = 0;
            tasks4[id] = 0;
        }
    }

    template<typename Type>
    __global__ void quickSortMain(
        int n,
        Type* __restrict__ data, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,      
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp)
    {

        cudaStream_t stream0;
        cudaStreamCreateWithFlags(&stream0, (unsigned int)cudaStreamNonBlocking);

        __syncthreads();

        cudaStreamDestroy(stream0);
    }



    // int data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues, 
            int* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            int* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template
    __global__ void mergeSortedChunks(
        const bool trackIdValues,int* __restrict__ tasks,int* __restrict__ arr, 
        int* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, int* __restrict__ arr,
            int* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);
    


    // short data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            short* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            short* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, short* __restrict__ arr,
            short* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, short* __restrict__ arr,
            short* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);



    // char data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            char* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            char* __restrict__ arrTmp, int* __restrict__ idArrTmp);



    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, char* __restrict__ arr,
            char* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, char* __restrict__ arr,
            char* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);


    // long data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            long* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,            
            int* __restrict__ idData,
            long* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, long* __restrict__ arr,
            long* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, long* __restrict__ arr,
            long* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);







    // unsigned int
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            unsigned int* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned int* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, unsigned int* __restrict__ arr,
            unsigned int* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, unsigned int* __restrict__ arr,
            unsigned int* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);




    // unsigned short data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            unsigned short* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned short* __restrict__ arrTmp, int* __restrict__ idArrTmp);



    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, unsigned short* __restrict__ arr,
            unsigned short* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, unsigned short* __restrict__ arr,
            unsigned short* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);


    // unsigned char data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            unsigned char* __restrict__ data,int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned char* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, unsigned char* __restrict__ arr,
            unsigned char* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, unsigned char* __restrict__ arr,
            unsigned char* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);





    // unsigned long data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            unsigned long* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,            
            int* __restrict__ idData,
            unsigned long* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, unsigned long* __restrict__ arr,
            unsigned long* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, unsigned long* __restrict__ arr,
            unsigned long* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);


    // float data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            float* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            float* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, float* __restrict__ arr,
            float * __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, float* __restrict__ arr,
            float* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    // double data
    template
        __global__ void copyTasksBack(
            const bool trackIdValues,
            double* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            double* __restrict__ arrTmp, int* __restrict__ idArrTmp);


    template
        __global__ void mergeSortedChunks(
            const bool trackIdValues, int* __restrict__ tasks, double* __restrict__ arr,
            double* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);

    template
        __global__ void copyMergedChunkBack(
            const bool trackIdValues, const int n, double* __restrict__ arr,
            double* __restrict__ arrTmp, int* __restrict__ idArr, int* __restrict__ idArrTmp);


}