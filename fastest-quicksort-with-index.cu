
#include"fastest-quicksort-with-index.cuh"

namespace QuickIndex
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
        Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template<typename Type>
    __global__ void bruteSort(Type* __restrict__ arr, int* __restrict__ numTasks,
        int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,        
        int* __restrict__ idData,
        Type* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    template<typename Type>
    __global__ void resetNumTasks(Type* __restrict__ data, int* __restrict__ numTasks,
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
            //printf("\n %i %i \n", numTasks[2], numTasks[3]);
            __syncthreads();

            if (numTasks[3] > 0)
                bruteSort << <numTasks[3], BRUTE_FORCE_LIMIT, 0, cudaStreamFireAndForget >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);


            if (numTasks[2] > 0)
            {
                quickSortWithoutStreamCompaction << <numTasks[2], 1024, 0, cudaStreamTailLaunch >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
                /*
                if (numTask1 < 64)
                    quickSortWithoutStreamCompaction << <numTasks[2], 1024, 0, cudaStreamTailLaunch >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
                else if (numTask1 < 64 * 16)
                    quickSortWithoutStreamCompaction << <numTasks[2], 512, 0, cudaStreamTailLaunch >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
                else if (numTask1 < 64 * 16 * 16)
                    quickSortWithoutStreamCompaction << <numTasks[2], 256, 0, cudaStreamTailLaunch >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
                else
                    quickSortWithoutStreamCompaction << <numTasks[2], DIRECTLY_COMPUTE_LIMIT, 0, cudaStreamTailLaunch >> > (data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
              */

            }


        }
    }




    template<typename Type>
    __global__ void copyTasksBack(Type* __restrict__ data, int* __restrict__ numTasks,
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
                data, numTasks, tasks, tasks2, tasks3, tasks4, idData, arrTmp, idArrTmp);
        }

    }



#define compareSwap(a,x,y) if(a[y]<a[x]){auto t = a[x];a[x]=a[y];a[y]=t;}
#define compSw(a,x,y,b) if(a[y]<a[x]){ auto t = a[x];a[x]=a[y];a[y]=t; auto u = b[x];b[x]=b[y];b[y]=u;}

    template<typename Type>
    __global__ void bruteSort(Type* __restrict__ arr, int* __restrict__ numTasks,
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
            idArr[startIncluded + id] = idTracker[id];
        }
    }


    __device__ void reductionWithBlock2x(
        int count, int * countCache,
        int count2, int* countCache2,
        const int id, const int bd,
        int * output, int * output2)
    {
        
        int wSum = count;
        int wSum2 = count2;
        for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
        {
            wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
            wSum2 += __shfl_down_sync(0xffffffff, wSum2, wOfs);
        }
        __syncthreads();
        if (id % 32 == 0)
        {
            countCache[id / 32] = wSum;
            countCache2[id / 32] = wSum2;
        }
        __syncthreads();
        if (id < 32)
        {
            wSum = countCache[id];
            wSum2 = countCache2[id];
            for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
            {
                wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
                wSum2 += __shfl_down_sync(0xffffffff, wSum2, wOfs);
            }
        }
        __syncthreads();
        if (id == 0)
        {
            countCache[0] = wSum;
            countCache2[0] = wSum2;
        }
        __syncthreads();
        *output = countCache[0];
        *output2 = countCache2[0];        
        return;
    }



    __device__ void reductionWithBlock1x(
        int count, int* countCache,
        const int id, const int bd,
        int* output)
    {

        int wSum = count;

        for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
        {
            wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
        }
        __syncthreads();
        if (id % 32 == 0)
        {
            countCache[id / 32] = wSum;

        }
        __syncthreads();
        if (id < 32)
        {
            wSum = countCache[id];

            for (unsigned int wOfs = 16; wOfs > 0; wOfs >>= 1)
            {
                wSum += __shfl_down_sync(0xffffffff, wSum, wOfs);
            }
        }
        __syncthreads();
        if (id == 0)
        {
            countCache[0] = wSum;
        }
        __syncthreads();
        *output = countCache[0];

        return;
    }

    // task pattern: 
    //              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
    //              start stop  start stop  start stop  start stop  ---> tasks buffer
    //              block 0     block 1     block 2     block 3     ---> cuda blocks
    template<typename Type>
    __global__ void quickSortWithoutStreamCompaction(
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
            copyTasksBack << <1, 1024, 0, cudaStreamTailLaunch >> > (arr, numTasks, tasks, tasks2, tasks3, tasks4, idArr, arrTmp, idArrTmp);


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


        if (num >= 2 && num <= DIRECTLY_COMPUTE_LIMIT && bd>= DIRECTLY_COMPUTE_LIMIT)
        {

            // warp-level odd-even parallel sort
            for (int i = 0; i < num; i++)
            {
                if (id + 1 < num)
                {
                    if ((id % 2 == 0))
                    {
                        compSw(arr, startIncluded+id, startIncluded+id + 1, idArr);
                    }
                }
                __syncthreads();
                if (id + 1 < num)
                {
                    if ((id % 2 == 1))
                    {
                        compSw(arr, startIncluded+id, startIncluded+id + 1, idArr);
                    }
                }
                __syncthreads();
            }
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
        __shared__ int idLoad[3];
        if (id == 0)
        {
            pivotLoad[0] = arr[startIncluded + (stopIncluded - startIncluded + 1) / 2];
            pivotLoad[1] = arr[startIncluded];
            pivotLoad[2] = arr[stopIncluded];
            idLoad[0] = idArr[startIncluded + (stopIncluded - startIncluded + 1) / 2];
            idLoad[1] = idArr[startIncluded];
            idLoad[2] = idArr[stopIncluded];
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
                if (data == pivotLeft)
                {
                    //atomicAdd(&indexPivotLeft, 1);
                    countPivotLeft++;
                }
                else if (data == pivot)
                {
                    //atomicAdd(&indexPivot, 1);
                    countPivot++;
                }
                else if (data == pivotRight)
                {
                    //atomicAdd(&indexPivotRight, 1);
                    countPivotRight++;
                }
                else
                {
                    if (data < pivotLeft)
                    {
                        //atomicAdd(&indexLeftLeft, 1);
                        countLeftLeft++;
                    }
                    else if (data < pivot)
                    {
                        //atomicAdd(&indexLeft, 1);
                        countLeft++;
                    }
                    else if (data < pivotRight)
                    {
                        //atomicAdd(&indexRight, 1);
                        countRight++;
                    }
                    else if (data > pivotRight)
                    {
                        //atomicAdd(&indexRightRight, 1);
                        countRightRight++;
                    }
                }

            }
        }
        __syncthreads();
        __shared__ int countCache[32];
        __shared__ int countCache2[32];

        // sum of all counts (reducing number of atomicAdd calls)
        int nLeftLeft = 0;
        int nPivotLeft = 0;
        int nLeft = 0;
        int nPivot = 0;
        int nRight = 0;
        int nPivotRight = 0;
        int nRightRight = 0;
        reductionWithBlock2x(countLeftLeft, countCache, countPivotLeft, countCache2, id, bd, &nLeftLeft, &nPivotLeft);
        reductionWithBlock2x(countLeft, countCache, countPivot, countCache2, id, bd, &nLeft, &nPivot);
        reductionWithBlock2x(countRight, countCache, countPivotRight, countCache2, id, bd, &nRight, &nPivotRight);
        reductionWithBlock1x(countRightRight, countCache, id, bd, &nRightRight);





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
                const int dataId = idArr[curId + startIncluded];
                // todo: pivot values are known so they are "counting", no need to copy values anywhere
                // but id values will be required
                if (data == pivotLeft)
                {
                    auto idx = atomicAdd(&indexPivotLeft, 1);
                    idArrTmp[offsetPivotLeft+idx] = dataId;
                    arrTmp[offsetPivotLeft + idx] = data;
                }
                else if (data == pivot)
                {
                    auto idx = atomicAdd(&indexPivot, 1);
                    idArrTmp[offsetPivot + idx] = dataId;
                    arrTmp[offsetPivot + idx] = data;
                }
                else if (data == pivotRight)
                {
                    auto idx = atomicAdd(&indexPivotRight, 1);
                    idArrTmp[offsetPivotRight + idx] = dataId;
                    arrTmp[offsetPivotRight + idx] = data;
                }
                else
                {
                    if (data < pivotLeft)
                    {
                        auto idx = atomicAdd(&indexLeftLeft, 1);
                        idArrTmp[offsetLeftLeft + idx] = dataId;
                        arrTmp[offsetLeftLeft + idx] = data;
                    }
                    else if (data < pivot)
                    {
                        auto idx = atomicAdd(&indexLeft, 1);
                        idArrTmp[offsetLeft + idx] = dataId;
                        arrTmp[offsetLeft + idx] = data;
                    }
                    else if (data < pivotRight)
                    {
                        auto idx = atomicAdd(&indexRight, 1);
                        idArrTmp[offsetRight + idx] = dataId;
                        arrTmp[offsetRight + idx] = data;
                    }
                    else if (data > pivotRight)
                    {
                        auto idx = atomicAdd(&indexRightRight, 1);
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
                arr[startIncluded + curId] = arrTmp[startIncluded + curId];
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
        __global__ void copyTasksBack(int* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            int* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // short data
    template
        __global__ void copyTasksBack(short* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            short* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // char data
    template
        __global__ void copyTasksBack(char* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            char* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // long data
    template
        __global__ void copyTasksBack(long* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,            
            int* __restrict__ idData,
            long* __restrict__ arrTmp, int* __restrict__ idArrTmp);










    // unsigned int
    template
        __global__ void copyTasksBack(unsigned int* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned int* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // unsigned short data
    template
        __global__ void copyTasksBack(unsigned short* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned short* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // unsigned char data
    template
        __global__ void copyTasksBack(unsigned char* __restrict__ data,int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            unsigned char* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // unsigned long data
    template
        __global__ void copyTasksBack(unsigned long* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,            
            int* __restrict__ idData,
            unsigned long* __restrict__ arrTmp, int* __restrict__ idArrTmp);



    // float data
    template
        __global__ void copyTasksBack(float* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            float* __restrict__ arrTmp, int* __restrict__ idArrTmp);

    // double data
    template
        __global__ void copyTasksBack(double* __restrict__ data, int* __restrict__ numTasks,
            int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
            int* __restrict__ idData,
            double* __restrict__ arrTmp, int* __restrict__ idArrTmp);





}