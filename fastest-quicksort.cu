
#include"fastest-quicksort.cuh"



constexpr int BRUTE_FORCE_LIMIT = 1024;



// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
template<typename Type>
__global__ void quickSortWithoutStreamCompaction(
    Type* __restrict__ arr, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight);

template<typename Type>
__global__ void bruteSort(Type* __restrict__ arr, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight);

template<typename Type>
__global__ void resetNumTasks(Type* __restrict__ data, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight)
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
            bruteSort << <numTasks[3], BRUTE_FORCE_LIMIT, 0, cudaStreamFireAndForget >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);


        if (numTasks[2] > 0)
        {
            //quickSortWithoutStreamCompaction << <numTasks[2], 1024, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4,leftLeft,rightRight);
            
            if (numTask1 < 64)
                quickSortWithoutStreamCompaction << <numTasks[2], 1024, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
            else if (numTask1 < 64*16)
                quickSortWithoutStreamCompaction << <numTasks[2], 512, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
            else if (numTask1 < 64*16*16)
                quickSortWithoutStreamCompaction << <numTasks[2], 256, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
            else if (numTask1 < 64 * 16 * 16 * 16)
                quickSortWithoutStreamCompaction << <numTasks[2], 128, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
            else
                quickSortWithoutStreamCompaction << <numTasks[2], 64, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
                
        }


    }
}




template<typename Type>
__global__ void copyTasksBack(Type* __restrict__ data, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight)
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

        resetNumTasks << <1, 1, 0, cudaStreamTailLaunch >> > (data, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);
    }

}



#define compareSwap(a,x,y) if(a[y]<a[x]){a[x]^=a[y];a[y]^=a[x];a[x]^=a[y];}
#define compSw(a,x,y) if(a[y]<a[x]){ auto t = a[x];a[x]=a[y];a[y]=t;}

template<typename Type>
__global__ void bruteSort(Type* __restrict__ arr, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight)
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
    if (startIncluded + id <= stopIncluded)
    {
        cache[id] = arr[startIncluded + id];
    }
    __syncthreads();


    for (int i = 0; i < num; i++)
    {
        if (id + 1 < num)
        {
            if ((id % 2 == 0))
            {
                compSw(cache, id, id + 1)
            }
        }
        __syncthreads();
        if (id + 1 < num)
        {
            if ((id % 2 == 1))
            {
                compSw(cache, id, id + 1)
            }
        }
        __syncthreads();
    }


    if (startIncluded + id <= stopIncluded)
    {
        arr[startIncluded + id] = cache[id];
    }
}

// task pattern: 
//              task 0      task 1      task 2      task 3      ---> array chunks to sort (no overlap)
//              start stop  start stop  start stop  start stop  ---> tasks buffer
//              block 0     block 1     block 2     block 3     ---> cuda blocks
template<typename Type>
__global__ void quickSortWithoutStreamCompaction(
    Type* __restrict__ arr, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight)
{
    // 1 block = 1 chunk of data
    const int gid = blockIdx.x;
    const int id = threadIdx.x;
    const int bd = blockDim.x;

    if (id == 0 && gid == 0)
        copyTasksBack << <1, 1024, 0, cudaStreamTailLaunch >> > (arr, left, right, numTasks, tasks, tasks2, tasks3, tasks4, leftLeft, rightRight);


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

    if (num == 2)
    {
        if (id == 0)
        {
            compSw(arr, startIncluded, startIncluded + 1);
        }

        return;
    }
    else if (num == 3)
    {
        if (id == 0)
        {
            compSw(arr, startIncluded, startIncluded + 1);
            compSw(arr, startIncluded, startIncluded + 2);
            compSw(arr, startIncluded + 1, startIncluded + 2);
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

    if (id == 0)
    {
        pivotLoad[0] = arr[startIncluded + (stopIncluded - startIncluded + 1) / 2];
        pivotLoad[1] = arr[startIncluded];
        pivotLoad[2] = arr[stopIncluded];
        compSw(pivotLoad, 0, 1);
        compSw(pivotLoad, 0, 2);
        compSw(pivotLoad, 1, 2);
    }
    __syncthreads();

    const Type pivotLeft = pivotLoad[0];
    const Type pivot = pivotLoad[1];
    const Type pivotRight = pivotLoad[2];

    int nLeftLeft = 0;
    int nPivotLeft = 0;
    int nLeft = 0;
    int nPivot = 0;
    int nRight = 0;
    int nPivotRight = 0;
    int nRightRight = 0;
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

    const int stepsArray = (num / bd) + 1;
    for (int i = 0; i < stepsArray; i++)
    {
        const int curId = i * bd + id;
        if (curId < num)
        {
            const Type data = arr[curId + startIncluded];
            if (data == pivotLeft)
                atomicAdd(&indexPivotLeft, 1);
            else if (data == pivot)
                atomicAdd(&indexPivot, 1);
            else if (data == pivotRight)
                atomicAdd(&indexPivotRight, 1);
            else
            {

                if (data < pivotLeft)
                    leftLeft[startIncluded + atomicAdd(&indexLeftLeft, 1)] = data;
                else if (data < pivot)
                    left[startIncluded + atomicAdd(&indexLeft, 1)] = data;
                else if (data < pivotRight)
                    right[startIncluded + atomicAdd(&indexRight, 1)] = data;
                else if (data > pivotRight)
                    rightRight[startIncluded + atomicAdd(&indexRightRight, 1)] = data;
            }

        }
    }


    __syncthreads();
    nLeftLeft = indexLeftLeft;
    nPivotLeft = indexPivotLeft;
    nLeft = indexLeft;
    nPivot = indexPivot;
    nRight = indexRight;
    nPivotRight = indexPivotRight;
    nRightRight = indexRightRight;


    // move leftleft
    const int stepsLeftLeft = (nLeftLeft / bd) + 1;
    for (int i = 0; i < stepsLeftLeft; i++)
    {
        const int curId = i * bd + id;
        if (curId < nLeftLeft)
        {
            arr[curId + startIncluded] = leftLeft[startIncluded + curId];
        }
    }

    // move pivotleft
    const int stepsPivotLeft = (nPivotLeft / bd) + 1;
    for (int i = 0; i < stepsPivotLeft; i++)
    {
        const int curId = i * bd + id;
        if (curId < nPivotLeft)
        {
            arr[curId + startIncluded + nLeftLeft] = pivotLeft;
        }
    }

    // move left
    const int stepsLeft = (nLeft / bd) + 1;
    for (int i = 0; i < stepsLeft; i++)
    {
        const int curId = i * bd + id;
        if (curId < nLeft)
        {
            arr[curId + startIncluded + nLeftLeft + nPivotLeft] = left[startIncluded + curId];

        }
    }


    // move mid (pivot)
    const int stepsMid = (nPivot / bd) + 1;
    for (int i = 0; i < stepsMid; i++)
    {
        const int curId = i * bd + id;
        if (curId < nPivot)
        {
            arr[curId + startIncluded + nLeftLeft + nPivotLeft + nLeft] = pivot;
        }
    }



    // move right
    const int stepsRight = (nRight / bd) + 1;
    for (int i = 0; i < stepsRight; i++)
    {
        const int curId = i * bd + id;
        if (curId < nRight)
        {
            arr[curId + startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot] = right[startIncluded + curId];
        }
    }

    // move pivot right
    const int stepsPivotRight = (nPivotRight / bd) + 1;
    for (int i = 0; i < stepsPivotRight; i++)
    {
        const int curId = i * bd + id;
        if (curId < nPivotRight)
        {
            arr[curId + startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight] = pivotRight;
        }
    }

    // move right right
    const int stepsRightRight = (nRightRight / bd) + 1;
    for (int i = 0; i < stepsRightRight; i++)
    {
        const int curId = i * bd + id;
        if (curId < nRightRight)
        {
            arr[curId + startIncluded + nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight] = rightRight[startIncluded + curId];
        }
    }

    __syncthreads();



    if (id == 0)
    {
        if (nLeftLeft + nPivotLeft + nLeft + nPivot + nRight + nPivotRight + nRightRight != num)
            printf(" @@ ERROR: wrong partition values!! @@");

        if (nLeftLeft > 1)
        {

            if (nLeftLeft <= BRUTE_FORCE_LIMIT && nLeftLeft > 3) // push new "brute-force" task
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

            if (nLeft <= BRUTE_FORCE_LIMIT && nLeft > 3) // push new "brute-force" task
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
            if (nRight <= BRUTE_FORCE_LIMIT && nRight > 3) // push new "brute-force" task
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
            if (nRightRight <= BRUTE_FORCE_LIMIT && nRightRight > 3) // push new "brute-force" task
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
    Type* __restrict__ data, Type* __restrict__ left, Type* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    Type* __restrict__ leftLeft, Type* __restrict__ rightRight)
{

    cudaStream_t stream0;
    cudaStreamCreateWithFlags(&stream0, (unsigned int)cudaStreamNonBlocking);

    __syncthreads();

    cudaStreamDestroy(stream0);
}



// int data
template
__global__ void copyTasksBack(int* __restrict__ data, int* __restrict__ left, int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    int* __restrict__ leftLeft, int* __restrict__ rightRight);

// short data
template
__global__ void copyTasksBack(short* __restrict__ data, short* __restrict__ left, short* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    short* __restrict__ leftLeft, short* __restrict__ rightRight);

// char data
template
__global__ void copyTasksBack(char* __restrict__ data, char* __restrict__ left, char* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    char* __restrict__ leftLeft, char* __restrict__ rightRight);

// long data
template
__global__ void copyTasksBack(long* __restrict__ data, long* __restrict__ left, long* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    long* __restrict__ leftLeft, long* __restrict__ rightRight);










// unsigned int
template
__global__ void copyTasksBack(unsigned int* __restrict__ data, unsigned int* __restrict__ left, unsigned int* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    unsigned int* __restrict__ leftLeft, unsigned int* __restrict__ rightRight);

// unsigned short data
template
__global__ void copyTasksBack(unsigned short* __restrict__ data, unsigned short* __restrict__ left, unsigned short* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    unsigned short* __restrict__ leftLeft, unsigned short* __restrict__ rightRight);

// unsigned char data
template
__global__ void copyTasksBack(unsigned char* __restrict__ data, unsigned char* __restrict__ left, unsigned  char* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    unsigned char* __restrict__ leftLeft, unsigned  char* __restrict__ rightRight);

// unsigned long data
template
__global__ void copyTasksBack(unsigned long* __restrict__ data, unsigned long* __restrict__ left, unsigned long* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    unsigned long* __restrict__ leftLeft, unsigned long* __restrict__ rightRight);



// float data
template
__global__ void copyTasksBack(float* __restrict__ data, float* __restrict__ left, float* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    float* __restrict__ leftLeft, float* __restrict__ rightRight);

// double data
template
__global__ void copyTasksBack(double* __restrict__ data, double* __restrict__ left, double* __restrict__ right, int* __restrict__ numTasks,
    int* __restrict__ tasks, int* __restrict__ tasks2, int* __restrict__ tasks3, int* __restrict__ tasks4,
    double* __restrict__ leftLeft, double* __restrict__ rightRight);


