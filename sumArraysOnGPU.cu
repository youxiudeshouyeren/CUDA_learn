#include <cuda_runtime.h>
#include <stdio.h>
#include<math.h>
#include<time.h>

#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSucces)                                             \
        {                                                                    \
            printf("Error:%s:%d,", __FILE__, __LINE__);                      \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }                                                                        \

void checkResult(float *hostRef, float *deviceRef, const int N)
{
    double epsilon = 1e-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - deviceRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("The CPU result is %5.5f, the GPU result is%5.5f ,index is %d\n", hostRef[i], deviceRef[i], i);
            break;
        }
    }

    if (match) {
        printf("Arrays match with the difference of %2.8f\n", epsilon);
    }
}


void initialData(float *p, int size)
{
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        p[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;
    printf("Vector size %d\n", nElem);

    //申请host内存
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);


    //初始化设备全局内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    //搬运数据
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    //在host执行核函数
    dim3 block(nElem);
    dim3 grid(nElem / block.x);

    sumArrayOnGPU <<< grid, block>>>(d_A, d_B, d_C);
    printf("Execution configuration <<<%d,%d>>>\n", grid.x, block.x);

    //从gpu搬运数据至cpu
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    //结果检查
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    //释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;


}

