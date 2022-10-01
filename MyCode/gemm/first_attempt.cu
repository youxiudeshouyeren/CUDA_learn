#include <omp.h>

#include <Eigen/Core>
#include <ctime>

//一个线程计算结果矩阵的一个元素 负责K次乘累加运算
__global__ void gemmKernel(const float *A, const float *B, float *C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K)
{
    //根据线程索引计算元素位置的偏移量
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    if (m >= M || n >= N) return;
    float c = 0;
    for (unsigned k = 0; k < K; ++k) {
        c += A[m * K + k] * B[k * N + n];
    }
    c = c * alpha;
    float result = c;
    if (beta != 0) {
        result = result + C[m * N + n] * beta;
    }
    C[m * N + n] = result;
}

//启动核函数
void gemmNaive(const float *A, const float *B, float *C, float alpha,
               float beta, unsigned M, unsigned N, unsigned K)
{
    dim3 block(32, 32);
    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

    gemmKernel <<< grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

//获取GPU计算能力
void getGPUInfo()
{   
    int gpu_index=0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_index);
    printf("GPU Name = %s\n", prop.name);
    printf("Compute Capability = %d.%d\n", prop.major, prop.minor); // 获得 SM 版本
    printf("GPU SMs = %d\n", prop.multiProcessorCount); // 获得 SM 数目
    printf("GPU SM clock rate = %.3f GHz\n", prop.clockRate / 1e6); // prop.clockRate 单位为 kHz，除以 1e6 之后单位为 GHz
    printf("GPU Mem clock rate = %.3f GHz\n", prop.memoryClockRate / 1e6); // 同上
    if ((prop.major == 8) && (prop.minor == 0)) { // SM 8.0，即 A100
// 根据公式计算峰值吞吐，其中 64、32、256、128 等数字是从前文 SM 吞吐表中查到
        printf("-----------CUDA Core Performance------------\n");
        printf("FP32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
        printf("FP64 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 32 * 2);
        printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
        printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 128 * 2);
        printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 256 * 2);
        printf("-----------Tensor Core Dense Performance------------\n");
        printf("TF32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 512 * 2);
        printf("FP64 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 64 * 2);
        printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
        printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
        printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
        printf("INT4 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
        printf("INT1 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 16384 * 2);
        printf("-----------Tensor Core Sparse Performance------------\n");
        printf("TF32 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 1024 * 2);
        printf("FP16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
        printf("BF16 Peak Performance = %.3f GFLOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 2048 * 2);
        printf("INT8 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 4096 * 2);
        printf("INT4 Peak Performance = %.3f GOPS.\n", prop.multiProcessorCount * (prop.clockRate / 1e6) * 8192 * 2);

    }
}

int main()
{

    getGPUInfo();
    omp_set_num_threads(omp_get_num_procs());
    unsigned M = 1024, N = 1024, K = 1024;
    float alpha = 1., beta = 0.;
    float *deviceAPrt, *deviceBPtr, *deviceCPtr;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K},
          B{K, N}, C{M, N};
    A.setRandom();
    B.setRandom();
    C.setRandom();
    cudaMalloc(&deviceAPrt, M * K * sizeof(float));
    cudaMemcpy(deviceAPrt, A.data(), M * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc(&deviceBPtr, K * N * sizeof(float));
    cudaMemcpy(deviceBPtr, B.data(), K * N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc(&deviceCPtr, M * N * sizeof(float));
    cudaMemcpy(deviceCPtr, C.data(), M * N * sizeof(float),
               cudaMemcpyHostToDevice);

    //使用cuda Event给核函数计时
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    gemmNaive(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU use: %.3f(ms)\n", milliseconds);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    //CPU数值验证
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    hostResult{M, N}, deviceResult{M, N};
    clock_t begin, end;
    begin = clock();
    hostResult = alpha * (A * B) + beta * C;
    end = clock();
    printf("CPU use: %.3f(ms)\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);
    cudaMemcpy(deviceResult.data(), deviceCPtr, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
        (hostResult - deviceResult).array().abs();
    printf("Max Error: %f\n", diffArray.maxCoeff());

    double GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput: %.3f GFLOPS\n", GFLOPS);
}
