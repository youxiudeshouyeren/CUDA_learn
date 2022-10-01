#include "util.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace
{
    __global__ void gemmKernel(const float *__restrict__ A, const float *__restrict__ B,
                               float *__restrict__ C, float alpha, float beta, unsigned M, unsigned N, unsigned K)
    {
        constexpr unsigned kCount = sizeof(float4) / sizeof(float);
        unsigned int m = (threadIdx.x + blockDim.x * blockIdx.x) * kCount;
        unsigned int n = (threadIdx.y + blockDim.y * blockIdx.y) * kCount;

        openmlsys::Tensor2D<const float> tensorA{A, M, K};
        tensorA.addOffset(m, 0);

        openmlsys::Tensor2D<const float4> tensorB{B, K, N / kCount};
        tensorB.addOffset(0, n / kCount);

        openmlsys::Tensor2D<float4> tensorC{C, M, N / kCount};
        tensorC.addOffset(m, n / kCount);

        if (!tensorC.validOffset(0, 0))
            return;

        openmlsys::float4 c[4];
        memset(c,0,sizeof(c));
        for(unsigned k=0;k<K;k++){
           float4 fragmentA{};
           for(unsigned i=0;i<kCount;i++){
            fragmentA[i]=tensorA(i,k);
           }
           float4 fragmentB=tensorB(k,0);

           for(unsigned i=0;i<kCount;++i){
            c[i]=c[i]
           }
        }
    }

}