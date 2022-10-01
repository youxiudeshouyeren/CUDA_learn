#include <iostream>

void gemm(float *A, float *B, float *C,float *D, float alpha, float beta, size_t M, size_t N, size_t K)
{

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float c = 0;
            for (int k = 0; k < K; k++)
            {
                c += A[m*K+k] * B[k*N+n];
            }
             D[m*K+n]=alpha*c+beta*C[m*K+n];
        }
       
    }
    

}