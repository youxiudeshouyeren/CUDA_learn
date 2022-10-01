

#ifndef GEMM_UTIL_CUH
#define GEMM_UTIL_CUH
namespace openmlsys
{

template <int _m, int _n, int _k = 1>
struct Layout {
    static constexpr int m = _m;
    static constexpr int n = _n;
    static constexpr int k = _k;
};

template<typename T>
struct __device_builtin__ Tensor2D {
    T *const __restrict__ ptr;
    const unsigned rows, cols;
    int _rowOffset{0}, _colOffset{0};

    template <typemname t>
    __host__ __device__ Tensor2D(t &&ptr, unsigned rows, unsigned cols): ptr{reinterpret_cast<T *>(ptr)}, rows{rows}, cols{cols} {};


    //重载括号运算符，实现对矩阵元素的二维索引
    __host__ __device__ T &operator()(unsigned row, unsigned col)const
    {
        return ptr[_colOffset + col + (row + _rowOffset) * cols];
    }

    //加入固定偏移
    template<typemname t = T>
    __host__ __device__ void addOffset(int rowOffset, int _colOffset)
    {
        _rowOffset += rowOffset;
        _colOffset += _colOffset * sizeof(t) / sizeof(T);
    }

    //判断偏移是否越界
    __host__ __device__ bool validRowOffset(int rowOffset) const
    {
        return (_rowOffset + rowOffset) < rows;
    }

    __host__ __device__ bool validColOffset(int colOffset)const
    {
        return (_colOffset + colOffset) < cols;
    }

    __host__ __device__ bool validOffset(int rowOffset, int colOffset)const
    {
        return validColOffset(colOffset) && validRowOffset(rowOffset);
    }

};



//使用128bit构建4个float的宽数据类型
struct __device_builtin__ __builtin_align__(16) float4 {
    float data[4];


    //重载[]运算符 可以通过索引访问float4内部元素
    __host__ __device__ float operator[](unsigned idx) const
    {
        return data[idx];
    }

    __host__ __device__ float &operator[](unsigned idx)
    {
        return data[idx];
    }

    //所有元素乘以一个固定值
    __host__ __device__ float4 operator*(float other)const
    {
        return float4{data[0] *other, data[1] *other, data[2] *other, data[3] *other};
    }

    //与另一个float4进行逐元素加法
    __host__ __device__ float4 operator+(const float &other)const
    {
        return float4{data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2], data[3] + other.data[3]};
    }
};
}
#endif