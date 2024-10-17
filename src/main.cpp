#include "nn/tensor/TensorGpuOpenCL.h"

#include <iostream>

int main(int argc, char** argv)
{
    std::cout << "Welcome to the Parallel AI Inference project" << std::endl;

    // create a Tensor
    TensorGpuOpenCL t1 = TensorGpuOpenCL({2, 3});
    t1.setData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    TensorGpuOpenCL t2 = TensorGpuOpenCL({3, 2});
    t2.setData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor t4 = t1.multiplyOnGpu(t2);
    const auto dims_t4 = t4.getDimensions();
    for (auto i = 0u; i < dims_t4[0]; ++i)
    {
        for (auto j= 0u; j < dims_t4[1]; ++j)
        {
            std::printf("i %u, j %u -> value %f\n", i, j, t4({i, j}));
        }
    }
}