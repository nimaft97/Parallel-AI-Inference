#include "nn/tensor/TensorGpuOpenCL.h"

#include <catch2/catch_all.hpp>


TEST_CASE("Matrix multiplication on GPU works correctly", "[TensorGpuOpenCL]")
{
    TensorGpuOpenCL t1 = TensorGpuOpenCL({2, 3});
    t1.setData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    TensorGpuOpenCL t2 = TensorGpuOpenCL({3, 2});
    t2.setData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor t3 = t1.multiplyOnGpu(t2);

    REQUIRE(t3({0, 0}) == Catch::Approx(22.0));
    REQUIRE(t3({0, 1}) == Catch::Approx(28.0));
    REQUIRE(t3({1, 0}) == Catch::Approx(49.0));
    REQUIRE(t3({1, 1}) == Catch::Approx(64.0));
}