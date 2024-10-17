#include "nn/tensor/TensorGpuOpenCL.h"

#include <catch2/catch_all.hpp>
#include <vector>
#include <numeric>


TEST_CASE("Tensor indexing works fine", "[Tensor]")
{

    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 0.0);

    Tensor t1 = Tensor({2, 3, 4});
    t1.setData(data);

    REQUIRE(t1({0, 0, 0}) == Catch::Approx(0.0));
    REQUIRE(t1({0, 2, 3}) == Catch::Approx(11.0));
    REQUIRE(t1({1, 1, 1}) == Catch::Approx(17.0));
    REQUIRE(t1({1, 1, 2}) == Catch::Approx(18.0));
}

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