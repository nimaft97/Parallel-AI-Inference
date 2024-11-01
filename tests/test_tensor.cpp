#include "nn/tensor/Tensor.h"

#include <catch2/catch_all.hpp>
#include <vector>
#include <numeric>

TEST_CASE("Tensor indexing works fine", "[TensorIndexing]")
{

    std::vector<float> data(2 * 3 * 4);
    std::iota(data.begin(), data.end(), 0.0);

    auto t1 = Tensor<float>();
    t1.set_host_data(data);
    t1.set_dims({2, 3, 4});

    REQUIRE(t1(0, 0, 0) == Catch::Approx(0.0));
    REQUIRE(t1(0, 2, 3) == Catch::Approx(11.0));
    REQUIRE(t1(1, 1, 1) == Catch::Approx(17.0));
    REQUIRE(t1(1, 1, 2) == Catch::Approx(18.0));
}

TEST_CASE("Matrix arithmetics on host works correctly", "[MatrixOperations]")
{
    auto t1 = Tensor<float>();
    t1.set_host_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t1.set_dims({2, 3});

    auto t2 = Tensor<float>();
    t2.set_host_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t2.set_dims({3, 2});

    auto t3 = t2 + t2;
    auto t4 = t1 * t2;

    REQUIRE(t4(0, 0) == Catch::Approx(22.0));
    REQUIRE(t4(0, 1) == Catch::Approx(28.0));
    REQUIRE(t4(1, 0) == Catch::Approx(49.0));
    REQUIRE(t4(1, 1) == Catch::Approx(64.0));

    REQUIRE(t3(0, 0) == Catch::Approx(2.0));
    REQUIRE(t3(2, 0) == Catch::Approx(10.0));
    REQUIRE(t3(1, 1) == Catch::Approx(8.0));
}