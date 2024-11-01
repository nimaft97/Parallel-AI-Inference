#include "nn/tensor/Tensor.h"

#include <iostream>

int main(int argc, char** argv)
{
    std::cout << "Welcome to the Parallel AI Inference project" << std::endl;

    auto t1 = Tensor<float>();
    t1.set_host_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t1.set_dims({2, 3});

    auto t2 = Tensor<float>();
    t2.set_host_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t2.set_dims({3, 2});

    auto t3 = t2 + t2;
    auto t4 = t1 * t2;

    std::cout << "t1: " << t1.to_string(true, true, true, true);
    std::cout << "t2: " << t2.to_string(true, true, true, true);
    std::cout << "t3: " << t3.to_string(true, true, true, true);
    std::cout << "t4: " << t4.to_string(true, true, true, true);
}