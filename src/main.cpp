#include "nn/tensor/Tensor.h"
#include "nn/model/Model.h"
#include "nn/layer/Dense.h"

#include <iostream>

int main(int argc, char** argv)
{
    std::cout << "Welcome to the Parallel AI Inference project" << std::endl;

    auto input = Tensor<float>();
    input.set_host_data({1.0f, 2.0f, 3.0f});
    input.set_dims({1, 3});

    auto weight = Tensor<float>();
    weight.set_host_data({3.0f, 2.0f, 1.0f});
    weight.set_dims({3, 1});

    auto bias  = Tensor<float>();
    bias.set_host_data({5.0f});
    bias.set_dims({1, 1});
    
    auto dense = Dense();
    dense.set_weight(weight);
    dense.set_bias(bias);

    auto model = Model();
    model.add_layer(&dense);

    auto result = model.execute(input);
    

    std::cout << "input: "    << input.to_string(true, true, true, true);
    std::cout << "weight: "   << weight.to_string(true, true, true, true);
    std::cout << "bias: "     << bias.to_string(true, true, true, true);
    std::cout << "result: "   << result.to_string(true, true, true, true);
}