#include "Dense.h"

Tensor<float> Dense::forward(const Tensor<float>& input) const
{
    // do something
    return input * m_weight + m_bias;
}

// void Dense::to_device()
// {
//     // m_weight.load_to_device();
//     // m_bias.load_to_device();
// }

// void Dense::to_host()
// {
//     // m_weight.load_to_host();
//     // m_bias.load_to_host();
// }

void Dense::set_weight(const Tensor<float>& weight)
{
    m_weight = weight;
}

void Dense::set_bias(const Tensor<float>& bias)
{
    m_bias = bias;
}