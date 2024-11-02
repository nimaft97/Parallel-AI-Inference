#include "Dense.h"

Tensor<float> Dense::forward(const Tensor<float>& input) const
{
    // do something
    return Tensor<float>();
}

void Dense::load_to_device()
{
    // m_weight.load_to_device();
    // m_bias.load_to_device();
}

void Dense::load_to_host()
{
    // m_weight.load_to_host();
    // m_bias.load_to_host();
}