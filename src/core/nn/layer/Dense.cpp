#include "Dense.h"

Tensor<float> Dense::forward(const Tensor<float>& input) const
{
    // do something
    if (m_platform != input.get_platform())
    {
        throw std::invalid_argument("Input and Layer are not on the same platform");
    }
    
    return input;
}

void Dense::to_device()
{
    if (m_platform != PLATFORM::DEVICE)
    {
        m_weight.load_to_device();
        m_bias.load_to_device();
    }
}

void Dense::to_host()
{
    if (m_platform != PLATFORM::HOST)
    {
        m_weight.load_to_host();
        m_bias.load_to_host();
    }
}

void Dense::set_weight(const Tensor<float>& weight)
{
    m_weight = weight;
}

void Dense::set_bias(const Tensor<float>& bias)
{
    m_bias = bias;
}