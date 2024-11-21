#include "Dense.h"

void Dense::forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const
{
    // do something
    if (m_platform != input->get_platform())
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Input and Layer are not on the same platform");
    }

    input->multiply(m_weight, result2);
    result2->add(m_bias, result1);
}

void Dense::to_device()
{
    if (m_platform != PLATFORM::DEVICE)
    {
        m_weight->load_to_device();
        m_bias->load_to_device();
        m_platform = PLATFORM::DEVICE;
    }
}

void Dense::to_host()
{
    if (m_platform != PLATFORM::HOST)
    {
        m_weight->load_to_host();
        m_bias->load_to_host();
        m_platform = PLATFORM::HOST;
    }
}

void Dense::set_weight(Tensor<float>* weight)
{
    m_weight = std::move(weight);
}

void Dense::set_bias(Tensor<float>* bias)
{
    m_bias = std::move(bias);
}