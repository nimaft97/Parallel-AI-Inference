#include "Relu.h"

void Relu::forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const
{
    // do something
    if (m_platform != input->get_platform())
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Input and Layer are not on the same platform");
    }

    input->relu(result1);
}

void Relu::to_device()
{
    m_platform = PLATFORM::DEVICE;
}

void Relu::to_host()
{
    m_platform = PLATFORM::HOST;
}
