#include "Activation.h"

Activation::Activation(ACTIVATION activation): Layer()
{
    set_activation(activation);
}

void Activation::forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const
{
    // do something
    if (m_platform != input->get_platform())
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Input and Layer are not on the same platform");
    }

    if (m_activation == ACTIVATION::UNKNOWN)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::invalid_argument("Activation type is not set properly. Unknown in unhandled.");
    }

    switch(m_activation)
    {
        case ACTIVATION::RELU:
            input->relu(result1);
            break;
        case ACTIVATION::ARGMAX:
            input->argmax(result1);
            break;
        default:
            std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
            throw std::invalid_argument("Unhndled Activation type rather than UNKNOWN is encountered.");
            break;
    }
}

void Activation::to_device()
{
    m_platform = PLATFORM::DEVICE;
}

void Activation::to_host()
{
    m_platform = PLATFORM::HOST;
}

ACTIVATION Activation::get_activation() const
{
    return m_activation;
}

void Activation::set_activation(ACTIVATION activation)
{
    m_activation = activation;
}