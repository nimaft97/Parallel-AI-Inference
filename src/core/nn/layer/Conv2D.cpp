#include "Conv2D.h"

void Conv2D::forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const
{
    std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
    // do something
}

// void Conv2D::to_device()
// {
//     // m_weight.load_to_device();
//     // m_bias.load_to_device();
// }

// void Conv2D::to_host()
// {
//     // m_jernel.load_to_host();
// }