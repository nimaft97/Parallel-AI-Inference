#include "NNConv2D.h"

NNConv2D::NNConv2D(dimType filters, dimType kernel_size, dimType stride, dimType padding)
    : m_filters(filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding),
      m_weights({filters, kernel_size, kernel_size}), m_biases({filters})
{
}

NNTensor NNConv2D::forward(const NNTensor& input) const
{
    NNTensor output = NNTensor(input);
    return output;
}