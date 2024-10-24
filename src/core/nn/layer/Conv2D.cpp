#include "Conv2D.h"

Conv2D::Conv2D(dimType filters, dimType kernel_size, dimType stride, dimType padding)
    : m_filters(filters), m_kernel_size(kernel_size), m_stride(stride), m_padding(padding),
      m_weights({filters, kernel_size, kernel_size}), m_biases({filters})
{
}

Tensor Conv2D::forward(const Tensor& input) const
{
    Tensor output = Tensor(input);
    return output;
}