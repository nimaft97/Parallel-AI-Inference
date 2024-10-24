#ifndef CONV2D_H
#define CONV2D_H

#include "Layer.h"

class Conv2D : public Layer
{
public:
    Conv2D(dimType filters, dimType kernel_size, dimType stride, dimType padding);
    Tensor forward(const Tensor& input) const override;

private:
    dimType m_filters;
    dimType m_kernel_size;
    dimType m_stride;
    dimType m_padding;
    Tensor m_weights;
    Tensor m_biases;
};

#endif