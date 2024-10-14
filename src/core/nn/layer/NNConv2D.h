#ifndef NNCONV2D_H
#define NNCONV2D_H

#include "NNLayer.h"

class NNConv2D : public NNLayer
{
public:
    NNConv2D(dimType filters, dimType kernel_size, dimType stride, dimType padding);
    NNTensor forward(const NNTensor& input) const override;

private:
    dimType m_filters;
    dimType m_kernel_size;
    dimType m_stride;
    dimType m_padding;
    NNTensor m_weights;
    NNTensor m_biases;
};

#endif