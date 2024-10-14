#ifndef NNLAYER_H
#define NNLAYER_H

#include "../tensor/NNTensor.h"

class NNLayer
{
public:
    NNLayer();
    virtual ~NNLayer() = default;
    virtual NNTensor forward(const NNTensor& input) const = 0;
};

class NNDense : public NNLayer
{
public:
    NNDense(dimType units);
    NNTensor forward(const NNTensor& input) const override;

private:
    dimType m_units;
    NNTensor m_weights;
    NNTensor m_biases;
};

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