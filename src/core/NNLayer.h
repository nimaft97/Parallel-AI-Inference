#ifndef NNLAYER_H
#define NNLAYER_H

#include "NNTensor.h"

class NNLayer
{
public:
    NNLayer();
    virtual ~NNLayer() = default;
    virtual NNTensor forward(const NNTensor& input) const = 0;
};

class NNDense : public NNLayer
{
    NNDense();
    NNTensor forward(const NNTensor& input) const override;
};

class NNConv2D : public NNLayer
{
    NNConv2D();
    NNTensor forward(const NNTensor& input) const override;
};

#endif