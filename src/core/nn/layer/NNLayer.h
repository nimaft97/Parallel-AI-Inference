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

#endif