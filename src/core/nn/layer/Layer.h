#ifndef LAYER_H
#define LAYER_H

#include "../tensor/Tensor.h"

class Layer
{
public:
    Layer();
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) const = 0;
};

#endif