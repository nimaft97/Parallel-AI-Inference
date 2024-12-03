#ifndef RELU_H
#define RELU_H

#include "Layer.h"

class Relu : public Layer
{
public:

    virtual void forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const override;
    virtual void to_device() override;
    virtual void to_host() override;
};

#endif