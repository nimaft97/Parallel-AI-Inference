#ifndef CONV2D_H
#define CONV2D_H

#include "Layer.h"

class Conv2D : public Layer
{
public:
    virtual void forward(const Tensor<float>* input, Tensor<float>* result1, Tensor<float>* result2) const override;
    // virtual void to_device() override;
    // virtual void to_host() override;

protected:
    Tensor<float> m_kernel;
};

#endif