#ifndef CONV2D_H
#define CONV2D_H

#include "Layer.h"

class Conv2D : public Layer
{
public:
    virtual Tensor<float> forward(const Tensor<float>& input) const override;
    // virtual void to_device() override;
    // virtual void to_host() override;

protected:
    Tensor<float> m_kernel;
};

#endif