#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer
{
public:

    virtual Tensor<float> forward(const Tensor<float>& input) const override;
    virtual void load_to_device() override;
    virtual void load_to_host() override;

protected:
    Tensor<float> m_weight;
    Tensor<float> m_bias;
};

#endif