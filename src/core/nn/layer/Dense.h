#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

class Dense : public Layer
{
public:

    virtual Tensor<float> forward(const Tensor<float>& input) const override;
    // virtual void to_device() override;
    // virtual void to_host() override;
    virtual void set_weight(const Tensor<float>& weight);
    virtual void set_bias(const Tensor<float>& bias);

protected:
    Tensor<float> m_weight;
    Tensor<float> m_bias;
};

#endif